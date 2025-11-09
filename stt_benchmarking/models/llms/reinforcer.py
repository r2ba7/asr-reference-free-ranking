import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import Levenshtein  # Required for edit distance guardrail. Install with: pip install python-Levenshtein
import re
from collections import Counter

from openai import OpenAI

from stt_benchmarking.utils import decorators
from stt_benchmarking.utils.text_processing import StandardArabicTextProcessor
from . import LOGGER

load_dotenv()

class GeneratedResponse(BaseModel):
    reinforced_results: List[Optional[Dict]]
    is_chunked: bool


class TokenReinforcer:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ) 
    MAX_RETRIES = 0

    def __init__(self):
        print(1)

    def extract_candidates_at_position(self, alignment_results, position):
        """
        Extract all tokens at position including None values.
        
        Args:
            alignment_results: list of alignment dicts
            position: index position
            
        Returns:
            list of dicts with token and metadata
        """
        candidates = []
        for result in alignment_results:
            token = result['tokens'][position]
            candidates.append({
                'token': token,
                'model_index': result['model_index'],
                'is_reference': result['is_reference'],
                'operation': result['operations'][position]
            })
        
        return candidates
    
    def _extract_candidates_for_selection(self, candidates):
        """
        Extract tokens for LLM selection, keeping None to represent no token.
        
        Args:
            candidates: list of candidate dicts
            
        Returns:
            dict with tokens list and metadata
        """
        # Extract all tokens including None
        all_tokens = [c['token'] for c in candidates]
        token_counts = Counter(all_tokens)
        
        # Check for reference
        reference_token = None
        has_reference = False
        for c in candidates:
            if c['is_reference']:
                reference_token = c['token']
                has_reference = True
                break
        
        # Get unique tokens (None included)
        unique_tokens = list(token_counts.keys())
        return {
            'unique_tokens': unique_tokens,
            'has_reference': has_reference,
            'reference_token': reference_token,
            'token_counts': dict(token_counts),
            'total_models': len(candidates),
            'has_none': None in unique_tokens,
            'none_count': token_counts.get(None, 0)
        }
        
    def _llm_token_selection(self, position, candidate_info):
        """
        Use LLM to select best token from candidates based on reference and votes.
        
        Args:
            position: current position index
            candidate_info: dict from _extract_candidates_for_selection
            
        Returns:
            dict with selection result and metadata
        """
        
        unique_tokens = candidate_info['unique_tokens']
        
        # Handle cases with 0 or 1 unique candidate
        if not unique_tokens:
            return {
                "position": position,
                "selected_token": None,
                "metadata": {
                    "candidate_info": candidate_info, 
                    "reason": "No candidates available"
                }
            }
        
        if len(unique_tokens) == 1:
            return {
                "position": position,
                "selected_token": unique_tokens[0],
                "metadata": {
                    "candidate_info": candidate_info, 
                    "reason": "Single unique candidate - no selection needed"
                }
            }
        
        base_rules = """
            You are a precise token selector for Arabic ASR fusion outputs. Your only task is to select the single best word from a list of noisy ASR candidates. You are acting as a voter.

            **Rules:**
            1. You MUST select exactly ONE option from the `candidates` list. The candidate `None` (represented as `null` in JSON) is a valid choice and means "select no word" (a deletion).
            2. Your selected word MUST be one of the provided candidates - no modifications.
            3. A `reference_token` is provided. This token is considered highly reliable. You should select the `reference_token` as the default choice.
            4. **CRITICAL EXCEPTION:** Only deviate from the `reference_token` if it is *clearly* and *unambiguously* incorrect (e.g., a nonsensical word) AND another candidate from the list is a *much* better fit.
            5. If `reference_token` is `null`, select the best candidate from the list, considering the `token_counts` as votes. The highest count is a strong signal.

            Input JSON:
            {
            "candidates": ["<word1>", "<word2>", null, ...],
            "reference_token": "<word_or_null>",
            "token_counts": {"<word1>": <count>, "<word2>": <count>, "null": <count>, ...}
            }

            Output JSON (Strict Schema):
            {
            "selected_token": "<exact word from candidates or null>"
            }
        """
        
        # json.dumps converts Python None to JSON null automatically
        input_payload = {
            "candidates": candidate_info['unique_tokens'],
            "reference_token": candidate_info['reference_token'],
            "token_counts": candidate_info['token_counts']
        }
        
        prompt = f"{base_rules}\n\nInput:\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
        
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                completion = self.CLIENT.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "TokenSelector",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "selected_token": {"type": ["string", "null"]}
                                },
                                "required": ["selected_token"],
                                "additionalProperties": False
                            }
                        }
                    }
                )
                
                response_text = completion.choices[0].message.content
                response = json.loads(response_text)
                # Validate that the LLM's choice is one of the unique tokens
                if response["selected_token"] not in unique_tokens:
                    LOGGER.warning(f"LLM selected invalid token '{response['selected_token']}' at pos {position}. Falling back.")
                    
                    # FALLBACK 1: Trust the reference
                    if candidate_info['has_reference']:
                        response["selected_token"] = candidate_info['reference_token']
                        reason = "Fallback: LLM selection was invalid. Using reference_token."
                    else:
                        # FALLBACK 2: No reference, use first unique token
                        response["selected_token"] = unique_tokens[0]
                        reason = "Fallback: LLM selection was invalid. No reference, using first unique candidate."
                else:
                    reason = "LLM selection" # Valid selection
                
                return {
                    "position": position,
                    "selected_token": response["selected_token"],
                    "metadata": {
                        "candidate_info": candidate_info,
                        "reason": reason
                    }
                }
                
            except json.JSONDecodeError as e:
                LOGGER.warning(f"LLM returned malformed JSON on attempt {attempt + 1} at pos {position}: {e}")
                if attempt >= self.MAX_RETRIES:
                    break
                    
            except Exception as e:
                LOGGER.error(f"LLM token selection failed on attempt {attempt + 1} at pos {position}: {e}")
                if attempt >= self.MAX_RETRIES:
                    break
        
        # FINAL FALLBACK (API/JSON errors): Trust the reference
        if candidate_info['has_reference']:
            fallback_token = candidate_info['reference_token']
            fallback_reason = "Fallback: API/JSON errors. Using reference_token."
        else:
            fallback_token = unique_tokens[0]
            fallback_reason = "Fallback: API/JSON errors. No reference, using first unique candidate."
            
        return {
            "position": position,
            "selected_token": fallback_token,
            "metadata": {
                "candidate_info": candidate_info,
                "prompt": prompt,
                "reason": fallback_reason
            }
        }

    def main(self, alignment_results):
        if not alignment_results:
            return {
                "selected_tokens": [],
                "selection_metadata": []
            }
        
        sequence_length = len(alignment_results[0]['tokens'])
        selected_tokens = []
        selection_metadata = []
        
        for position in range(sequence_length):
            # 1. Get all candidate data for the current position
            current_candidates_data = self.extract_candidates_at_position(alignment_results, position)
            
            # 2. Process data to get unique tokens, reference, and counts
            candidate_info = self._extract_candidates_for_selection(current_candidates_data)
            
            # 3. Pass this structured info to the LLM voter
            selection_result = self._llm_token_selection(
                position=position,
                candidate_info=candidate_info
            )
            
            # (Optional) Print statements for debugging
            # print(f"--- Position {position} ---")
            # print(f"Candidate Info: {candidate_info}")
            # print(f"Selection Result: {selection_result['selected_token']} (Reason: {selection_result['reasoning']})")
            selected_tokens.append(selection_result["selected_token"])
            selection_metadata.append(selection_result)

        return {
            "selected_tokens": selected_tokens,
            "selection_metadata": selection_metadata
        }

class FusionReinforcer:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    # CONFIGURATION: Set a threshold for how much the LLM is allowed to change the sentence.
    # A value of 0.3 means a change of more than 33% will be rejected.
    MAX_NORMALIZED_EDIT_DISTANCE = 0.33
    MAX_RETRIES = 3

    # No changes needed for _chunk_needed, _format_sentence_table, _prepare_chunked_operations
    # These methods correctly handle data preparation.
    def _chunk_needed(self, fusion_tokens, max_tokens):
        if len(fusion_tokens) <= max_tokens:
            return False
        return True

    def _format_sentence_table(self, fusion_tokens: List[str], candidate_tokens: List[List[str]],
                               start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Return structured JSON input with fusion_sentence + options."""
        fusion_sentence = " ".join(tok for tok in fusion_tokens[start_idx:end_idx] if tok)
        options = []
        for candidate in candidate_tokens:
            candidate_sentence = " ".join(tok for tok in candidate[start_idx:end_idx] if tok)
            if candidate_sentence and candidate_sentence != fusion_sentence:
                options.append(candidate_sentence)
        return {"fusion_sentence": fusion_sentence, "options": options}

    def _prepare_chunked_operations(self, fusion_tokens, candidate_tokens, chunk_size, overlap):
        chunked_data = []
        total_len = len(fusion_tokens)
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            formatted_sentences = self._format_sentence_table(fusion_tokens, candidate_tokens, start, end)
            context_before = None
            if start > 0:
                context_before = self._format_sentence_table(fusion_tokens, candidate_tokens,
                                                             max(0, start - overlap), start)
            context_after = None
            if end < total_len:
                context_after = self._format_sentence_table(fusion_tokens, candidate_tokens,
                                                            end, min(total_len, end + overlap))
            chunked_data.append(
                {
                    "formatted_sentences": formatted_sentences,
                    "context_before": context_before,
                    "context_after": context_after,
                }
            )
            start += chunk_size
        return chunked_data

    def _reinforce_chunk(self, chunked_data):
        results = []
        for i, chunk in enumerate(chunked_data):
            res = self._llm_reinforcement(idx=i,
                                          formatted_sentences=chunk["formatted_sentences"],
                                          context_before=chunk["context_before"],
                                          context_after=chunk["context_after"]
                                          )
            results.extend(res)
        return results

    def _llm_reinforcement(self, idx: int, formatted_sentences: Dict[str, Any],
                           context_before=None, context_after=None) -> List[Dict[str, Any]]:
        """
        [MODIFIED] Sends a more constrained prompt to a stronger model and validates the output.
        """
        def _apply_guardrails(response: Dict, original_sentence: str, options: List[str]) -> Dict:
            """
            [REVISED] Validates the LLM's output against programmatic rules.
            Assumes original_sentence and options are pre-normalized.
            Normalizes the LLM's output before comparison.
            """
            if not response.get("is_modified") or not response.get("sentence"):
                return response

            # STEP 1: Normalize the untrusted LLM output. This is the only normalization needed.
            new_sentence = StandardArabicTextProcessor.main(response["sentence"])
            response["sentence"] = new_sentence # Update the response with the clean version.

            # Guardrail 1: Vocabulary Check
            # Builds the allowed vocabulary from the already-normalized inputs.
            allowed_tokens = set(original_sentence.split())
            for opt in options:
                allowed_tokens.update(opt.split())
            
            output_tokens = set(new_sentence.split())
            if not output_tokens.issubset(allowed_tokens):
                LOGGER.warning("Guardrail Triggered: LLM hallucinated new tokens (post-normalization). Reverting.")
                response["sentence"] = original_sentence
                response["is_modified"] = False
                response["reasoning"] = "Rejected by vocabulary guardrail."
                return response

            # Guardrail 2: Edit Distance Threshold
            if not original_sentence: return response
            
            distance = Levenshtein.distance(original_sentence, new_sentence)
            normalized_distance = distance / max(len(original_sentence), len(new_sentence))
            if normalized_distance > self.MAX_NORMALIZED_EDIT_DISTANCE:
                LOGGER.warning(f"Guardrail Triggered: Edit distance ({normalized_distance:.2f}) exceeded threshold. Reverting.")
                response["sentence"] = original_sentence
                response["is_modified"] = False
                response["reasoning"] = "Rejected by edit distance guardrail."
                return response
                
            return response
        
        def _salvage_broken_json(malformed_string: str) -> dict | None:
            """
            Attempts to repair a malformed JSON string from an LLM.
            Specifically designed to handle unterminated strings by looking for known keys
            and extracting the content that follows them.
            """
            try:
                salvaged_data = {}
                
                # 1. Extract 'reasoning'
                # Pattern looks for "reasoning":" and captures everything until the next key ("selected_token")
                reasoning_match = re.search(r'"reasoning"\s*:\s*"(.+?)(?=","selected_token"|})', malformed_string, re.DOTALL)
                if reasoning_match:
                    salvaged_data['reasoning'] = reasoning_match.group(1).strip()

                # 2. Extract 'selected_token' (this is often the broken one)
                # Pattern looks for "selected_token":" and captures everything until end or closing brace
                token_match = re.search(r'"selected_token"\s*:\s*"(.+?)(?="|})', malformed_string, re.DOTALL)
                if token_match:
                    # Clean up potential trailing characters if the string was unterminated
                    salvaged_data['selected_token'] = token_match.group(1).strip().rstrip('"').rstrip(',')

                # 3. Validate the salvaged data
                if all(key in salvaged_data for key in ['reasoning', 'selected_token']):
                    LOGGER.warning("Successfully salvaged a broken JSON response.")
                    return salvaged_data
                    
            except Exception as e:
                LOGGER.error(f"Salvage operation failed: {e}")
            
            return None
        

        # CHANGE 1: The prompt is now highly restrictive, focusing only on surgical corrections.
        base_rules = """
            You are a high-fidelity, deterministic error corrector for Arabic ASR ensemble outputs. Your function is to perform cautious, surgical corrections based on a "do no harm" principle.

            **Core Context:**
            - This text is a transcript of a spoken, informal conversation. It must be an exact reflection of what was said, even if grammatically imperfect.
            - The `fusion_sentence` is your baseline and is **presumed to be the correct transcription**.

            **Decision Framework:**
            1.  **Identify Conflicts**: Compare `options` against the `fusion_sentence`.
            2.  **Evaluate for Improvement**: An improvement must be a **near-certain correction of an obvious ASR error**, not a stylistic or grammatical enhancement.
            3.  **Default to Baseline**: If you have **any doubt whatsoever**, return the original `fusion_sentence`.

            **Strict Rules:**
            1.  **Substitution Threshold (Very High)**: Only substitute a word if multiple `options` agree on a better alternative that fixes a clear grammatical or logical error.
            
            2.  **Deletion (Last Resort)**: Deletions are strongly forbidden unless correcting a clear ASR stutter (e.g., "ال ال") that is also omitted by the majority of `options`.
            
            3.  **Insertion (Strictly Forbidden for Fluency)**: Your most common error is adding words to improve grammar. This is not allowed.
                - You are FORBIDDEN from inserting words to make a sentence more fluent or grammatically complete.
                - **Example**: If `fusion_sentence` is "انا اذهب مدرسه" and an `option` is "انا اذهب الى المدرسه", you MUST NOT add "الى" or "ال". The goal is to transcribe what was said, not to fix the speaker's grammar.
                - The only time an insertion is allowed is to fix a clear word omission that makes the sentence meaningless, and this fix must be strongly supported by the `options`.

            4.  **No Hallucinations**: Every single word in your final `sentence` MUST exist in either the original `fusion_sentence` or one of the `options`.

            5.  **Reasoning**: Provide a brief, token-level justification for any change. If no change is made, state "No correction needed; baseline is optimal."

            Input JSON:
            {
              "fusion_sentence": "<string>",
              "options": ["<string>", ...],
              "context_before": "<string or null>",
              "context_after": "<string or null>"
            }

            Output JSON (Strict Schema):
            {
              "reasoning": "<string: Justify the change or state no change needed.>",
              "sentence": "<string>" OR null,
              "is_modified": true|false
            }
        """
        input_payload = {
            "fusion_sentence": formatted_sentences.get("fusion_sentence"),
            "options": formatted_sentences.get("options", []),
            "context_before": context_before,
            "context_after": context_after,
        }
        prompt = f"{base_rules}\n\nInput:\n{json.dumps(input_payload, ensure_ascii=False, indent=2)}"
        original_sentence = formatted_sentences.get("fusion_sentence", "")
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                completion = self.CLIENT.chat.completions.create(
                    model="google/gemini-2.5-flash",
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "SentenceCorrector",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "reasoning": {"type": "string"},
                                    "sentence": {"type": ["string", "null"]},
                                    "is_modified": {"type": "boolean"},
                                },
                                "required": ["reasoning", "sentence", "is_modified"],
                                "additionalProperties": False
                            },
                        },
                    },
                )
                response_text = completion.choices[0].message.content
                response = json.loads(response_text)
                validated_response = _apply_guardrails(response, original_sentence, input_payload["options"])
                validated_response["idx"] = idx
                validated_response["metadata"] = {
                    "fusion_sentence": original_sentence,
                    "options": input_payload["options"],
                    "options_count": len(input_payload["options"]),
                    "context_before": context_before,
                    "context_after": context_after,
                }
                return [validated_response]

            except json.JSONDecodeError as e:
                LOGGER.warning(f"LLM returned malformed JSON on attempt {attempt + 1}. Error: {e}")
                salvaged_response = _salvage_broken_json(response_text)
                if salvaged_response:
                    LOGGER.info("Successfully salvaged the broken JSON.")
                    validated_response = _apply_guardrails(salvaged_response, original_sentence, input_payload["options"])
                    validated_response["idx"] = idx
                    validated_response["metadata"] = {
                        "fusion_sentence": original_sentence,
                        "options": input_payload["options"],
                        "options_count": len(input_payload["options"]),
                        "context_before": context_before,
                        "context_after": context_after,
                    }
                    return [validated_response]
                
                if attempt >= self.MAX_RETRIES:
                    LOGGER.error("Max retries reached for malformed JSON. Falling back.")
                    break

            except Exception as e:
                LOGGER.error(f"LLM reinforcement failed on attempt {attempt + 1}: {e}. Falling back.")
                break 

        return [{
            "sentence": original_sentence, 
            "is_modified": False, 
            "reasoning": "Fell back due to persistent API/JSON errors.", 
            "idx": idx, 
            "metadata": {
                "fusion_sentence": original_sentence,
                "options": input_payload["options"],
                "options_count": len(input_payload["options"]),
                "context_before": context_before,
                "context_after": context_after,
            }
        }]


    def main(self, fusion_tokens: List[str], candidate_tokens: List[List[str]],
             max_tokens: int = 10, chunk_size: int = 10, overlap: int = 2) -> GeneratedResponse:
        is_chunked = self._chunk_needed(fusion_tokens, max_tokens)
        if is_chunked:
            chunked_data = self._prepare_chunked_operations(fusion_tokens, candidate_tokens, chunk_size, overlap)
            reinforced_results = self._reinforce_chunk(chunked_data)
        else:
            formatted_sentences = self._format_sentence_table(fusion_tokens, candidate_tokens, 0, len(fusion_tokens))
            reinforced_results = self._llm_reinforcement(idx=0, formatted_sentences=formatted_sentences)

        return GeneratedResponse(
            reinforced_results=reinforced_results,
            is_chunked=is_chunked,
        )