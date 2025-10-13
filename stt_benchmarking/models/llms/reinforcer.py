import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import Levenshtein  # Required for edit distance guardrail. Install with: pip install python-Levenshtein

from openai import OpenAI

from stt_benchmarking.utils import decorators
from stt_benchmarking.utils.text_processing import StandardArabicTextProcessor
from . import LOGGER

load_dotenv()

class GeneratedResponse(BaseModel):
    reinforced_results: List[Optional[Dict]]
    is_chunked: bool


class FusionReinforcer:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    # CONFIGURATION: Set a threshold for how much the LLM is allowed to change the sentence.
    # A value of 0.3 means a change of more than 30% will be rejected.
    MAX_NORMALIZED_EDIT_DISTANCE = 0.3
    MAX_RETRIES = 3

    def __init__(self):
        print(45)

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

    def _apply_guardrails(self, response: Dict, original_sentence: str, options: List[str]) -> Dict:
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

    def _llm_reinforcement(self, idx: int, formatted_sentences: Dict[str, Any],
                           context_before=None, context_after=None) -> List[Dict[str, Any]]:
        """
        [MODIFIED] Sends a more constrained prompt to a stronger model and validates the output.
        """
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
                    # CHANGE 2: Upgraded model for better reasoning and instruction-following.
                    model="google/gemini-2.5-flash",
                    # CHANGE 3: Temperature set to 0 for deterministic, repeatable outputs.
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
                                    # CHANGE 4: Added 'reasoning' to the schema to force justification.
                                    "reasoning": {"type": "string"},
                                    "sentence": {"type": ["string", "null"]},
                                    "is_modified": {"type": "boolean"},
                                },
                                "required": ["reasoning", "sentence", "is_modified"],
                            },
                        },
                    },
                )
                response = json.loads(completion.choices[0].message.content)
                validated_response = self._apply_guardrails(response, original_sentence, input_payload["options"])
                validated_response["idx"] = idx
                validated_response["metadata"] = {
                    "fusion_sentence": original_sentence,
                    "options_count": len(input_payload["options"]),
                    "has_context_before": bool(context_before),
                    "has_context_after": bool(context_after),
                }
                return [validated_response]

            except json.JSONDecodeError as e:
                LOGGER.warning(f"LLM returned malformed JSON on attempt {attempt + 1}. Error: {e}")
                if attempt >= self.MAX_RETRIES:
                    LOGGER.error("Max retries reached for malformed JSON. Falling back.")
                    break

            except Exception as e:
                LOGGER.error(f"LLM reinforcement failed on attempt {attempt + 1}: {e}. Falling back.")
                break 

        return [{"sentence": original_sentence, "is_modified": False, 
                 "reasoning": "Fell back due to persistent API/JSON errors.", "idx": idx, "metadata": {}}]


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