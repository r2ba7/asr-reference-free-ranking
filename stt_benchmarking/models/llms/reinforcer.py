import json
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re

from openai import OpenAI

from stt_benchmarking.utils import decorators
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
    
    def __init__(self):
        print(26)

    def _process_chunks(
        self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
        chunk_size: int, overlap: int
    ) -> tuple[List[str], bool]:
        adjusted_tokens = {}  # Use dict keyed by position for deduplication
        any_modification = False
        start_idx = 0
        while start_idx < len(fusion_tokens):
            end_idx = min(start_idx + chunk_size, len(fusion_tokens))
            chunk_tokens, chunk_modified = self._validate_chunk(
                fusion_tokens, candidate_tokens, start_idx, end_idx
            )
            
            if chunk_modified:
                any_modification = True
            
            # Write tokens with their absolute positions
            for i, token in enumerate(chunk_tokens):
                absolute_pos = start_idx + i
                # Only write if not already written (first occurrence wins)
                if absolute_pos not in adjusted_tokens:
                    adjusted_tokens[absolute_pos] = token
            
            start_idx += chunk_size - overlap
        
        # Convert dict back to list in correct order
        final_tokens = [adjusted_tokens[i] for i in range(len(fusion_tokens))]
        return final_tokens, any_modification

    def _chunk_needed(self, fusion_tokens, max_tokens):
        if len(fusion_tokens) <= max_tokens:
            return False
        
        return True
    
    def _filter_tokens(self, tokens: List[str]) -> str:
        return " ".join(token for token in tokens if token is not None)

    def _format_token_table(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], start_idx: int, end_idx: int) -> str:
        formatted = []
        num_models = len(candidate_tokens)
        for idx in range(start_idx, min(end_idx, len(fusion_tokens))):
            fusion_tok = fusion_tokens[idx]
            fusion_tok_str = "null" if fusion_tok is None else fusion_tok
            options = []
            for m in range(num_models):
                if idx < len(candidate_tokens[m]):
                    tok = candidate_tokens[m][idx]
                    tok_str = "null" if tok is None else tok
                    if tok_str != fusion_tok_str:
                        options.append(tok_str)

            formatted.append(f"{idx}: voted_token='{fusion_tok_str}' | options={options}")
        return "\n".join(formatted)
    
    def _prepare_chunked_operations(self, fusion_tokens, candidate_tokens, chunk_size, overlap):
        """
        Create token_table, context_before, and context_after for each chunk.
        Each chunk overlaps with previous and next by `overlap` tokens.
        """
        chunked_data = []
        start = 0
        total_len = len(fusion_tokens)
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunk_start, chunk_end = start, end
            token_table = self._format_token_table(
                fusion_tokens=fusion_tokens,
                candidate_tokens=candidate_tokens,
                start_idx=chunk_start,
                end_idx=chunk_end
            )
            # define context ranges
            if chunk_start == 0:
                context_before = None
            else:
                context_before_start_idx = max(0, chunk_start - overlap)
                context_before_end_idx = chunk_start
                context_before = self._format_token_table(
                    fusion_tokens=fusion_tokens,
                    candidate_tokens=candidate_tokens,
                    start_idx=context_before_start_idx,
                    end_idx=context_before_end_idx
                )

            if chunk_end >= total_len:
                context_after = None
            else:
                context_after_start_idx = chunk_end
                context_after_end_idx = min(total_len, chunk_end + overlap)
                context_after = self._format_token_table(
                    fusion_tokens=fusion_tokens,
                    candidate_tokens=candidate_tokens,
                    start_idx=context_after_start_idx,
                    end_idx=context_after_end_idx
                )
            chunked_data.append({
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "token_table": token_table,
                "context_before": context_before,
                "context_after": context_after
            })

            start += chunk_size
        return chunked_data
    
    def _reinforce_chunk(self, chunked_data):
        reinforced_results = []
        for chunk in chunked_data:
            validated_tokens = self._llm_reinforcement(token_table=chunk["token_table"], 
                                                       context_before=chunk["context_before"], 
                                                       context_after=chunk["context_after"])
            reinforced_results.extend(validated_tokens)
        return reinforced_results
    
    def _llm_reinforcement(self, token_table, context_before=None, context_after=None) -> List[Dict[str, Any]]:
        def _extract_voted_tokens(token_table: str) -> list[str]:
            pattern = r"voted_token='([^']*)'"
            return re.findall(pattern, token_table)
        
        def prompt_management(token_table: str, context_before: str | None = None, context_after: str | None = None) -> str:
            base_rules = """
            You are an Arabic text reinforcement system for ASR ensemble outputs.
            Your task: select the most linguistically correct token at each position from the provided candidates.

            Definitions:
            - voted_token: consensus token from majority voting (primary candidate)
            - options: alternative tokens from minority models
            - JSON null (no quotes): represents token deletion/absence

            Critical constraints:
            1. Output EXACTLY one entry per input position — no more, no less
            2. Each entry format: {"idx": N, "token": "string" OR null, "is_modified": true/false}
            3. is_modified=true ONLY if you changed voted_token or set it to null
            4. is_modified=false if you kept voted_token unchanged
            5. NEVER create new words not present in voted_token or options
            6. NEVER merge tokens unless the merge candidate exists verbatim in options
            7. Use null ONLY when voted_token is linguistically invalid AND no valid option exists
            8. Preserve position indices exactly as given (0, 1, 2, ... N-1)
            9. Output must be valid JSON with no trailing commas

            Token selection priority:
            a) If voted_token is grammatically valid in context → keep it (is_modified=false)
            b) If voted_token is invalid but an option is valid → select that option (is_modified=true)
            c) If all candidates are invalid → use null (is_modified=true)

            Context usage:
            - Use context_before and context_after to resolve ambiguity between similar options
            - Do NOT use context to justify inventing tokens

            Output format:
            {
            "selections": [
                {"idx": 0, "token": "word1", "is_modified": false},
                {"idx": 1, "token": null, "is_modified": true},
                {"idx": 2, "token": "word2", "is_modified": true}
            ]
            }

            Example:
            token_table:
            0: voted_token='كفر' | options=['كفرالشيخ', 'كفر الشيخ']
            1: voted_token='الشيخ' | options=[null]

            Correct output (merge exists in options):
            {
            "selections": [
                {"idx": 0, "token": "كفرالشيخ", "is_modified": true},
                {"idx": 1, "token": null, "is_modified": true}
            ]
            }

            Incorrect output (inventing new token):
            {
            "selections": [
                {"idx": 0, "token": "كفر_الشيخ", "is_modified": true}  ← WRONG: invented token
            ]
            }
            """
            #             base_rules = """
            # You are an Arabic text reinforcement system for ASR ensemble outputs.
            # Your task: select the most linguistically correct token at each position from the provided candidates, and apply grammatical corrections where necessary.

            # Definitions:
            # - voted_token: consensus token from majority voting (primary candidate)
            # - options: alternative tokens from minority models
            # - JSON null (no quotes): represents token deletion/absence

            # Critical constraints:
            # 1. Output EXACTLY one entry per input position — no more, no less.
            # 2. Each entry format: {"idx": N, "token": "string" OR null, "is_modified": true/false, "correction_details": "description" OR null}.
            # 3. is_modified=true ONLY if you changed the voted_token (selected an option, corrected it, or set it to null).
            # 4. is_modified=false if you kept voted_token unchanged.
            # 5. **Correction Rule: You may generate a new word ONLY if it is a direct grammatical correction of a candidate (voted_token or an option). Corrections are limited to verb conjugations, gender/number agreement, or tense adjustments. You must justify this change in the "correction_details" field. Do not invent unrelated words.**
            # 6. NEVER merge tokens unless the merge candidate exists verbatim in options.
            # 7. Use null ONLY when voted_token is linguistically invalid, no valid option exists, AND no direct grammatical correction is possible.
            # 8. Preserve position indices exactly as given (0, 1, 2, ... N-1).
            # 9. Output must be valid JSON with no trailing commas.

            # Token selection priority:
            # a) If voted_token is grammatically valid in context → keep it (is_modified=false).
            # b) If voted_token is invalid but an option is valid → select that option (is_modified=true).
            # c) **If all candidates are grammatically invalid but one can be corrected → apply the correction (is_modified=true) and explain in "correction_details".**
            # d) If all candidates are invalid and cannot be corrected → use null (is_modified=true).

            # Context usage:
            # - Use context_before and context_after to resolve ambiguity and justify corrections.
            # - Do NOT use context to justify inventing unrelated tokens.

            # Output format:
            # {
            # "selections": [
            #     {"idx": 0, "token": "word1", "is_modified": false, "correction_details": null},
            #     {"idx": 1, "token": null, "is_modified": true, "correction_details": "Token deleted as it was grammatically incorrect and had no valid alternatives."},
            #     {"idx": 2, "token": "corrected_word", "is_modified": true, "correction_details": "Corrected verb conjugation from 'اتصل' to 'تصل' to match the feminine subject 'كادت'."}
            # ]
            # }
            # """
            if context_before or context_after:
                return (
                    f"{base_rules}\n\nContext before:\n{context_before}\n\n"
                    f"Context after:\n{context_after}\n\ntoken_table:\n{token_table}\n"
                )
            return f"{base_rules}\n\ntoken_table:\n{token_table}\n"
                
        prompt = prompt_management(token_table=token_table, context_before=context_before, context_after=context_after)
        print(prompt)
        try:
            completion = self.CLIENT.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "FusionReinforcer",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "selections": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "idx": {"type": "integer"},
                                            "token": {"type": ["string", "null"]},
                                            "is_modified": {"type": "boolean"}
                                        },
                                        "required": ["idx", "token", "is_modified"],
                                        "additionalProperties": False
                                    }
                                }
                            },
                            "required": ["selections"],
                            "additionalProperties": False
                        },
                    },
                },
            )

            result = json.loads(completion.choices[0].message.content)
            selections = result["selections"]
            original_chunk_tokens = _extract_voted_tokens(token_table)
            if len(selections) != len(original_chunk_tokens):
                LOGGER.warning(f"LLM returned incorrect selection length: expected {len(original_chunk_tokens)}, got {len(selections)}. Falling back.")
                return [{"idx": i, "token": tok, "is_modified": False} for i, tok in enumerate(original_chunk_tokens)]

            return selections

        except Exception as e:
            LOGGER.error(f"LLM validation failed: {e}. Falling back to fusion tokens.")
            original_chunk_tokens = _extract_voted_tokens(token_table)
            return [{"idx": i, "token": tok, "is_modified": False} for i, tok in enumerate(original_chunk_tokens)]

    def main(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
             max_tokens: int = 25, chunk_size: int = 15, overlap: int = 3) -> GeneratedResponse:
        is_chunked = self._chunk_needed(fusion_tokens=fusion_tokens, max_tokens=max_tokens)
        if is_chunked:
            chunked_data = self._prepare_chunked_operations(fusion_tokens, candidate_tokens, chunk_size, overlap)
            reinforced_results = self._reinforce_chunk(chunked_data)
        else:
            token_table = self._format_token_table(fusion_tokens=fusion_tokens, candidate_tokens=candidate_tokens, start_idx=0, end_idx=len(fusion_tokens))
            reinforced_results = self._llm_reinforcement(token_table=token_table)
        
        return GeneratedResponse(
            reinforced_results=reinforced_results,
            is_chunked=is_chunked
        )