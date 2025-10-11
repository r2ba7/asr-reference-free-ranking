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
        print(32)

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
    
    def _reinforce_chunk(self, fusion_transcript, chunked_data):
        reinforced_results = []
        for chunk in chunked_data:
            validated_tokens = self._llm_reinforcement(fusion_transcript=fusion_transcript,
                                                       token_table=chunk["token_table"], 
                                                       context_before=chunk["context_before"], 
                                                       context_after=chunk["context_after"])
            reinforced_results.extend(validated_tokens)
        return reinforced_results
    
    def _llm_reinforcement(self, fusion_transcript, token_table, context_before=None, context_after=None) -> List[Dict[str, Any]]:
        def _extract_voted_tokens(token_table: str) -> list[str]:
            pattern = r"voted_token='([^']*)'"
            return re.findall(pattern, token_table)
        
        def prompt_management(token_table: str, context_before: str | None = None, context_after: str | None = None) -> str:
            base_rules = """
            You are a deterministic Arabic token selector operating on ASR ensemble outputs.
            Goal: choose the most correct token per position based on candidates, not grammar or logic inference.

            Definitions:
            - voted_token: token chosen by voting system
            - options: alternative tokens from other ASR models
            - null: deletion marker (represents no token)

            Rules:
            1. Produce EXACTLY one selection per input position.
            2. Output format:
            {"idx": N, "token": "string" OR null, "is_modified": boolean}
            3. Selection logic:
                a. If any model proposes null (deletion) at this index → output null (is_modified=true)
                b. If current index token candidates (voted_token + options) all mismatch likely context,
                    but a candidate at adjacent index (idx-1 or idx+1) clearly matches (and exists in options there),
                    shift selection from that adjacent index.
                    - If match found at idx-1 → use token from idx-1 options (is_modified=true)
                    - If match found at idx+1 → use token from idx+1 options (is_modified=true)
                c. Otherwise, pick the most frequent non-null token among (voted_token + options)
                d. If frequencies tie, prefer the voted_token
            4. Never generate or correct words — must exist exactly in voted_token or options.
            5. Do not infer meaning, grammar, or sentence structure.
            6. Keep punctuation and numbers as-is; if duplicated consecutively, delete redundant copies (null for duplicates).
            7. If a word is in english, translate it to arabic

            Output:
            {
            "selections": [
                {"idx": 0, "token": "word", "is_modified": false}
            ]
            }
            """
            if context_before or context_after:
                return (
                    f"{base_rules}\n\nfusion_transcript: {fusion_transcript}\nContext before:\n{context_before}\n\n"
                    f"Context after:\n{context_after}\n\ntoken_table:\n{token_table}\n"
                )
            return f"{base_rules}\n\nfusion_transcript: {fusion_transcript}\ntoken_table:\n{token_table}\n"
                
        prompt = prompt_management(token_table=token_table, context_before=context_before, context_after=context_after)
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
             max_tokens: int = 10, chunk_size: int = 10, overlap: int = 2) -> GeneratedResponse:
        is_chunked = self._chunk_needed(fusion_tokens=fusion_tokens, max_tokens=max_tokens)
        fusion_transcript = " ".join(
            token for token in fusion_tokens if token is not None
        )
        if is_chunked:
            chunked_data = self._prepare_chunked_operations(fusion_tokens, candidate_tokens, chunk_size, overlap)
            reinforced_results = self._reinforce_chunk(fusion_transcript, chunked_data)
        else:
            token_table = self._format_token_table(fusion_tokens=fusion_tokens, candidate_tokens=candidate_tokens, start_idx=0, end_idx=len(fusion_tokens))
            reinforced_results = self._llm_reinforcement(fusion_transcript=fusion_transcript, token_table=token_table)
        
        return GeneratedResponse(
            reinforced_results=reinforced_results,
            is_chunked=is_chunked
        )