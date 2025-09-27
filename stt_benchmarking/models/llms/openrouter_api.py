import json
from typing import List, Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from openai import OpenAI

from stt_benchmarking.utils import decorators
from . import LOGGER

load_dotenv()

class ValidatedResponse(BaseModel):
    original_fusion_tokens: List
    original_fusion_transcript: str
    adjusted_fusion_tokens: List[Optional[str]]
    adjusted_transcript: str


class TokenValidator:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    def __init__(self):
        print(11)
    
    def _filter_tokens(self, tokens: List[str]) -> str:
        """Extract non-None tokens and join into string."""
        return " ".join(token for token in tokens if token is not None)

    # use all tokens, dont edit any token
    def _build_chunk_data(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
                          start_idx: int, end_idx: int) -> Dict[int, Dict]:
        """Build chunk data for validation."""
        chunk_data = {}
        num_models = len(candidate_tokens)
        for idx in range(start_idx, min(end_idx, len(fusion_tokens))):
            candidates = []
            for m in range(num_models):
                if idx < len(candidate_tokens[m]):
                    candidates.append(candidate_tokens[m][idx])
            
            chunk_data[idx] = {
                "fusion_token": fusion_tokens[idx],
                "candidate_tokens": candidates
            }
        
        return chunk_data
    
    def _validate_chunk_data(self, chunk_data: Dict[int, Dict], fusion_tokens: List[str], 
                             start_idx: int, end_idx: int) -> List[str]:
        """
        Validate chunk data using LLM.
        
        Args:
            chunk_data: Dictionary with token information for the chunk
            fusion_tokens: Complete list of fusion tokens
            start_idx: Starting index for validation
            end_idx: Ending index for validation
            
        Returns:
            List of validated tokens for the chunk
        """
        context_before = " ".join(token for token in fusion_tokens[max(0, start_idx - 10):start_idx] if token is not None)
        context_after = " ".join(token for token in fusion_tokens[end_idx:end_idx + 10] if token is not None)
        
        # 2. Define original tokens for fallback
        original_chunk_tokens = fusion_tokens[start_idx:end_idx]

        # **FIX**: Convert the input data from a dictionary to a list of objects.
        # This makes the mapping task much easier for the LLM.
        chunk_data_as_list = [
            {
                "position": pos,
                "fusion_token": data["fusion_token"],
                "candidate_tokens": data["candidate_tokens"]
            }
            for pos, data in chunk_data.items()
        ]

        # 3. Construct the prompt with the NEW list-based input structure
        prompt = f"""
        You are an expert system for correcting Arabic ASR transcripts. Your task is to optimize a sequence of tokens with extreme care.

        CONTEXT BEFORE CHUNK: "{context_before}"
        CONTEXT AFTER CHUNK: "{context_after}"

        ---
        ## INSTRUCTIONS:
        1.  **Primary Goal (Selection)**: For each JSON object in the input array, select the single best token from its "fusion_token" and "candidate_tokens".
        2.  **Secondary Action (Deletion)**: You may replace a token with `null` **only if it is a clear and obvious error**, such as a nonsensical word, a stutter, or a hallucinated repetition.
        3.  **Conservative Rule**: **When in doubt, you must preserve the token over deleting it.** It is better to keep a slightly awkward word than to delete a potentially correct one.
        4.  **Strict Constraint**: You **MUST NOT** introduce new words.
        5.  **Output Format**: Your response must be a single JSON object with the key "optimized_tokens", containing an array of the selected tokens (strings or `null`). The output array's length **must exactly match** the input array's length.

        ---
        ## INPUT TOKEN DATA (Array):
        {json.dumps(chunk_data_as_list, ensure_ascii=False, indent=2)}

        ---
        Return the JSON object containing the optimized token array:
        """
        try:
            completion = self.CLIENT.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "OptimizedChunkResponse",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "optimized_tokens": {
                                    "type": "array",
                                    "items": {"type": ["string", "null"]}
                                }
                            },
                            "required": ["optimized_tokens"],
                        }
                    }
                }
            )
            
            result = json.loads(completion.choices[0].message.content)
            optimized_tokens = result["optimized_tokens"]

            if len(optimized_tokens) != len(original_chunk_tokens):
                LOGGER.warning(f"LLM returned list of incorrect length. Expected {len(original_chunk_tokens)}, got {len(optimized_tokens)}. Falling back.")
                return original_chunk_tokens

            return optimized_tokens

        except Exception as e:
            LOGGER.error(f"LLM optimization for chunk failed: {e}. Falling back to original tokens.")
            return original_chunk_tokens
    
    def _process_chunks(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
                        chunk_size: int, overlap: int) -> List[str]:
        """Process tokens in chunks and return validated results."""
        adjusted_tokens = fusion_tokens.copy()
        start_idx = 0
        
        while start_idx < len(fusion_tokens):
            end_idx = min(start_idx + chunk_size, len(fusion_tokens))
            chunk_result = self._validate_tokens(fusion_tokens, candidate_tokens, start_idx, end_idx)
            for i, token in enumerate(chunk_result):
                adjusted_tokens[start_idx + i] = token
            
            start_idx += chunk_size - overlap
        
        return adjusted_tokens
    
    def _validate_tokens(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
                        start_idx: int, end_idx: int) -> List[str]:
        """Validate a range of tokens."""
        chunk_data = self._build_chunk_data(fusion_tokens, candidate_tokens, start_idx, end_idx)
        return self._validate_chunk_data(chunk_data, fusion_tokens, start_idx, end_idx)
    
    def main(self, fusion_transcript: str, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
            max_tokens: int = 20, chunk_size: int = 10, overlap: int = 2) -> ValidatedResponse:
        """Main method to validate tokens with optional chunking."""
        if len(fusion_tokens) <= max_tokens:
            validated_tokens = self._validate_tokens(fusion_tokens, candidate_tokens, 0, len(fusion_tokens))
        else:
            validated_tokens = self._process_chunks(fusion_tokens, candidate_tokens, chunk_size, overlap)
            
        validated_tokens = [
            None if isinstance(token, str) and token.upper() in ["NULL", "NONE"] else token
            for token in validated_tokens
        ]
        return ValidatedResponse(
            original_fusion_tokens=fusion_tokens,
            original_fusion_transcript=fusion_transcript,
            adjusted_fusion_tokens=validated_tokens,
            adjusted_transcript=self._filter_tokens(validated_tokens)
        )