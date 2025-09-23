import json
from typing import List, Dict, Optional
from pydantic import BaseModel
from openai import OpenAI

from stt_benchmarking.utils import decorators
from . import LOGGER


class ValidatedResponse(BaseModel):
    is_optimal: bool
    adjusted_fusion_tokens: List[Optional[str]]
    adjusted_transcript: str


class TokenValidator:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="Xxx",
    )
    
    def __init__(self):
        pass
    
    def _build_chunk_data(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
                         start_idx: int, end_idx: int) -> Dict[int, Dict]:
        """
        Build chunk data for validation.
        
        Args:
            fusion_tokens: List of fusion tokens
            candidate_tokens: List of candidate token lists
            start_idx: Starting index for the chunk
            end_idx: Ending index for the chunk
            
        Returns:
            Dictionary with token information for the chunk
        """
        chunk_data = {}
        num_models = len(candidate_tokens)
        for idx in range(start_idx, min(end_idx, len(fusion_tokens))):
            fusion_token = fusion_tokens[idx]
            candidates = set()
            for m in range(num_models):
                if idx < len(candidate_tokens[m]):
                    token = candidate_tokens[m][idx]
                    if token is not None and token != fusion_token:
                        candidates.add(token)
            
            chunk_data[idx] = {
                "fusion_token": fusion_token if fusion_token is not None else "",
                "candidate_tokens": list(candidates)
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
        # Build full sentence context
        full_sentence = " ".join([t for t in fusion_tokens if t is not None])
        prompt = f"""
        You are validating Arabic transcription tokens. All text is normalized.
        FULL FUSED SENTENCE CONTEXT: "{full_sentence}"
        Current tokens being validated (positions {start_idx} to {end_idx-1}):
        {json.dumps(chunk_data, ensure_ascii=False, indent=2)}

        For each position, select the BEST token from either the fusion_token or one of the candidate_tokens.
        IMPORTANT RULES:
        1. You can ONLY choose from the provided tokens (fusion_token or candidate_tokens)
        2. DO NOT generate new words
        3. Consider how tokens combine with adjacent ones to form proper Arabic words
        4. Prioritize meaningful place names, proper nouns, or complete phrases
        5. Ensure overall sentence coherence
        6. Look at previous and next tokens to decide if merging or nullifying creates better flow, such as combining split parts into one token and setting others to null if redundant

        If candidate_tokens is empty for a position, use the fusion_token.
        You can also output an empty string or null if after combining results, you found it's the suitable answer
        Return an array of the best token for each position in order (from position {start_idx} to {end_idx-1}).
        """
        try:
            completion = self.CLIENT.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ChunkTokenResponse",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "validated_tokens": {
                                    "type": "array",
                                    "items": {"type": ["string", "null"]}
                                }
                            },
                            "required": ["validated_tokens"],
                            "additionalProperties": False
                        }
                    }
                }
            )
            raw_output = completion.choices[0].message.content
            result = json.loads(raw_output)
            validated_tokens = result["validated_tokens"]
            
            # Ensure correct length and no None values
            expected_length = end_idx - start_idx
            if len(validated_tokens) != expected_length:
                return [t if t is not None else "" for t in fusion_tokens[start_idx:end_idx]]
            
            return [token if token is not None else "" for token in validated_tokens]
            
        except Exception as e:
            LOGGER.error(f"Validation failed: {e}")
            return [t if t is not None else "" for t in fusion_tokens[start_idx:end_idx]]
    
    def _needs_chunking(self, fusion_tokens: List[str], max_tokens: int = 20) -> bool:
        """
        Check if the fusion_tokens need to be chunked.
        
        Args:
            fusion_tokens: List of fusion tokens
            max_tokens: Maximum tokens before chunking is needed
            
        Returns:
            True if chunking is needed, False otherwise
        """
        return len(fusion_tokens) > max_tokens
    
    def _process_chunks(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
                       chunk_size: int, overlap: int) -> List[str]:
        """
        Process tokens in chunks and return validated results.
        
        Args:
            fusion_tokens: List of fusion tokens
            candidate_tokens: List of candidate token lists
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of validated tokens
        """
        adjusted_tokens = fusion_tokens.copy()
        start_idx = 0
        
        while start_idx < len(fusion_tokens):
            end_idx = min(start_idx + chunk_size, len(fusion_tokens))
            chunk_data = self._build_chunk_data(fusion_tokens, candidate_tokens, start_idx, end_idx)
            chunk_result = self._validate_chunk_data(chunk_data, fusion_tokens, start_idx, end_idx)
            for i, token in enumerate(chunk_result):
                pos = start_idx + i
                if pos < len(adjusted_tokens):
                    adjusted_tokens[pos] = token
            
            # Move to next chunk with overlap
            start_idx += chunk_size - overlap
            if start_idx >= len(fusion_tokens):
                break
            
        return adjusted_tokens
    
    def main(self, fusion_tokens: List[str], candidate_tokens: List[List[str]], 
             max_tokens: int = 20, chunk_size: int = 10, overlap: int = 2) -> ValidatedResponse:
        """
        Main method to validate tokens with optional chunking.
        
        Args:
            fusion_tokens: List of fusion tokens (may contain None)
            candidate_tokens: List of candidate token lists (may contain None)
            max_tokens: Maximum tokens before chunking is applied
            chunk_size: Size of each chunk when chunking is needed
            overlap: Overlap between chunks
            
        Returns:
            ValidatedResponse with validation results
        """
        # Clean input tokens
        if len(fusion_tokens) <= max_tokens:
            chunk_data = self._build_chunk_data(fusion_tokens, candidate_tokens, 0, len(fusion_tokens))
            validated_tokens = self._validate_chunk_data(chunk_data, fusion_tokens, 0, len(fusion_tokens))
        else:
            validated_tokens = self._process_chunks(fusion_tokens, candidate_tokens, chunk_size, overlap)
        
        # Check if changes were made
        changes_made = any(orig != new for orig, new in zip(fusion_tokens, validated_tokens))
        return ValidatedResponse(
            is_optimal=not changes_made,
            adjusted_fusion_tokens=validated_tokens,
            adjusted_transcript=" ".join(token for token in validated_tokens if token is not None)
        )