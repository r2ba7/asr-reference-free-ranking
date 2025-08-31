import json
from openai import OpenAI
from pydantic import BaseModel

from stt_benchmarking.utils import (
    decorators, 
)
# initialize client
CLIENT = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

class ValidatedResponse(BaseModel):
    is_optimal: bool
    adjusted_fusion_tokens: list[str]

def build_token_map(candidate_tokens, fusion_tokens):
    """
    Build a map where each element is a dictionary with index, fusion, and candidates.
    This helps LLM understand both local and global context.
    """    
    return {
        idx: {
            "fusion": fusion,
            "candidates": list(candidates)
        }
        for idx, (fusion, candidates) in enumerate(zip(fusion_tokens, zip(*candidate_tokens)))
    }

# @decorators.Decorators.timeout_with_retry
def get_response(candidate_tokens, fusion_tokens):
    token_map = build_token_map(candidate_tokens, fusion_tokens)
    prompt = f"""
    You are given an ordered sequence of tokens. Each token has:
    - Its current fusion choice.
    - A list of candidate alternatives.

    The full sequence is:
    {token_map}

    Task:
    1. Consider the **entire sentence globally**, not just locally word-by-word.
    2. For each token, decide if the chosen "fusion" word is the best and most logical in context.
    3. If any fusion token is suboptimal, replace it with a better candidate from its list.
    4. Ensure the final "adjusted_fusion_tokens" list has the same length as the original fusion tokens.
    5. Output only valid JSON in the following format:
    Every element in "adjusted_fusion_tokens" MUST be a non-empty string.
    Do not output null, None, or blank entries.
    {{
    "is_optimal": true/false,
    "adjusted_fusion_tokens": ["...", "..."]
    }}
    """
    completion = CLIENT.chat.completions.create(
        model="google/gemini-2.5-flash",   # or gpt-4o, claude-3.5, etc.
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ValidatedResponse",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_optimal": {"type": "boolean"},
                        "adjusted_fusion_tokens": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "minLength": 1,
                                "nullable": False  # prevents empty string too
                            }
                        }
                    },
                    "required": ["is_optimal", "adjusted_fusion_tokens"],
                    "additionalProperties": False
                }
            }
        }
    )

    # Model’s output is guaranteed to follow the schema
    raw_output = completion.choices[0].message.content
    result = ValidatedResponse(**json.loads(raw_output))
    cleaned_tokens = []
    for i, tok in enumerate(result.adjusted_fusion_tokens):
        if tok is None:
            cleaned_tokens.append(fusion_tokens[i])
        else:
            cleaned_tokens.append(str(tok))  # force string in case of int/float
    result.adjusted_fusion_tokens = cleaned_tokens

    if len(result.adjusted_fusion_tokens) != len(fusion_tokens):
        result.adjusted_fusion_tokens = fusion_tokens

    return result
