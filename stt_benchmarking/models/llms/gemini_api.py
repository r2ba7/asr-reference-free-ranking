import json
from google import genai
from stt_benchmarking.utils import (
    decorators, 
)
# initialize client
CLIENT = genai.Client(api_key="AIzaSyDT0QiGhz1VWyuNJ8qyBPfsAbK3iaAwlto")

def build_token_map(candidate_tokens, fusion_tokens):
    # zip(*candidate_tokens) transposes the list of lists
    return [
        {"fusion_token": fusion, "candidates": list(candidates)}
        for fusion, candidates in zip(fusion_tokens, zip(*candidate_tokens))
    ]

@decorators.Decorators.timeout_with_retry
def get_response(candidate_tokens, fusion_tokens):
    token_map = build_token_map(candidate_tokens, fusion_tokens)
    content = f"""
        You are given a list of fusion tokens with their candidate alternatives:

        {token_map}

        Task:
        1. For each fusion token, decide if the chosen token is the best/most logical among its candidates.
        2. If any fusion token is suboptimal, adjust it with a better candidate.
        3. Ensure the final "adjusted_fusion_tokens" list has the same length as the fusion tokens.

        Return only valid JSON in the following format:
        {{
        "is_optimal": true or false,
        "adjusted_fusion_tokens": ["...", "..."]
        }}
    """

    # call Gemini
    response = CLIENT.models.generate_content(
        model="gemini-2.5-flash",
        contents=content
    )

    # Extract text from the response
    raw_output = response.text.strip()

    # Try to parse JSON safely
    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        # If Gemini adds extra text, try to extract JSON with regex
        import re
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            result = json.loads(match.group(0))
        else:
            raise ValueError(f"Invalid JSON from model: {raw_output}")

    # Ensure adjusted_fusion_tokens length matches
    if len(result["adjusted_fusion_tokens"]) != len(fusion_tokens):
        raise ValueError(
            f"Length mismatch: expected {len(fusion_tokens)} tokens, got {len(result['adjusted_fusion_tokens'])}"
        )

    return result
