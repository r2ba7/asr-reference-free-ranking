import json
from google import genai

# initialize client
CLIENT = genai.Client(api_key="YOUR_API_KEY")

def get_response(candidate_tokens, fusion_tokens):
    content = f"""
        We have the following:
        - Candidate ASR tokens: {candidate_tokens}
        - Fusion tokens: {fusion_tokens}

        Task:
        1. Decide if the fusion tokens represent the best/most logical output ("is_valid": true/false).
        2. If adjustments are needed, return corrected tokens as "adjusted_fusion_tokens".
        3. The length of adjusted_fusion_tokens must equal the length of fusion_tokens.
        Return only valid JSON:
        {{
        "is_valid": true,
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

    # Validate required keys
    if not isinstance(result, dict) or "is_valid" not in result or "adjusted_fusion_tokens" not in result:
        raise ValueError(f"Response missing required fields: {result}")

    # Ensure adjusted_fusion_tokens length matches
    if len(result["adjusted_fusion_tokens"]) != len(fusion_tokens):
        raise ValueError(
            f"Length mismatch: expected {len(fusion_tokens)} tokens, got {len(result['adjusted_fusion_tokens'])}"
        )

    return result
