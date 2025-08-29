# -*- coding: utf-8 -*-
import json
from typing import List
from pydantic import BaseModel, ValidationError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Define structured schema with Pydantic
class FusionResult(BaseModel):
    is_valid: bool
    adjusted_fusion_tokens: List[str]


class FusionValidator:
    def __init__(self, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        token = "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"
        # Load tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token
        )

    def _build_messages(self, candidate_tokens, fusion_tokens):
        return [
            {"role": "system", "content": "You are an expert in Arabic ASR fusion validation."},
            {"role": "user", "content": f"""
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
            """},
        ]

    def get_response(self, candidate_tokens, fusion_tokens) -> FusionResult:
        messages = self._build_messages(candidate_tokens, fusion_tokens)

        # Apply chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # Generate response
        outputs = self.model.generate(
            input_ids,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
        )

        # Slice only the new tokens
        response_tokens = outputs[0][input_ids.shape[-1]:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Parse JSON
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            data = json.loads(response_text[json_start:json_end])
            result = FusionResult(**data)
        except (json.JSONDecodeError, ValidationError, ValueError):
            result = FusionResult(is_valid=False, adjusted_fusion_tokens=fusion_tokens)

        # Enforce length safety
        if len(result.adjusted_fusion_tokens) != len(fusion_tokens):
            result = FusionResult(is_valid=False, adjusted_fusion_tokens=fusion_tokens)

        return result


# ================== Example Usage ==================
if __name__ == "__main__":
    validator = FusionValidator()

    candidate_tokens = [["abu", "dhabi"], ["abudhabi"], ["abu", "dhaby"]]
    fusion_tokens = ["abu", "dhabi"]

    result = validator.get_response(candidate_tokens, fusion_tokens, max_new_tokens=128)
    print(result.model_dump_json(indent=2))
