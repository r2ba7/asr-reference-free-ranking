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
        print(21)

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
            options = {fusion_tokens[idx]}
            fusion_tok = fusion_tokens[idx]
            options = []
            for m in range(num_models):
                if idx < len(candidate_tokens[m]):
                    tok = candidate_tokens[m][idx] or "Null"
                    if tok != fusion_tok:
                        options.append(tok)
            formatted.append(f"{idx}: voted_token='{fusion_tok}' | options={options}")
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
                أنت نظام تعزيز وتصحيح نصوص عربية لناتج التعرف التلقائي على الكلام (ASR).
                مهمتك: بناء نص عربي صحيح لغوياً ونحوياً من الخيارات المتاحة.

                مصطلحات:
                - voted_token: الكلمة المتفق عليها بعد عملية التصويت (الخيار الأول في كل موضع).
                - options: بدائل النماذج الأخرى. إن كانت فارغة → جميع النماذج متفقة على voted_token.
                - استخدم JSON null (بدون علامات اقتباس) لتمثيل حذف/غياب كلمة.

                قواعد ملزمة:
                1. لكل موضع في token_table يجب أن تُعاد قيمة واحدة فقط: إما token (سلسلة) أو null.
                2. لا تُغيّر ترتيب المواقع أو تُحذف موضعاً دون تعويض (كل موقع له قرار).
                3. لا تُنشئ كلمات جديدة خارج الخيارات إلا في حالة **تصحيح تجميعي مقبول لغوياً** (مثل دمج 'كفر' + 'الشيخ' → 'كفرالشيخ') — هذا مسموح فقط عندما يكون واضحاً لغوياً.
                4. الخيار الأول في كل موضع هو voted_token — فضّل الحفاظ عليه إذا كان صحيحاً لغوياً.
                5. استخدم المعرفة السياقية (السياق قبل/بعد) عند الحاجة إذا الخيارات متقاربة.
                6. يمكن إخراج null فقط عند وجود خطأ واضح أو تكرار أو عدم قابلية أي خيار للاستمرار.
                7. لكل selection أعد أيضا حقل is_modified: true إذا اختفت الكلمة أو تغيّرت (أي إذا الاختيار ليس مساويًا لـ voted_token)، وإلا false.
                8. أعِد أيضاً مجموعياً حقل is_modified (true/false) يبيّن إن حصل أي تعديل عبر المقطع.

                الناتج المطلوب (صيغة JSON، استخدم null للغياب):
                {
                "selections":[
                    {"position": 0, "token": "كلمة", "is_modified": false},
                    {"position": 1, "token": null, "is_modified": true},
                    ...
                ],
                "is_modified": true|false
                }

                مثال مبسّط:
                token_table:
                0: voted_token='كفر' | options=['كفرالشيخ', 'كفر الشيخ']
                1: voted_token='الشيخ' | options=['Null']

                مخرجات صالحة:
                {
                "selections":[
                    {"position":0, "token":"كفرالشيخ", "is_modified": true},
                    {"position":1, "token": null, "is_modified": true}
                ],
                "is_modified": true
                }
                """
            if context_before is not None or context_after is not None:
                return f"{base_rules}\n\nالسياق قبل المقطع:\n{context_before}\n\nالسياق بعد المقطع:\n{context_after}\n\nخيارات الكلمات:\n{token_table}\n"
            else:
                return f"{base_rules}\n\nخيارات الكلمات:\n{token_table}\n" 
                
        prompt = prompt_management(token_table=token_table, context_before=context_before, context_after=context_after)
        try:
            completion = self.CLIENT.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ValidatedChunkResponse",
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
                                        "required": ["idx", "token", "is_modified"]
                                    }
                                }
                            },
                            "required": ["selections"],
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
             max_tokens: int = 30, chunk_size: int = 20, overlap: int = 3) -> GeneratedResponse:
        is_chunked = self._chunk_needed(fusion_tokens=fusion_tokens, max_tokens=max_tokens)
        LOGGER.info(f"is chunked: {is_chunked}")
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
    

class TranscriptGenerator:
    CLIENT = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    def __init__(self):
        print(6)

    def _filter_tokens(self, tokens: List[str]) -> str:
        return " ".join(token for token in tokens if token is not None)

    def _format_token_table(
        self, fusion_tokens: List[str], candidate_tokens: List[List[str]]
    ) -> str:
        num_models = len(candidate_tokens)
        formatted = []

        for idx, f_tok in enumerate(fusion_tokens):
            opts = [f_tok]
            for m in range(num_models):
                if idx < len(candidate_tokens[m]):
                    opts.append(candidate_tokens[m][idx])
            seen = set()
            opts = [o for o in opts if not (o in seen or seen.add(o))]
            formatted.append(f"{idx+1}. options = {opts}")

        return "\n".join(formatted)

    def _generate_corrected_transcript(
        self, fusion_transcript: str, fusion_tokens: List[str], candidate_tokens: List[List[str]]
    ) -> tuple[str, bool]:
        token_table = self._format_token_table(fusion_tokens, candidate_tokens)
        prompt = f"""
            أنت نظام تعزيز وتصحيح نصوص عربية لناتج التعرف التلقائي على الكلام (ASR).
            مهمتك: بناء نص عربي صحيح لغوياً ونحوياً من الخيارات المتاحة.

            القواعد:
            1. لكل موضع، اختر كلمة واحدة فقط من الخيارات المتاحة
            2. الخيار الأول في كل موضع هو نتيجة التصويت الأكثري
            3. إذا كان الخيار الأول صحيحاً لغوياً، اختره وأعِد fusion_modified: false
            4. إذا كان الخيار الأول خاطئاً، اختر البديل الصحيح من الخيارات الأخرى وأعِد fusion_modified: true
            5. لا تُنشئ كلمات جديدة، لا تغيّر الترتيب، لا تحذف أو تضيف مواضع

            خيارات الكلمات (الخيار الأول = نتيجة التصويت):
            {token_table}

            أعِد كائن JSON بهذا الشكل:
            {{"corrected_transcript": "<النص النهائي>", "fusion_modified": true/false}}
            """

        try:
            completion = self.CLIENT.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CorrectedTranscriptResponse",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "corrected_transcript": {"type": "string"},
                                "fusion_modified": {"type": "boolean"}
                            },
                            "required": ["corrected_transcript", "fusion_modified"],
                        },
                    },
                },
            )

            result = json.loads(completion.choices[0].message.content)
            corrected_transcript = result["corrected_transcript"].strip()
            fusion_modified = result["fusion_modified"]
            if not corrected_transcript:
                LOGGER.warning("LLM returned empty transcript. Falling back.")
                return fusion_transcript, False

            return corrected_transcript, fusion_modified

        except Exception as e:
            LOGGER.error(f"LLM generation failed: {e}. Falling back to fusion transcript.")
            return fusion_transcript, False

    def main(
        self, fusion_transcript: str, fusion_tokens: List[str], candidate_tokens: List[List[str]]
    ) -> GeneratedResponse:
        corrected_transcript, fusion_modified = self._generate_corrected_transcript(
            fusion_transcript, fusion_tokens, candidate_tokens
        )
        adjusted_tokens = corrected_transcript.split() if corrected_transcript else []

        return GeneratedResponse(
            original_fusion_tokens=fusion_tokens,
            original_fusion_transcript=fusion_transcript,
            adjusted_fusion_tokens=adjusted_tokens,
            adjusted_transcript=corrected_transcript,
            fusion_modified=fusion_modified
        )

