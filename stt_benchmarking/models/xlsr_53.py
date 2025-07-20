import re

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from . import LOGGER
from stt_benchmarking.utils import metrics

def clean_arabic_text(text: str) -> str:
    """
    Cleans Arabic text by:
    - Removing Arabic diacritics
    - Stripping redundant whitespace
    - Returning a clean string
    """
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model(device, LANG_ID="ar", MODEL_ID="jonatasgrosman/wav2vec2-large-xlsr-53-arabic"):
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        torch_dtype=torch.float16 if "cuda" == device else torch.float32
    ).to(device)
    model.eval()
    LOGGER.info("Loaded XLSR-53 Model")
    return model, processor

def run_inference_one_by_one(device, records, model, processor):
    all_refs = []
    all_hyps = []
    for i, record in enumerate(records):
        waveform = record["waveform"]
        inputs = processor(
            waveform,
            sampling_rate=record['sample_rate'],
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_sentence = processor.batch_decode(predicted_ids)[0]
        predicted_sentence_clean = clean_arabic_text(predicted_sentence)
        all_refs.append(record["transcription"])
        all_hyps.append(predicted_sentence_clean)
        sample_metrics = metrics.S2TMetrics.evaluate(refs=record["transcription"], hyps=predicted_sentence_clean)

        LOGGER.info("-" * 100)
        LOGGER.info(f"Sample {i+1}, Name: {record['audio_path']}")
        LOGGER.info(f"Reference: {record['transcription']}")
        LOGGER.info(f"Prediction: {predicted_sentence_clean}")
        LOGGER.info(f"Evaluation Metrics: {sample_metrics}")

    # ---------- Overall Metrics ----------
    LOGGER.info("=" * 100)
    overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)
    LOGGER.info("Overall Evaluation Metrics:")
    for k, v in overall_metrics.items():
        LOGGER.info(f"{k}: {v}")

