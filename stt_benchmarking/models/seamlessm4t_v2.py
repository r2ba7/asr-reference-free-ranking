import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model

from . import LOGGER
from stt_benchmarking.utils import metrics


def load_model(device, MODEL_ID="facebook/seamless-m4t-v2-large"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SeamlessM4Tv2Model.from_pretrained(MODEL_ID,
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        torch_dtype=torch.float16 if "cuda" == device else torch.float32
    ).to(device)
    model.eval()
    LOGGER.info("Loaded Seamless M4T Model")
    return model, processor

def run_seamlessm4t_inference_one_by_one(device, records, model, processor):
    all_refs, all_hyps = [], []

    for i, record in enumerate(records):
        waveform = record["waveform"]
        try:
            inputs = processor(
                audios=waveform,
                sampling_rate=record['sample_rate'],
                return_tensors="pt"
            ).to(device)

            # Ask for text output only
            with torch.no_grad():
                output = model.generate(**inputs, generate_speech=False, tgt_lang="arb")

            # Token-level output, decode to string
            prediction = processor.decode(output[0][0].tolist(), skip_special_tokens=True)

        except Exception as e:
            LOGGER.error(f"⚠️ Sample {i+1},  Name: {record['audio_path']}, failed: {e}")
            continue

        all_refs.append(record["transcription"])
        all_hyps.append(prediction)
        sample_metrics = metrics.S2TMetrics.evaluate(refs=record["transcription"], hyps=prediction)

        LOGGER.info(f"Sample {i+1}, Name: {record['audio_path']}")
        LOGGER.info(f"Reference: {record['transcription']}")
        LOGGER.info(f"Prediction: {prediction}")
        LOGGER.info(f"Evaluation Metrics: {sample_metrics}")

    # Overall metrics
    LOGGER.info("\n" + "="*80)
    overall = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)
    for k, v in overall.items():
        LOGGER.info(f"{k}: {v}")