import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from . import LOGGER
from stt_benchmarking.utils import metrics



def load_model(device, MODEL_ID="openai/whisper-large-v2"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, 
        low_cpu_mem_usage=True, 
        use_safetensors=True,
        torch_dtype=torch.float16 if "cuda" == device else torch.float32  # Use FP16 on GPU
    ).to(device)
    
    model.config.forced_decoder_ids = None
    model.eval()
    LOGGER.info("Loaded Whisper V2 Model")
    return model, processor

def run_whisper_inference_one_by_one(device, records, model, processor):
    all_refs = []
    all_hyps = []
    for i, record in enumerate(records):
        waveform = record["waveform"]
        try:
            inputs = processor(
                waveform,
                sampling_rate=record['sample_rate'],
                return_tensors="pt"
            ).inputs.to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language="arabic", task="transcribe"
            )
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs,
                    forced_decoder_ids=forced_decoder_ids
                )

            prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        except Exception as e:
            LOGGER.error(f"⚠️ Sample {i+1},  Name: {record['audio_path']}, failed: {e}")
            continue

        all_refs.append(record["transcription"])
        all_hyps.append(prediction)
        sample_metrics = metrics.S2TMetrics.evaluate(refs=record["transcription"], hyps=prediction)

        LOGGER.info("-" * 100)
        LOGGER.info(f"Sample {i+1}, Name: {record['audio_path']}")
        LOGGER.info(f"Reference: {record['transcription']}")
        LOGGER.info(f"Prediction: {prediction}")
        LOGGER.info(f"Evaluation Metrics: {sample_metrics}")

    # ---------- Overall Metrics ----------
    LOGGER.info("=" * 100)
    overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)
    LOGGER.info("Overall Evaluation Metrics:")
    for k, v in overall_metrics.items():
        LOGGER.info(f"{k}: {v}")
