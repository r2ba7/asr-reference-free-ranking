import torch
from transformers import AutoModelForCTC, AutoProcessor
from stt_benchmarking.utils import metrics

def load_model(device, MODEL_ID="MostafaAhmed98/Conformer-CTC-Arabic-ASR"):
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID)
    model = model.to(device)
    return model, processor


def run_ctc_inference_one_by_one(device, records, model, processor):
    model.eval()
    all_refs = []
    all_hyps = []

    for i, record in enumerate(records):
        waveform = record["waveform"]
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()

        try:
            inputs = processor(
                waveform,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                logits = model(**inputs).logits

            pred_ids = torch.argmax(logits, dim=-1)
            prediction = processor.batch_decode(pred_ids)[0].strip()

        except Exception as e:
            print(e)
            print(f"Sample {i+1}, Name: {record['audio_path']}")
            continue

        all_refs.append(record["transcription"])
        all_hyps.append(prediction)
        sample_metrics = metrics.S2TMetrics.evaluate(refs=record["transcription"], hyps=prediction)

        print("-" * 100)
        print(f"Sample {i+1}, Name: {record['audio_path']}")
        print("Reference :", record["transcription"])
        print("Prediction:", prediction)
        print("Evaluation Metrics: ", sample_metrics)

    # ---------- Overall Metrics ----------
    print("=" * 100)
    overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)
    print("Overall Evaluation Metrics:")
    for k, v in overall_metrics.items():
        print(f"{k}: {v}")
