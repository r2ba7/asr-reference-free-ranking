import gc
import time

from datasets import Dataset
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, pipeline
from tqdm import tqdm
import numpy as np
import pandas as pd

from . import LOGGER, NORMALIZER_OBJ
from stt_benchmarking.utils import (
    helpers,
    decorators,
    metrics
)

class Wav2Vec2FullInterface:
    """
    Loading and running inference with Wav2Vec2 models using Pipeline.
    """

    def __init__(self, device, model_id="facebook/wav2vec2-large-960h-lv60-self", chunk_length_s=20, stride_length_s=1, batch_size=16):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_id = model_id
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.batch_size = batch_size

        self.model, self.processor = self._load_model()
        self.pipe = self._load_pipeline()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}

        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self):
        processor = Wav2Vec2Processor.from_pretrained(self.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(
            self.model_id,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        model.eval()
        LOGGER.info(f"Loaded Wav2Vec2 model: {self.model_id}")
        return model, processor

    def _load_pipeline(self):
        pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=0 if self.device.type == "cuda" else -1,
            torch_dtype=torch.float16,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
            batch_size=self.batch_size,
        )
        LOGGER.info(
            f"Loaded Wav2Vec2 Pipeline "
            f"({self.chunk_length_s}s chunks / {self.stride_length_s}s stride)"
        )
        return pipe

    @decorators.Decorators.calculate_execution_time
    def run_inference_one_by_one(self, records):
        all_audio_paths = []
        if not isinstance(records, list):
            records = [records]
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            duration = record["audio_duration"]
            all_audio_paths.append(audio_path)
            try:
                start_time = time.time()
                # Wav2Vec2 expects float32 input values, not generic audio dicts
                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.squeeze().numpy().astype(np.float32)
                else:
                    waveform_np = np.array(waveform, dtype=np.float32)

                input_values = self.processor(
                    waveform_np,
                    sampling_rate=record["sample_rate"],
                    return_tensors="pt",
                    padding="longest"
                ).input_values.to(self.device, dtype=torch.float32)

                with torch.no_grad():
                    logits = self.model(input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                raw_prediction = self.processor.batch_decode(predicted_ids)[0]
                normalized_prediction = NORMALIZER_OBJ(raw_prediction)
                inference_time = time.time() - start_time

                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration

                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                inference_time = None
                raw_prediction = ""
                normalized_prediction = ""

            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized_prediction,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if (inference_time and duration > 0) else None,
            }

        self._finalize_info(all_audio_paths=all_audio_paths)

    @decorators.Decorators.calculate_execution_time
    def run_batch_inference(self, records):
        def process_batch(batch):
            batch_audio = batch["waveform"]
            batch_audio = [
                np.array(audio, dtype=np.float32) if not isinstance(audio, np.ndarray) else audio.astype(np.float32)
                for audio in batch_audio
            ]
            with torch.no_grad():
                try:
                    results = self.pipe(
                        batch_audio,
                        batch_size=len(batch_audio),
                    )
                    batch_hyps = []
                    norm_batch_hyps = []
                    for result in results:
                        if isinstance(result, dict):
                            raw_prediction = result.get("text", "")
                        elif isinstance(result, list) and len(result) > 0:
                            raw_prediction = " ".join([
                                chunk.get("text", "") for chunk in result if isinstance(chunk, dict)
                            ])
                        else:
                            raw_prediction = str(result)
                        normalized_prediction = NORMALIZER_OBJ(raw_prediction)
                        batch_hyps.append(raw_prediction)
                        norm_batch_hyps.append(normalized_prediction)
                    return {"predictions": batch_hyps, "norm_predictions": norm_batch_hyps}
                except Exception as e:
                    LOGGER.error(f"Batch processing failed: {e}")
                    return {
                        "predictions": ["" for _ in batch_audio],
                        "norm_predictions": ["" for _ in batch_audio]
                    }

        processed_records = []
        all_audio_paths = []
        for record in records:
            waveform = record["waveform"]
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().numpy()
            processed_records.append({
                "waveform": waveform,
                "audio_path": record["audio_path"],
                "transcription": record["transcription"],
                "normalized_transcription": record["normalized_transcription"],
                "sample_rate": record["sample_rate"],
                "duration": record["audio_duration"]
            })

        dataset = Dataset.from_dict({
            "waveform": [r["waveform"] for r in processed_records],
            "audio_path": [r["audio_path"] for r in processed_records],
            "transcription": [r["transcription"] for r in processed_records],
            "normalized_transcription": [r["normalized_transcription"] for r in processed_records],
            "sample_rate": [r["sample_rate"] for r in processed_records],
            "duration": [r["duration"] for r in processed_records]
        })

        batch_start_time = time.time()
        total_duration = sum(r["duration"] for r in processed_records)

        dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=self.batch_size,
            desc="Processing Batches",
            drop_last_batch=False
        )

        batch_inference_time = time.time() - batch_start_time
        self._total_inference_time += batch_inference_time
        self._total_audio_duration += total_duration
        self._processed_count += len(records)

        for audio_path, transcription, normalized_transcription, prediction, normalized_prediction, duration in zip(
            dataset["audio_path"], dataset["transcription"], dataset["normalized_transcription"],
            dataset["predictions"], dataset["norm_predictions"], dataset["duration"]
        ):
            all_audio_paths.append(audio_path)
            sample_rtf = (batch_inference_time / len(records)) / duration if duration > 0 else None
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": prediction,
                "normalized_prediction": normalized_prediction,
                "duration": duration,
                "inference_time": batch_inference_time / len(records),
                "rtf": sample_rtf,
            }

        self._finalize_info(all_audio_paths=all_audio_paths)

    @decorators.Decorators.calculate_execution_time
    def run_inference_optimized(self, records, duration_threshold=30.0):
        if not isinstance(records, list):
            records = [records]

        short_records = [r for r in records if r["audio_duration"] < duration_threshold]
        long_records = [r for r in records if r["audio_duration"] >= duration_threshold]

        if short_records:
            LOGGER.info(f"Running one-by-one inference on {len(short_records)} short files (<{duration_threshold}s)")
            self.run_inference_one_by_one(short_records)

        if long_records:
            LOGGER.info(f"Running batch inference on {len(long_records)} long files (>={duration_threshold}s)")
            self.run_batch_inference(long_records)

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def _finalize_info(self, all_audio_paths):
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(
                        refs=self._samples_info[audio_path]["normalized_transcription"],
                        hyps=self._samples_info[audio_path]["normalized_prediction"],
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {"metrics": helpers._empty_metrics()}
            else:
                self._samples_info[audio_path] = {"metrics": helpers._empty_metrics()}

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def get_performance_summary(self):
        if self._total_audio_duration == 0:
            return None
        return {
            "average_rtf": self._total_inference_time / self._total_audio_duration,
            "total_inference_time": self._total_inference_time,
            "total_audio_duration": self._total_audio_duration,
            "processed_samples": self._processed_count
        }

    def summary_of_evaluation(self):
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
        perf = self.get_performance_summary()
        if perf:
            LOGGER.info(f"Average RTF: {perf['average_rtf']:.4f}")
            LOGGER.info(f"Total Inference Time: {perf['total_inference_time']:.2f}s")
            LOGGER.info(f"Total Audio Duration: {perf['total_audio_duration']:.2f}s")
            LOGGER.info(f"Processed Samples: {perf['processed_samples']}")

    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        LOGGER.info("Wav2Vec2 instance has been reset. Model and processor remain loaded.")

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info