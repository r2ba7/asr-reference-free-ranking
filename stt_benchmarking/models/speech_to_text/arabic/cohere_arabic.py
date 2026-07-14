import gc
import time

import torch
from tqdm import tqdm
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

from . import LOGGER
from stt_benchmarking.utils import (
    helpers,
    text_processing,
    decorators,
    metrics
)


class CohereArabicInference:
    """
    Loading and inference for CohereLabs/cohere-transcribe-arabic-07-2026.
    Dedicated Conformer-encoder / Transformer-decoder AED, Arabic-only.
    Mirrors the Seamless interface: processor -> generate -> decode.
    """

    MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"

    def __init__(self, device, max_new_tokens=256):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self):
        processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            device_map="auto",
        )
        model.eval()
        LOGGER.info("Loaded Cohere Transcribe Arabic (07-2026)")
        return model, processor

    @decorators.Decorators.calculate_execution_time
    def run_inference_one_by_one(self, records, substitute=False, normalize_final_letters=True):
        LOGGER.info(f"Using substitute: {substitute}, Replace Final Char: {normalize_final_letters}")
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
                inputs = self.processor(
                    waveform,
                    sampling_rate=record["sample_rate"],
                    return_tensors="pt",
                    language="ar",
                )
                inputs = inputs.to(self.model.device, dtype=self.model.dtype)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
                raw_prediction = self.processor.decode(outputs[0], skip_special_tokens=True)
                inference_time = time.time() - start_time
                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration
                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"Sample {i+1}, Name: {audio_path}, failed: {e}")
                inference_time = None
                raw_prediction = ""

            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": None,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if (inference_time and duration > 0) else None,
            }

        self._finalize_info(all_audio_paths=all_audio_paths, substitute=substitute, normalize_final_letters=normalize_final_letters)

    def _finalize_info(self, all_audio_paths, substitute, normalize_final_letters):
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    prediction = self._samples_info[audio_path]["raw_prediction"]
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(prediction, substitute=substitute, normalize_final_letters=normalize_final_letters)
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(
                        refs=self._samples_info[audio_path]["normalized_transcription"],
                        hyps=self._samples_info[audio_path]["normalized_prediction"],
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "normalized_prediction": None,
                        "metrics": helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def get_performance_summary(self):
        if self._total_audio_duration == 0:
            return None
        return {
            "average_rtf": self._total_inference_time / self._total_audio_duration if self._total_audio_duration > 0 else None,
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
        LOGGER.info("CohereArabicInference instance has been reset. Model remains loaded.")

    @property
    def samples_info(self):
        return self._samples_info

    @property
    def overall_metrics(self):
        return self._overall_metrics