import gc
import time

import torch
from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline

from . import LOGGER, NORMALIZER_OBJ
from stt_benchmarking.utils import (
    helpers, 
    decorators, 
    metrics,
    text_processing
)

class FasterWhisperInference:
    """
    A class for loading and running inference with FasterWhisper models.
    Supports both Whisper V2 and V3.
    """

    def __init__(self, device="cuda", model_version="v3", batch_size=32):
        self.device = device
        self.model_version = model_version.lower()
        self.batch_size = batch_size
        if self.model_version not in ["v2", "v3"]:
            raise ValueError("model_version must be either 'v2' or 'v3'")

        self.model = self._load_model()
        self.pipeline = self._load_pipeline()
        self._samples_info = {}
        self._overall_metrics = None
        
        # Add timing/memory tracking
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self):
        """
        Load the FasterWhisper model.
        """
        if self.model_version == "v2":
            MODEL_ID = "large-v2"
        else:
            MODEL_ID = "large-v3"

        # FasterWhisper automatically handles device and quantization
        model = WhisperModel(MODEL_ID, device="cuda", compute_type="float16")
        LOGGER.info(f"Loaded FasterWhisper {self.model_version.upper()} Model")
        return model
    
    def _load_pipeline(self):
        """
        Load the Whisper pipeline with automatic chunking support.
        
        Returns:
            pipeline: Hugging Face pipeline for automatic speech recognition
        """
        pipe = BatchedInferencePipeline(model=self.model)
        LOGGER.info(f"Loaded Whisper {self.model_version.upper()} Pipeline")
        return pipe

    @decorators.Decorators.calculate_execution_time 
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one and compute metrics.

        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_audio_paths = []
        if not isinstance(records, list): records = [records]
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            duration = record["audio_duration"]
            all_audio_paths.append(audio_path)
            try:
                # Time inference
                start_time = time.time()
                segments, _ = self.model.transcribe(
                    audio_path,
                    language="en",
                    task="transcribe"
                )
                raw_prediction = " ".join([seg.text for seg in segments])
                normalized_prediction = NORMALIZER_OBJ(raw_prediction)
                end_time = time.time()
                inference_time = end_time - start_time
                
                # Track timing (skip first 5 for warmup)
                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration
                
                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
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
        """
        Run inference on audio records one by one and compute metrics.

        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_audio_paths = []
        if not isinstance(records, list): records = [records]
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            duration = record["audio_duration"]
            all_audio_paths.append(audio_path)
            try:
                start_time = time.time()
                segments, _ = self.pipeline.transcribe(
                    audio_path,
                    batch_size=self.batch_size,
                    language="en",
                    task="transcribe"
                )

                # Concatenate all segment texts
                raw_prediction = " ".join([seg.text for seg in segments])
                normalized_prediction = NORMALIZER_OBJ(raw_prediction)
                inference_time = time.time() - start_time
                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration
                
                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
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

    def _finalize_info(self, all_audio_paths):
        """
        Finalize predictions by normalizing them and computing metrics for each sample.

        Args:
            all_audio_paths (list): List of audio file paths.
            all_refs_normalized (list): List of normalized reference texts.
            all_hyps_normalized (list): List of normalized hypothesis texts.
        """
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
                    self._samples_info[audio_path] = {
                        "metrics":  helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "metrics":  helpers._empty_metrics()
                }

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def get_performance_summary(self):
        """Return performance metrics dict"""
        if self._total_audio_duration == 0:
            return None
        
        return {
            "average_rtf": self._total_inference_time / self._total_audio_duration if self._total_audio_duration > 0 else None,
            "total_inference_time": self._total_inference_time,
            "total_audio_duration": self._total_audio_duration,
            "processed_samples": self._processed_count
        }

    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
        
        # Add performance metrics
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
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

        LOGGER.info("FasterWhisperInference instance has been reset. Model remains loaded.")

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info

# Example usage:
# whisper_v2 = WhisperInference(device="cuda", model_version="v2")
# whisper_v3 = WhisperInference(device="cuda", model_version="v3")
# 
# # Run inference
# metrics_v2 = whisper_v2.run_inference_one_by_one(records)
# whisper_v2.summary_of_evaluation()  # Display formatted summary
# 
# metrics_v3 = whisper_v3.run_inference_one_by_one(records)
# whisper_v3.summary_of_evaluation()  # Display formatted summary
# Example usage:
# whisper_v2 = WhisperInference(device="cuda", model_version="v2")
# whisper_v3 = WhisperInference(device="cuda", model_version="v3")
# 
# # Run inference
# metrics_v2 = whisper_v2.run_inference_one_by_one(records)
# whisper_v2.summary_of_evaluation()  # Display formatted summary
# 
# metrics_v3 = whisper_v3.run_inference_one_by_one(records)
# whisper_v3.summary_of_evaluation()  # Display formatted summary