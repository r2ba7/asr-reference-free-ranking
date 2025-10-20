import gc
import time

import torch
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm

from .. import LOGGER
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics,
    helpers
)

class w2vBERTInference:
    def __init__(self, device):
        """
        Initialize the HubertArabicInference class.

        Args:
            device (str or torch.device): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self):
        """
        Load the HuBERT Arabic model and processor.

        Returns:
            tuple: (model, processor)
        """
        MODEL_ID = "whitefox123/w2v-bert-2.0-arabic-4"
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCTC.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)

        model.eval()
        LOGGER.info(f"Loaded model w2v Bert Arabic")
        return model, processor

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
            audio_path = record['audio_path']
            transcription = record['transcription']
            normalized_transcription = record['normalized_transcription']
            duration = record["audio_duration"]
            waveform = record["waveform"]
            all_audio_paths.append(audio_path)
            try:
                # Time inference
                start_time = time.time()
                inputs = self.processor(
                    audio=waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt",
                ).to(self.device, dtype=self.dtype)
                input_features = inputs["input_features"].to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    logits = self.model(input_features).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                raw_prediction = self.processor.decode(predicted_ids[0])
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
                    prediction = self._samples_info[audio_path]["raw_prediction"]
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(prediction, substitute=True)
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(
                        refs=self._samples_info[audio_path]["normalized_transcription"],
                        hyps=self._samples_info[audio_path]["normalized_prediction"],
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "normalized_prediction": None,
                        "metrics":  helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
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
        """
        Reset the inference results and metrics without reinitializing the model.
        This clears all stored results from previous inference runs while keeping
        the loaded model intact.
        """
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
    
        LOGGER.info("W2v Bert instance has been reset. Model and processor remain loaded.")

    @property
    def samples_info(self):
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics
