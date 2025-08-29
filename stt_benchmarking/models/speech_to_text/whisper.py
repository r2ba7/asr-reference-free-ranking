import gc

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
from faster_whisper import WhisperModel

from .. import LOGGER
from stt_benchmarking.utils import (
    validate, 
    decorators, 
    metrics,
    text_processing
)


class FasterWhisperInference:
    """
    A class for loading and running inference with FasterWhisper models.
    Supports both Whisper V2 and V3.
    """

    def __init__(self, device="cuda", model_version="v3"):
        """
        Initialize the FasterWhisperInference class.

        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Whisper model to use ('v2' or 'v3')
        """
        self.device = device
        self.model_version = model_version.lower()
        if self.model_version not in ["v2", "v3"]:
            raise ValueError("model_version must be either 'v2' or 'v3'")

        self.model = self._load_model()
        self._samples_info = {}
        self._overall_metrics = None

    def _load_model(self):
        """
        Load the FasterWhisper model.
        """
        if self.model_version == "v2":
            MODEL_ID = "openai/whisper-large-v2"
        else:
            MODEL_ID = "large-v3"

        # FasterWhisper automatically handles device and quantization
        device_str = "cuda" if str(self.device) == "cuda" else "cpu"
        model = WhisperModel(
            MODEL_ID,
            device=device_str,
            compute_type="float16" if self.device == "cuda" else "int8",  # can adjust for speed/accuracy
        )

        LOGGER.info(f"Loaded FasterWhisper {self.model_version.upper()} Model")
        return model

    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one and compute metrics.

        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_refs_normalized = []
        all_hyps = []
        all_audio_paths = []

        if not isinstance(records, list):
            records = [records]

        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            all_audio_paths.append(audio_path)

            try:
                # FasterWhisper handles long audios internally
                segments, info = self.model.transcribe(
                    audio_path,
                    language="ar",   # Arabic
                    task="transcribe"
                )

                # Concatenate all segment texts
                raw_prediction = " ".join([seg.text for seg in segments])

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
                continue

            all_refs_normalized.append(normalized_transcription)
            all_hyps.append(raw_prediction)
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": None,
            }

        all_hyps_normalized = text_processing.StandardArabicTextProcessor.normalize_texts(all_hyps)
        self._finalize_info(all_audio_paths=all_audio_paths, all_refs_normalized=all_refs_normalized, all_hyps_normalized=all_hyps_normalized)
        self._overall_metrics = metrics.StandardSTTMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)

    def _finalize_info(self, all_audio_paths, all_refs_normalized, all_hyps_normalized):
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
                    self._samples_info[audio_path]["normalized_prediction"] = all_hyps_normalized[i]
                    sample_metrics = metrics.StandardSTTMetrics.evaluate(
                        refs=all_refs_normalized[i],
                        hyps=all_hyps_normalized[i],
                        single_sample=True
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path]["metrics"] = {
                        "word_accuracy": None,
                        "char_accuracy": None,
                        "average_score": None,
                    }
                    continue

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

    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}

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