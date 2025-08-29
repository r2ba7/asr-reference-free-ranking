import torch
from transformers import AutoProcessor, AutoModelForCTC
from tqdm import tqdm

from .. import LOGGER
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics
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
        all_refs_normalized = []
        all_hyps = []
        all_audio_paths = []
        if not isinstance(records, list):
            records = [records]
            
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            audio_path = record['audio_path']
            transcription = record['transcription']
            normalized_transcription = record['normalized_transcription']
            all_audio_paths.append(audio_path)
            try:
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

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
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
        if not hasattr(self, 'overall_metrics') or not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        
        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
        
    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info

