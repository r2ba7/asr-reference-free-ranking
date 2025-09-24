import gc

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from tqdm import tqdm

from .. import LOGGER
from stt_benchmarking.utils import (
    helpers, 
    decorators, 
    metrics,
    text_processing
)

class XLSRInference:
    """
    A class for loading and running inference with Wav2Vec2 models.
    """
    
    def __init__(self, device, model_version="xlsr-53", lang_id="ar"):
        """
        Initialize the Wav2Vec2Inference class.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Wav2Vec2 model to use ('xlsr-53')
            lang_id (str): Language ID for the model ('ar' for Arabic)
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_version = model_version.lower()
        self.lang_id = lang_id
        if self.model_version not in ['xlsr-53']:
            raise ValueError("model_version must be 'xlsr-53' (only xlsr-53 is currently supported)")
        
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}

    def _load_model(self):
        """
        Load the Wav2Vec2 model and processor based on the specified version.
        
        Returns:
            tuple: (model, processor)
        """
        # Set MODEL_ID based on version and language
        if self.model_version == "xlsr-53" and self.lang_id == "ar":
            MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
        else:
            raise ValueError(f"Unsupported combination: {self.model_version} with {self.lang_id}")
        
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_ID, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        model.eval()
        LOGGER.info(f"Loaded Wav2Vec2 {self.model_version.upper()} Model")
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
                    waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt",
                )
                input_values = inputs['input_values'].to(self.device, dtype=self.dtype)      
                with torch.no_grad():
                    logits = self.model(input_values).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                raw_prediction = self.processor.decode(predicted_ids[0])

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                raw_prediction = ""
            
            all_refs_normalized.append(normalized_transcription)
            all_hyps.append(raw_prediction)
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": None,
            }

        all_hyps_normalized = text_processing.StandardArabicTextProcessor.normalize_texts(all_hyps, substitute=True)
        self._finalize_info(all_audio_paths=all_audio_paths, all_refs_normalized=all_refs_normalized, all_hyps_normalized=all_hyps_normalized)
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)
    
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
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(
                        refs=all_refs_normalized[i],
                        hyps=all_hyps_normalized[i],
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path]["metrics"] = {
                        "word_error_rate": {
                            "wer (%)": None,
                            "substitutions": None,
                            "deletions": None,
                            "insertions": None,
                            "hits": None,
                        },
                        "character_error_rate": {
                            "cer (%)": None,
                            "substitutions": None,
                            "deletions": None,
                            "insertions": None,
                            "hits": None,
                        },
                    }
                    continue
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": {
                        "word_error_rate": {
                            "wer (%)": None,
                            "substitutions": None,
                            "deletions": None,
                            "insertions": None,
                            "hits": None,
                        },
                        "character_error_rate": {
                            "cer (%)": None,
                            "substitutions": None,
                            "deletions": None,
                            "insertions": None,
                            "hits": None,
                        },
                    },
                }

    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not hasattr(self, '_overall_metrics') or not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        
        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}
        
        # Optional: Clear GPU cache if using CUDA
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        
        LOGGER.info("SeamlessM4t instance has been reset. Model and processor remain loaded.")

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info

# Example usage:
# xlsr_object = XLSRInference(device="cuda", model_version="xlsr-53", lang_id="ar")
# 
# # Run inference
# metrics = xlsr_object.run_inference_one_by_one(records)
# xlsr_object.summary_of_evaluation()  # Display simple summary
#
# # Use clean_arabic_text as a static method
# cleaned_text = XLSRInference.clean_arabic_text("some arabic text with diacritics")