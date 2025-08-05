import gc

import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
from tqdm import tqdm

from . import LOGGER
from stt_benchmarking.utils import (
    validate,
    text_processing, 
    decorators, 
    metrics
)


class SeamlessM4TInference:
    """
    A class for loading and running inference with Seamless M4T models.
    """
    
    def __init__(self, device, model_version="v2"):
        """
        Initialize the SeamlessM4TInference class.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Seamless M4T model to use ('v2')
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_version = model_version.lower()
        
        if self.model_version not in ['v2']:
            raise ValueError("model_version must be 'v2' (only v2 is currently supported)")
        
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}

    def _load_model(self):
        """
        Load the Seamless M4T model and processor based on the specified version.
        
        Returns:
            tuple: (model, processor)
        """
        # Set MODEL_ID based on version
        if self.model_version == "v2":
            MODEL_ID = "facebook/seamless-m4t-v2-large"
        
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = SeamlessM4Tv2Model.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        model.eval()
        LOGGER.info(f"Loaded Seamless M4T {self.model_version.upper()} Model")
        return model, processor

    @decorators.Decorators.calculate_execution_time 
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one and compute metrics.
        
        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_refs_processed = []
        all_hyps = []
        all_audio_paths = []
        if not isinstance(records, list):
            records = [records]
             
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            audio_path = record['audio_path']
            transcription = record['transcription']
            processed_transcription = record['processed_transcription']
            all_audio_paths.append(audio_path)
            try:
                inputs = self.processor(
                    audios=waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt"
                ).to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    output = self.model.generate(**inputs, generate_speech=False, tgt_lang="arb")

                raw_prediction = self.processor.decode(output[0][0].tolist(), skip_special_tokens=True)
                validated_raw_prediction = validate.ValidateText.validate_text_in_ar(raw_prediction)

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                continue

            all_refs_processed.append(processed_transcription)
            all_hyps.append(validated_raw_prediction)
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "processed_transcription": processed_transcription,
                "raw_prediction": validated_raw_prediction,
                "normalized_prediction": None,
                "processed_prediction": None    
            }

        all_hyps_normalized = text_processing.ArabicTextProcessor.normalize_texts(all_hyps)
        all_hyps_processed =  text_processing.ArabicTextProcessor.process_texts(all_hyps)     
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    self._samples_info[audio_path]["normalized_prediction"] = all_hyps_normalized[i]
                    self._samples_info[audio_path]["processed_prediction"] = all_hyps_processed[i]
                    sample_metrics = metrics.FilteredS2TMetrics.evaluate(
                        refs=all_refs_processed[i],
                        hyps=all_hyps_processed[i],
                        single_sample=True
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    LOGGER.info(f"{self._samples_info[audio_path]}")
                    self._samples_info[audio_path]['metrics'] = {"word_accuracy": None, "char_accuracy": None, "average_score": None}
                    continue
                
        self._overall_metrics = metrics.FilteredS2TMetrics.evaluate(refs=all_refs_processed, hyps=all_hyps_processed)

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
# seamless_m4t = SeamlessM4TInference(device="cuda", model_version="v2")
# 
# # Run inference
# metrics = seamless_m4t.run_inference_one_by_one(records)
# seamless_m4t.summary_of_evaluation()  # Display simple summary