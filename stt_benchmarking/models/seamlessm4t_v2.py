import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
from tqdm import tqdm

from . import LOGGER
from stt_benchmarking.utils import (
    helpers, 
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
        all_refs = []
        all_hyps = []
        if isinstance(records, str):
            records = [records]  
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            audio_path = record['audio_path']
            transcription = record['transcription']
            try:
                inputs = self.processor(
                    audios=waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt"
                ).to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    output = self.model.generate(**inputs, generate_speech=False, tgt_lang="arb")

                prediction = self.processor.decode(output[0][0].tolist(), skip_special_tokens=True)
                predicted_sentence_clean = helpers.clean_arabic_text(prediction)
                sample_metrics = metrics.S2TMetrics.evaluate(refs=transcription, hyps=predicted_sentence_clean)

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                continue

            all_refs.append(transcription)
            all_hyps.append(predicted_sentence_clean)
            self._samples_info[audio_path] = {
                "transcription": transcription,
                "prediction": predicted_sentence_clean,
                "metrics": sample_metrics
            }

        self._overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)

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
        
    @property
    def _verall_metrics(self):
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