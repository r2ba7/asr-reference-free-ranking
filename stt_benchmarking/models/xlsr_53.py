import re
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from . import LOGGER
from stt_benchmarking.utils import metrics


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
        self.device = device
        self.model_version = model_version.lower()
        self.lang_id = lang_id
        if self.model_version not in ['xlsr-53']:
            raise ValueError("model_version must be 'xlsr-53' (only xlsr-53 is currently supported)")
        
        self.model, self.processor = self._load_model()
        self.model_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
    
    @staticmethod
    def clean_arabic_text(text: str) -> str:
        """
        Cleans Arabic text by:
        - Removing Arabic diacritics
        - Stripping redundant whitespace
        - Returning a clean string
        
        Args:
            text (str): Input Arabic text to clean
            
        Returns:
            str: Cleaned Arabic text
        """
        text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
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
        torch_dtype = torch.float32
        model = Wav2Vec2ForCTC.from_pretrained(
            MODEL_ID, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            torch_dtype=torch_dtype
        ).to(self.device)
        
        model.eval()
        LOGGER.info(f"Loaded Wav2Vec2 {self.model_version.upper()} Model")
        return model, processor
    
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one and compute metrics.
        
        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_refs = []
        all_hyps = []
        for i, record in enumerate(records):
            waveform = record["waveform"]
            try:
                inputs = self.processor(
                    waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    logits = self.model(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                predicted_sentence = self.processor.batch_decode(predicted_ids)[0]
                predicted_sentence_clean = self.clean_arabic_text(predicted_sentence)
                
            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                continue
            
            all_refs.append(record["transcription"])
            all_hyps.append(predicted_sentence_clean)
            sample_metrics = metrics.S2TMetrics.evaluate(refs=record["transcription"], hyps=predicted_sentence_clean)

            LOGGER.info("-" * 100)
            LOGGER.info(f"Sample {i+1}, Name: {record['audio_path']}")
            LOGGER.info(f"Reference: {record['transcription']}")
            LOGGER.info(f"Prediction: {predicted_sentence_clean}")
            LOGGER.info(f"Evaluation Metrics: {sample_metrics}")

        self.overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)

    
    def summary_of_evaluation(self):
        """
        Display a simple summary of the overall evaluation metrics.
        """
        if not hasattr(self, 'overall_metrics') or not self.overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        
        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self.overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
        
    @property
    def metrics(self):
        return self.overall_metrics

# Example usage:
# xlsr_object = XLSRInference(device="cuda", model_version="xlsr-53", lang_id="ar")
# 
# # Run inference
# metrics = xlsr_object.run_inference_one_by_one(records)
# xlsr_object.summary_of_evaluation()  # Display simple summary
#
# # Use clean_arabic_text as a static method
# cleaned_text = XLSRInference.clean_arabic_text("some arabic text with diacritics")