import torch
from transformers import AutoModel, AutoProcessor

from . import LOGGER
from stt_benchmarking.utils import metrics

class HubertInference:
    def __init__(self, device):
        """
        Initialize the WhisperInference class.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
    
    def _load_model(self):
        """
        Load the Whisper model and processor based on the specified version.
        
        Returns:
            tuple: (model, processor)
        """
        MODEL_ID = "asafaya/hubert-large-arabic-transcribe"
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        model.eval()
        LOGGER.info(f"Loaded Hubert Model")
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
                    audios=waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt"
                ).to(self.device, dtype=self.dtype)

                # Ask for text output only
                with torch.no_grad():
                    logits = self.model(**inputs).logits

                predicted_ids = torch.argmax(logits, dim=-1)
                prediction = self.processor.batch_decode(predicted_ids)[0]

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                continue

            all_refs.append(record["transcription"])
            all_hyps.append(prediction)
            sample_metrics = metrics.S2TMetrics.evaluate(refs=record["transcription"], hyps=prediction)

            LOGGER.info("-" * 100)
            LOGGER.info(f"Sample {i+1}, Name: {record['audio_path']}")
            LOGGER.info(f"Reference: {record['transcription']}")
            LOGGER.info(f"Prediction: {prediction}")
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

