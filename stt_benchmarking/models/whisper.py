import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from . import LOGGER
from stt_benchmarking.utils import metrics


class WhisperInference:
    """
    A class for loading and running inference with Whisper models.
    Supports both Whisper V2 and V3 models.
    """
    
    def __init__(self, device, model_version="v3"):
        """
        Initialize the WhisperInference class.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Whisper model to use ('v2' or 'v3')
        """
        self.device = device
        self.model_version = model_version.lower()
        if self.model_version not in ['v2', 'v3']:
            raise ValueError("model_version must be either 'v2' or 'v3'")
        
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
    
    def _load_model(self):
        """
        Load the Whisper model and processor based on the specified version.
        
        Returns:
            tuple: (model, processor)
        """
        # Set MODEL_ID based on version
        if self.model_version == "v2":
            MODEL_ID = "openai/whisper-large-v2"
        else:  # v3
            MODEL_ID = "openai/whisper-large-v3"
        
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            MODEL_ID, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32 # if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="arabic", task="transcribe")
        model.eval()
        LOGGER.info(f"Loaded Whisper {self.model_version.upper()} Model")
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
                    return_tensors="pt"
                    )
                input_features = inputs["input_features"].to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    generated_ids = self.model.generate(
                                        input_features=input_features,
                                    )

                prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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