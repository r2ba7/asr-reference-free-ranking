import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
import gc

from . import LOGGER
from stt_benchmarking.utils import (
    validate, 
    decorators, 
    metrics,
    text_processing
)


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
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_version = model_version.lower()
        if self.model_version not in ['v2', 'v3']:
            raise ValueError("model_version must be either 'v2' or 'v3'")
        
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
        self._samples_info = {}
        self._overall_metrics = None

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
                    waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt"
                    )
                input_features = inputs["input_features"].to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    generated_ids = self.model.generate(input_features=input_features)

                raw_prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
        if not hasattr(self, 'overall_metrics') or not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        
        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
    
    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}
        
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        
        LOGGER.info("WhisperInference instance has been reset. Model and processor remain loaded.")

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