import gc

import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model, pipeline
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
        all_refs_normalized = []
        all_hyps = []
        all_audio_paths = []
        if not isinstance(records, list):
            records = [records]
             
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
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

        all_hyps_normalized = text_processing.BasicArabicTextProcessing.normalize_texts(all_hyps)
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
                        single_sample=True
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path]["metrics"] = {
                        "word_accuracy": None,
                        "char_accuracy": None,
                    }
                    continue

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
    
class SeamlessM4TPipelineInference:
    """
    A class for loading and running inference with Seamless M4T models using Pipeline.
    """
    
    def __init__(self, device, model_version="v2", chunk_length_s=30):
        """
        Initialize the SeamlessM4TInference class.
        
        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Seamless M4T model to use ('v2')
            chunk_length_s (int): Chunk length in seconds for processing long audio
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_version = model_version.lower()
        self.chunk_length_s = chunk_length_s
        
        if self.model_version not in ['v2']:
            raise ValueError("model_version must be 'v2' (only v2 is currently supported)")
        
        self.pipe = self._load_pipeline()
        self._overall_metrics = None
        self._samples_info = {}

    def _load_pipeline(self):
        """
        Load the Seamless M4T pipeline based on the specified version.
        
        Returns:
            pipeline: Hugging Face pipeline for automatic speech recognition
        """
        # Set MODEL_ID based on version
        if self.model_version == "v2":
            MODEL_ID = "facebook/seamless-m4t-v2-large"
        
        # Create pipeline with automatic chunking
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=0 if self.device.type == "cuda" else -1,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            chunk_length_s=self.chunk_length_s,  # Enable chunking
            stride_length_s=5,
            ignore_warning=True  # Overlap between chunks for better continuity
        )
        
        LOGGER.info(f"Loaded Seamless M4T {self.model_version.upper()} Pipeline with {self.chunk_length_s}s chunking")
        return pipe

    @decorators.Decorators.calculate_execution_time 
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one using Pipeline and compute metrics.
        
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
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            sample_rate = record['sample_rate']
            all_audio_paths.append(audio_path)
            
            try:
                # Prepare audio input - Pipeline expects numpy array or raw audio
                if isinstance(waveform, torch.Tensor):
                    audio_input = waveform.numpy()
                else:
                    audio_input = waveform
                
                # Run inference with automatic chunking
                result = self.pipe(
                    audio_input,
                    generate_kwargs={"tgt_lang": "arb"}  # Target language: Arabic
                )
                
                # Extract text from result
                if isinstance(result, dict):
                    raw_prediction = result.get("text", "")
                else:
                    raw_prediction = str(result)

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

        # Normalize predictions using your existing text processor
        all_hyps_normalized = text_processing.BasicArabicTextProcessing.normalize_texts(all_hyps)
        self._finalize_info(all_audio_paths=all_audio_paths, all_refs_normalized=all_refs_normalized, all_hyps_normalized=all_hyps_normalized)
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)
    
    def run_batch_inference(self, records, batch_size=8):
        """
        Run batch inference using Pipeline for better efficiency.
        
        Args:
            records (list): List of audio records
            batch_size (int): Number of samples to process in each batch
        """
        all_refs_normalized = []
        all_hyps = []
        all_audio_paths = []
        
        if not isinstance(records, list):
            records = [records]
        
        # Process in batches
        for i in tqdm(range(0, len(records), batch_size), desc="Processing Batches"):
            batch = records[i:i + batch_size]
            batch_audio = []
            batch_sample_rates = []
            batch_paths = []
            batch_refs = []
            
            for record in batch:
                waveform = record["waveform"]
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.numpy()
                
                batch_audio.append(waveform)
                batch_sample_rates.append(record['sample_rate'])
                batch_paths.append(record["audio_path"])
                batch_refs.append(record["normalized_transcription"])
            
            try:
                # Run batch inference
                results = self.pipe(
                    batch_audio,
                    batch_size=batch_size,
                    generate_kwargs={"tgt_lang": "arb"}
                )
                
                for j, (result, audio_path, transcription, normalized_transcription) in enumerate(zip(results, batch_paths, [records[i+j]["transcription"] for j in range(len(batch))], batch_refs)):
                    if isinstance(result, dict):
                        raw_prediction = result.get("text", "")
                    else:
                        raw_prediction = str(result)
                    
                    all_refs_normalized.append(normalized_transcription)
                    all_hyps.append(raw_prediction)
                    all_audio_paths.append(audio_path)
                    
                    self._samples_info[audio_path] = {
                        "raw_transcription": transcription,
                        "normalized_transcription": normalized_transcription,
                        "raw_prediction": raw_prediction,
                        "normalized_prediction": None,
                    }
                    
            except Exception as e:
                LOGGER.error(f"⚠️ Batch starting at {i} failed: {e}")
                continue
        
        # Normalize predictions
        all_hyps_normalized = text_processing.BasicArabicTextProcessing.normalize_texts(all_hyps)
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
                        "word_accuracy": None,
                        "char_accuracy": None,
                    }
                    continue
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": {
                        "word_accuracy": None,
                        "char_accuracy": None,
                    }}

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