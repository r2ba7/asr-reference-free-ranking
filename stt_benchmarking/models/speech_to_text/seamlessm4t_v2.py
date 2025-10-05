import gc
import time

from datasets import Dataset
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model, pipeline, SeamlessM4TProcessor, SeamlessM4Tv2ForSpeechToText
from tqdm import tqdm
import numpy as np

from . import LOGGER
from stt_benchmarking.utils import (
    helpers,
    text_processing, 
    decorators, 
    metrics
)
    
class SeamlessM4TFullInterface:
    """
    A class for loading and running inference with Seamless M4T models using Pipeline.
    """
    
    def __init__(self, device, model_version="v2", chunk_length_s=20, stride_length_s=1, batch_size=16):
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
        self.stride_length_s = stride_length_s
        self.batch_size = batch_size
        if self.model_version not in ['v2']:
            raise ValueError("model_version must be 'v2' (only v2 is currently supported)")
        
        self.model, self.processor = self._load_model()
        self.pipe = self._load_pipeline()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}
                
        # Add timing/memory tracking
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0


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
        model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
            MODEL_ID,
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            # device_map="auto",
            torch_dtype=torch.float16,
        ).to(self.device)
        
        model.eval()
        LOGGER.info(f"Loaded Seamless M4T {self.model_version.upper()} Model")
        return model, processor

    def _load_pipeline(self):
        """
        Load the Seamless M4T pipeline based on the specified version.
        
        Returns:
            pipeline: Hugging Face pipeline for automatic speech recognition
        """
        # Set MODEL_ID based on version
        if self.model_version == "v2":
            MODEL_ID = "facebook/seamless-m4t-v2-large"
        
        # Create pipeline with utomatic chunking
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=0,
            torch_dtype=torch.float16,
            chunk_length_s=self.chunk_length_s,  # Enable chunking
            stride_length_s= self.stride_length_s,
            batch_size=self.batch_size,
            ignore_warning=True,
        )
        
        LOGGER.info(
            f"Loaded Seamless M4T {self.model_version.upper()} Pipeline "
            f"({self.chunk_length_s}s chunks / {self.stride_length_s}s stride)"
        )
        return pipe

    @decorators.Decorators.calculate_execution_time 
    def run_inference_one_by_one(self, records, language_ids):
        """
        Run inference on audio records one by one and compute metrics.
        
        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                        transcription, and audio_path
            language_ids (list): List of language IDs to evaluate for each sample
        """
        all_refs_normalized, all_hyps, all_audio_paths = [], [], []
        if not isinstance(records, list): records = [records]    
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            duration = record["audio_duration"]
            all_audio_paths.append(audio_path)
            try:
                start_time = time.time()
                inputs = self.processor(
                    audios=waveform,
                    sampling_rate=record['sample_rate'],
                    return_tensors="pt"
                ).to(self.device, dtype=self.dtype)
                
                best_prediction = None
                best_wer = float('inf')
                best_lang = None
                
                for lang_id in language_ids:
                    with torch.no_grad():
                        output = self.model.generate(**inputs, generate_speech=False, tgt_lang=lang_id)[0]
                    candidate = self.processor.decode(output[0], skip_special_tokens=True)
                    candidate_normalized = text_processing.StandardArabicTextProcessor.main([candidate], substitute=True)[0]
                    sample_wer = metrics.BasicSTTMetrics.evaluate(
                        refs=normalized_transcription,
                        hyps=candidate_normalized
                    )["word_error_rate"]["wer (%)"]
                    
                    if sample_wer < best_wer:
                        best_wer = sample_wer
                        best_prediction = candidate
                
                raw_prediction = best_prediction
                inference_time = time.time() - start_time
                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration
                
                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {record['audio_path']}, failed: {e}")
                inference_time = None
                raw_prediction = ""

            all_refs_normalized.append(normalized_transcription)
            all_hyps.append(raw_prediction)
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": None,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if (inference_time and duration > 0) else None,
            }

        all_hyps_normalized = text_processing.StandardArabicTextProcessor.main(all_hyps, substitute=True)
        self._finalize_info(all_audio_paths=all_audio_paths, all_refs_normalized=all_refs_normalized, all_hyps_normalized=all_hyps_normalized)
    
    @decorators.Decorators.calculate_execution_time
    def run_batch_inference(self, records,):
        """
        Run batch inference using Hugging Face Dataset for better efficiency.
        Selects best language by evaluating on a sample subset.
        
        Args:
            records (list): List of audio records
            language_ids (list): Language IDs to evaluate
            sample_size (int): Number of samples to use for language selection
        """
        def process_batch(batch):
            batch_audio = batch["waveform"]
            batch_audio = [np.array(audio, dtype=np.float32) if not isinstance(audio, np.ndarray) else audio for audio in batch_audio]
            with torch.no_grad():
                try:
                    results = self.pipe(
                        batch_audio,
                        batch_size=len(batch_audio),
                        generate_kwargs={"tgt_lang": "arb", 
                                        "num_beams": 4,
                                        "do_sample": False,
                                        "no_repeat_ngram_size": 2}
                    )

                    batch_hyps = []
                    for result in results:
                        if isinstance(result, dict):
                            raw_prediction = result.get("text", "")
                        elif isinstance(result, list) and len(result) > 0:
                            raw_prediction = " ".join([chunk.get("text", "") for chunk in result if isinstance(chunk, dict)])
                        else:
                            raw_prediction = str(result)
                        batch_hyps.append(raw_prediction)

                    return {"predictions": batch_hyps}
                except Exception as e:
                    LOGGER.error(f"Batch processing failed: {e}")
                    return {"predictions": ["" for _ in batch_audio]}
        
        processed_records = []
        all_refs_normalized = []
        all_hyps = []
        all_audio_paths = []
        for record in records:
            waveform = record["waveform"]
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.squeeze().numpy()

            processed_records.append({
                "waveform": waveform,
                "audio_path": record["audio_path"],
                "transcription": record["transcription"],
                "normalized_transcription": record["normalized_transcription"],
                "sample_rate": record["sample_rate"],
                "duration": record["audio_duration"]
            })

        dataset = Dataset.from_dict({
            "waveform": [r["waveform"] for r in processed_records],
            "audio_path": [r["audio_path"] for r in processed_records],
            "transcription": [r["transcription"] for r in processed_records],
            "normalized_transcription": [r["normalized_transcription"] for r in processed_records],
            "sample_rate": [r["sample_rate"] for r in processed_records],
            "duration": [r["duration"] for r in processed_records]
        })

        batch_start_time = time.time()
        total_duration = sum(r["duration"] for r in processed_records)
        
        dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=self.batch_size,
            desc="Processing Batches",
            drop_last_batch=False
        )

        batch_inference_time = time.time() - batch_start_time
        self._total_inference_time += batch_inference_time
        self._total_audio_duration += total_duration
        self._processed_count += len(records)

        for i, (audio_path, transcription, normalized_transcription, prediction, duration) in enumerate(zip(
            dataset["audio_path"], dataset["transcription"], dataset["normalized_transcription"], 
            dataset["predictions"], dataset["duration"]
        )):
            all_refs_normalized.append(normalized_transcription)
            all_hyps.append(prediction)
            all_audio_paths.append(audio_path)
            
            sample_rtf = (batch_inference_time / len(records)) / duration if duration > 0 else None
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": prediction,
                "normalized_prediction": None,
                "duration": duration,
                "inference_time": batch_inference_time / len(records),
                "rtf": sample_rtf,
            }

        all_hyps_normalized = text_processing.StandardArabicTextProcessor.main(all_hyps, substitute=True)
        self._finalize_info(all_audio_paths=all_audio_paths, all_refs_normalized=all_refs_normalized, all_hyps_normalized=all_hyps_normalized)

    @decorators.Decorators.calculate_execution_time
    def run_inference_optimized(self, records, language_ids=["arb", "ary", "arz"], duration_threshold=30.0):
        """
        Run inference using the best method depending on audio length.
        
        Args:
            records (list): List of audio record dicts
            language_ids (list): Language IDs to evaluate
            duration_threshold (float): Duration threshold for method selection
        """
        if not isinstance(records, list):
            records = [records]

        short_records = [r for r in records if r["audio_duration"] < duration_threshold]
        long_records  = [r for r in records if r["audio_duration"] >= duration_threshold]

        if short_records:
            LOGGER.info(f"Running one-by-one inference on {len(short_records)} short files (<{duration_threshold}s)")
            self.run_inference_one_by_one(short_records, language_ids)

        if long_records:
            LOGGER.info(f"Running batch inference on {len(long_records)} long files (≥{duration_threshold}s)")
            self.run_batch_inference(long_records, language_ids)

        refs = [self._samples_info[r["audio_path"]]["normalized_transcription"] 
                for r in records if r["audio_path"] in self._samples_info]
        hyps = [self._samples_info[r["audio_path"]]["normalized_prediction"] 
                for r in records if r["audio_path"] in self._samples_info]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

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
                    self._samples_info[audio_path]["metrics"] = helpers._empty_metrics()
                    continue
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

    def get_performance_summary(self):
        """Return performance metrics dict"""
        if self._total_audio_duration == 0:
            return None
        
        return {
            "average_rtf": self._total_inference_time / self._total_audio_duration if self._total_audio_duration > 0 else None,
            "total_inference_time": self._total_inference_time,
            "total_audio_duration": self._total_audio_duration,
            "processed_samples": self._processed_count
        }

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
        
        # Add performance metrics
        perf = self.get_performance_summary()
        if perf:
            LOGGER.info(f"Average RTF: {perf['average_rtf']:.4f}")
            LOGGER.info(f"Total Inference Time: {perf['total_inference_time']:.2f}s")
            LOGGER.info(f"Total Audio Duration: {perf['total_audio_duration']:.2f}s")
            LOGGER.info(f"Processed Samples: {perf['processed_samples']}")

    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0
        gc.collect()
        if self.device == "cuda":
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