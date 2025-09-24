import gc

from datasets import Dataset
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm
from faster_whisper import WhisperModel, BatchedInferencePipeline
import numpy as np

from .. import LOGGER
from stt_benchmarking.utils import (
    validate, 
    decorators, 
    metrics,
    text_processing
)

class WhisperPipelineInference:
    """
    A class for loading and running inference with Whisper models using Transformers Pipeline.
    Supports both Whisper V2 and V3 with automatic chunking.
    """

    def __init__(self, device="cuda", model_version="v3", chunk_length_s=30, batch_size=16):
        """
        Initialize the WhisperPipelineInference class.

        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Whisper model to use ('v2' or 'v3')
            chunk_length_s (int): Chunk length in seconds for processing long audio
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_version = model_version.lower()
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        if self.model_version not in ["v2", "v3"]:
            raise ValueError("model_version must be either 'v2' or 'v3'")

        self.pipe = self._load_pipeline()
        self._samples_info = {}
        self._overall_metrics = None

    def _load_pipeline(self):
        """
        Load the Whisper pipeline with automatic chunking support.
        
        Returns:
            pipeline: Hugging Face pipeline for automatic speech recognition
        """
        # Set MODEL_ID based on version
        if self.model_version == "v2":
            MODEL_ID = "openai/whisper-large-v2"
        else:  # v3
            MODEL_ID = "openai/whisper-large-v3"

        # Create pipeline with automatic chunking
        pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_ID,
            device=0,
            torch_dtype=torch.float16,
            chunk_length_s=self.chunk_length_s,  # Enable automatic chunking
            stride_length_s=5,  # Overlap between chunks for better continuity
            ignore_warning=True,
            batch_size=self.batch_size,
        )

        LOGGER.info(f"Loaded Whisper {self.model_version.upper()} Pipeline with {self.chunk_length_s}s chunking")
        return pipe

    @decorators.Decorators.calculate_execution_time 
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one with automatic chunking.

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
                # Prepare audio input for pipeline
                if isinstance(waveform, torch.Tensor):
                    audio_input = waveform.squeeze().numpy()
                else:
                    audio_input = waveform.squeeze() if hasattr(waveform, 'squeeze') else waveform

                # Ensure audio is float32 and 1D
                if audio_input.dtype != np.float32:
                    audio_input = audio_input.astype(np.float32)
                
                if len(audio_input.shape) > 1:
                    audio_input = audio_input[0] if audio_input.shape[0] < audio_input.shape[1] else audio_input[:, 0]

                # Run inference with automatic chunking
                # Pipeline will automatically handle long audio by chunking
                result = self.pipe(
                    audio_input,
                    generate_kwargs={
                        "language": "arabic",  # Specify Arabic language
                        "task": "transcribe"   # Transcribe (not translate)
                    }
                )

                # Extract text from result
                if isinstance(result, dict):
                    raw_prediction = result.get("text", "")
                elif isinstance(result, list) and len(result) > 0:
                    # If timestamps are returned, concatenate all text
                    raw_prediction = " ".join([chunk.get("text", "") for chunk in result if isinstance(chunk, dict)])
                else:
                    raw_prediction = str(result)

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
                continue

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

    @decorators.Decorators.calculate_execution_time 
    def run_batch_inference(self, records):
        """
        Run batch inference using Hugging Face Dataset for better efficiency.

        Args:
            records (list): List of audio records
            batch_size (int): Number of samples to process in each batch
        """
        def process_batch(batch):
            batch_audio = batch["waveform"]
            # Ensure batch_audio is a list of NumPy arrays
            batch_audio = [np.array(audio, dtype=np.float32) if not isinstance(audio, np.ndarray) else audio for audio in batch_audio]
            with torch.no_grad():
                try:
                    # Run pipeline on batch
                    results = self.pipe(
                        batch_audio,
                        batch_size=len(batch_audio),  # Use actual batch size
                        generate_kwargs={
                            "language": "arabic",
                            "task": "transcribe"
                        }
                    )

                    # Process results
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
                
        if not isinstance(records, list):
            records = [records]

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
                "sample_rate": record["sample_rate"]
            })

        # Convert to Hugging Face Dataset
        dataset = Dataset.from_dict({
            "waveform": [r["waveform"] for r in processed_records],
            "audio_path": [r["audio_path"] for r in processed_records],
            "transcription": [r["transcription"] for r in processed_records],
            "normalized_transcription": [r["normalized_transcription"] for r in processed_records],
            "sample_rate": [r["sample_rate"] for r in processed_records]
        })


        # Process dataset in batches with explicit batch_size
        dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=self.batch_size,
            desc="Processing Batches",
            drop_last_batch=False  # Keep partial batches
        )

        # Collect results
        for i, (audio_path, transcription, normalized_transcription, prediction) in enumerate(zip(
            dataset["audio_path"], dataset["transcription"], dataset["normalized_transcription"], dataset["predictions"]
        )):
            all_refs_normalized.append(normalized_transcription)
            all_hyps.append(prediction)
            all_audio_paths.append(audio_path)
            
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": prediction,
                "normalized_prediction": None,
            }

        # Normalize predictions and finalize
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
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

        LOGGER.info("FasterWhisperInference instance has been reset. Model remains loaded.")

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info


class FasterWhisperInference:
    """
    A class for loading and running inference with FasterWhisper models.
    Supports both Whisper V2 and V3.
    """

    def __init__(self, device="cuda", model_version="v3", batch_size=32):
        """
        Initialize the FasterWhisperInference class.

        Args:
            device (str): Device to run the model on ('cuda' or 'cpu')
            model_version (str): Version of Whisper model to use ('v2' or 'v3')
        """
        self.device = device
        self.model_version = model_version.lower()
        self.batch_size = batch_size
        if self.model_version not in ["v2", "v3"]:
            raise ValueError("model_version must be either 'v2' or 'v3'")

        self.model = self._load_model()
        self.pipeline = self._load_pipeline()
        self._samples_info = {}
        self._overall_metrics = None

    def _load_model(self):
        """
        Load the FasterWhisper model.
        """
        if self.model_version == "v2":
            MODEL_ID = "large-v2"
        else:
            MODEL_ID = "large-v3"

        # FasterWhisper automatically handles device and quantization
        model = WhisperModel(MODEL_ID, device="cuda", compute_type="float16")
        LOGGER.info(f"Loaded FasterWhisper {self.model_version.upper()} Model")
        return model
    
    def _load_pipeline(self):
        """
        Load the Whisper pipeline with automatic chunking support.
        
        Returns:
            pipeline: Hugging Face pipeline for automatic speech recognition
        """
        pipe = BatchedInferencePipeline(model=self.model)
        LOGGER.info(f"Loaded Whisper {self.model_version.upper()} Pipeline")
        return pipe

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
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            all_audio_paths.append(audio_path)

            try:
                # FasterWhisper handles long audios internally
                segments, _ = self.model.transcribe(
                    audio_path,
                    language="ar",   # Arabic
                    task="transcribe"
                )

                # Concatenate all segment texts
                raw_prediction = " ".join([seg.text for seg in segments])

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
                continue

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

    @decorators.Decorators.calculate_execution_time 
    def run_batch_inference(self, records):
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
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            all_audio_paths.append(audio_path)
            try:
                # FasterWhisper handles long audios internally
                segments, _ = self.pipeline.transcribe(
                    audio_path,
                    batch_size=self.batch_size,
                    language="ar",   # Arabic
                    task="transcribe"
                )

                # Concatenate all segment texts
                raw_prediction = " ".join([seg.text for seg in segments])

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
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
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return

        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")

    def reset(self):
        self._overall_metrics = None
        self._samples_info = {}

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

        LOGGER.info("FasterWhisperInference instance has been reset. Model remains loaded.")

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