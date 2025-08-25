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
                    return_tensors="pt"
                    )
                input_features = inputs["input_features"].to(self.device, dtype=self.dtype)
                with torch.no_grad():
                    generated_ids = self.model.generate(input_features=input_features)

                raw_prediction = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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

        all_hyps_normalized = text_processing.StandardArabicTextProcessor.normalize_texts(all_hyps)
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    self._samples_info[audio_path]["normalized_prediction"] = all_hyps_normalized[i]
                    sample_metrics = metrics.StandardSTTMetrics.evaluate(
                        refs=all_refs_normalized[i],
                        hyps=all_hyps_normalized[i],
                        single_sample=True
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    LOGGER.info(f"{self._samples_info[audio_path]}")
                    self._samples_info[audio_path]['metrics'] = {"word_accuracy": None, "char_accuracy": None, "average_score": None}
                    continue
                
        self._overall_metrics = metrics.StandardSTTMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)

        
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


# class OptimizedWhisperInference:
#     """
#     Optimized class for running inference with Whisper models using faster-whisper.
#     Supports batch processing, quantization, and chunking for long audio.
#     """
    
#     def __init__(self, device="cuda", model_version="v3", use_chunking=True, chunk_length=30, overlap=2, batch_size=4):
#         """
#         Initialize the WhisperInference class.
        
#         Args:
#             device (str): Device to run the model on ('cuda' or 'cpu')
#             model_version (str): Version of Whisper model ('v2' or 'v3')
#             use_chunking (bool): Whether to use chunking for long audio
#             chunk_length (int): Length of each chunk in seconds
#             overlap (int): Overlap between chunks in seconds
#             batch_size (int): Number of audio samples to process in a batch
#         """
#         self.device = device if isinstance(device, torch.device) else torch.device(device)
#         self.model_version = model_version.lower()
#         if self.model_version not in ['v2', 'v3']:
#             raise ValueError("model_version must be either 'v2' or 'v3'")
        
#         self.use_chunking = use_chunking
#         self.chunk_length = chunk_length
#         self.overlap = overlap
#         self.batch_size = batch_size
        
#         self.model = self._load_model()
#         self._samples_info = {}
#         self._overall_metrics = None

#     def _load_model(self):
#         """
#         Load the faster-whisper model with quantization.
        
#         Returns:
#             WhisperModel: Loaded model
#         """
#         MODEL_ID = "large-v2" if self.model_version == "v2" else "large-v3"
#         try:
#             model = WhisperModel(
#                 MODEL_ID,
#                 device=self.device.type,
#                 compute_type="float16" if self.device.type == "cuda" else "float32",
#                 local_files_only=False,
#                 cpu_threads=4 if self.device.type == "cpu" else 1
#             )
#             LOGGER.info(f"Loaded Faster-Whisper {self.model_version.upper()} Model")
#             return model
#         except Exception as e:
#             LOGGER.error(f"Failed to load model: {e}")
#             raise

#     def _chunk_audio(self, waveform, sample_rate):
#         """
#         Split audio into overlapping chunks.
        
#         Args:
#             waveform: Audio waveform (numpy array)
#             sample_rate: Sample rate
        
#         Returns:
#             List of audio chunks with metadata
#         """
#         chunk_samples = int(self.chunk_length * sample_rate)
#         overlap_samples = int(self.overlap * sample_rate)
#         step = chunk_samples - overlap_samples
        
#         chunks = []
#         start = 0
        
#         while start < len(waveform):
#             end = min(start + chunk_samples, len(waveform))
#             chunk = waveform[start:end]
            
#             chunks.append({
#                 'audio': chunk,
#                 'start_time': start / sample_rate,
#                 'end_time': end / sample_rate,
#                 'is_last': end >= len(waveform)
#             })
            
#             if end >= len(waveform):
#                 break
#             start += step
        
#         return chunks

#     def _merge_transcriptions(self, transcriptions):
#         """
#         Merge overlapping transcriptions, handling duplicates.
        
#         Args:
#             transcriptions: List of transcription dictionaries with 'text', 'start_time', 'end_time'
        
#         Returns:
#             str: Merged transcription
#         """
#         if not transcriptions:
#             return ""
        
#         if len(transcriptions) == 1:
#             return transcriptions[0]['text']
        
#         merged_text = transcriptions[0]['text']
        
#         for i in range(1, len(transcriptions)):
#             current_text = transcriptions[i]['text']
#             prev_words = merged_text.split()
#             curr_words = current_text.split()
            
#             max_overlap = min(8, len(prev_words), len(curr_words))
#             overlap_found = 0
            
#             for overlap_len in range(max_overlap, 0, -1):
#                 if prev_words[-overlap_len:] == curr_words[:overlap_len]:
#                     overlap_found = overlap_len
#                     break
            
#             if overlap_found > 0:
#                 merged_text += " " + " ".join(curr_words[overlap_found:])
#             else:
#                 merged_text += " " + current_text
        
#         return merged_text.strip()

#     def _transcribe_with_chunking(self, waveform, sample_rate):
#         """
#         Transcribe audio using chunking for long audio files.
        
#         Args:
#             waveform: Audio waveform (numpy array)
#             sample_rate: Sample rate
        
#         Returns:
#             str: Transcribed text
#         """
#         audio_duration = len(waveform) / sample_rate
#         if audio_duration <= self.chunk_length or not self.use_chunking:
#             try:
#                 segments, _ = self.model.transcribe(
#                     waveform,
#                     language="ar",
#                     beam_size=5,
#                     vad_filter=True,
#                     vad_parameters=dict(min_silence_duration_ms=500)
#                 )
#                 transcription = " ".join(segment.text for segment in segments)
#                 return transcription.strip()
#             except Exception as e:
#                 LOGGER.error(f"Failed to transcribe audio: {e}")
#                 return ""
        
#         chunks = self._chunk_audio(waveform, sample_rate)
#         all_transcriptions = []
        
#         for i, chunk_data in enumerate(chunks):
#             try:
#                 segments, _ = self.model.transcribe(
#                     chunk_data['audio'],
#                     language="ar",
#                     beam_size=5,
#                     vad_filter=True,
#                     vad_parameters=dict(min_silence_duration_ms=500)
#                 )
#                 chunk_transcription = " ".join(segment.text for segment in segments)
#                 all_transcriptions.append({
#                     'text': chunk_transcription.strip(),
#                     'start_time': chunk_data['start_time'],
#                     'end_time': chunk_data['end_time']
#                 })
#                 gc.collect()
#                 if self.device.type == "cuda":
#                     torch.cuda.empty_cache()
#             except Exception as e:
#                 LOGGER.error(f"Failed to process chunk {i}: {e}")
#                 continue
        
#         return self._merge_transcriptions(all_transcriptions)

#     def run_inference_batched(self, records):
#         """
#         Run inference on audio records in batches and compute metrics.
        
#         Args:
#             records (list): List of audio records containing waveform, sample_rate, 
#                           transcription, normalized_transcription, and audio_path
#         """
#         all_refs_normalized = []
#         all_hyps = []
#         all_audio_paths = []
#         errors = []
        
#         if not isinstance(records, list):
#             records = [records]
        
#         for i in range(0, len(records), self.batch_size):
#             batch_records = records[i:i + self.batch_size]
#             batch_waveforms = [r["waveform"] for r in batch_records]
#             batch_sample_rate = batch_records[0]["sample_rate"]
            
#             for j, record in enumerate(batch_records):
#                 try:
#                     transcription = self._transcribe_with_chunking(record["waveform"], record["sample_rate"])
#                     all_refs_normalized.append(record["normalized_transcription"])
#                     all_hyps.append(transcription)
#                     all_audio_paths.append(record["audio_path"])
#                     self._samples_info[record["audio_path"]] = {
#                         "raw_transcription": record["transcription"],
#                         "normalized_transcription": record["normalized_transcription"],
#                         "raw_prediction": transcription,
#                         "normalized_prediction": None
#                     }
#                 except Exception as e:
#                     errors.append(f"Sample {i+j+1}, Name: {record['audio_path']}, failed: {e}")
#                     continue
            
#             gc.collect()
#             if self.device.type == "cuda":
#                 torch.cuda.empty_cache()
        
#         if errors:
#             LOGGER.error(f"Encountered {len(errors)} errors: {errors}")
        
#         all_hyps_normalized = text_processing.StandardArabicTextProcessor.normalize_texts(all_hyps)
#         for i, audio_path in enumerate(all_audio_paths):
#             try:
#                 self._samples_info[audio_path]["normalized_prediction"] = all_hyps_normalized[i]
#                 sample_metrics = metrics.FilteredS2TMetrics.evaluate(
#                     refs=all_refs_normalized[i],
#                     hyps=all_hyps_normalized[i],
#                     single_sample=True
#                 )
#                 self._samples_info[audio_path]["metrics"] = sample_metrics
#             except Exception as e:
#                 LOGGER.error(f"Error processing metrics for sample {i+1}, Name: {audio_path}, failed: {e}")
#                 self._samples_info[audio_path]["metrics"] = {
#                     "word_accuracy": None,
#                     "char_accuracy": None,
#                     "average_score": None
#                 }
                
#         self.reorder_samples_info(records=records)
#         self._overall_metrics = metrics.FilteredS2TMetrics.evaluate(
#             refs=all_refs_normalized,
#             hyps=all_hyps_normalized
#         )

#     def run_inference_parallel(self, records, max_workers=4):
#         """
#         Run inference in parallel using multiple processes.
        
#         Args:
#             records (list): List of audio records
#             max_workers (int): Number of parallel workers
#         """
#         if not isinstance(records, list):
#             records = [records]
        
#         chunk_size = max(1, len(records) // max_workers)
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = [
#                 executor.submit(self.run_inference_batched, records[i:i + chunk_size])
#                 for i in range(0, len(records), chunk_size)
#             ]
#             for future in tqdm(futures, total=len(futures), desc="Processing Records in Parallel"):
#                 future.result()

#     def reorder_samples_info(self, records):
#         ordered_info = OrderedDict()
#         for record in records:
#             audio_path = record["audio_path"]
#             if audio_path in self._samples_info:
#                 ordered_info[audio_path] = self._samples_info[audio_path]
#         self._samples_info = ordered_info

#     def summary_of_evaluation(self):
#         """
#         Display a summary of the overall evaluation metrics.
#         """
#         if not self._overall_metrics:
#             LOGGER.warning("No evaluation metrics available. Run inference first.")
#             return
        
#         LOGGER.info("Overall Evaluation Summary:")
#         for k, v in self._overall_metrics.items():
#             LOGGER.info(f"{k}: {v}")

#     def reset(self):
#         """
#         Reset the inference state and clear memory.
#         """
#         self._overall_metrics = None
#         self._samples_info = {}
#         gc.collect()
#         if self.device.type == "cuda":
#             torch.cuda.empty_cache()
#         LOGGER.info("OptimizedWhisperInference instance has been reset.")

#     @property
#     def overall_metrics(self):
#         return self._overall_metrics

#     @property
#     def samples_info(self):
#         return self._samples_info

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