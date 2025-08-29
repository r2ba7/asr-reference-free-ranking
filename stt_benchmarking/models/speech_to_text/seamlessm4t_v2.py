import gc
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
from tqdm import tqdm
import numpy as np
from difflib import SequenceMatcher

from .. import LOGGER
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics
)

# ----------------------------
# Helper function: merge partial
# ----------------------------
def merge_partial(prev_text, curr_text):
    """Merge two strings removing duplicated words at the boundary."""
    s = SequenceMatcher(None, prev_text.split(), curr_text.split())
    match = s.get_matching_blocks()[-1]
    overlap_idx = match.a + match.size
    merged = " ".join(prev_text.split() + curr_text.split()[overlap_idx:])
    return merged

# ----------------------------
# Main class
# ----------------------------
class SeamlessM4TInference:
    """SeamlessM4T inference with context-aware chunking."""
    
    def __init__(self, device, model_version="v2"):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_version = model_version.lower()
        
        if self.model_version not in ['v2']:
            raise ValueError("model_version must be 'v2'")
        
        self.model, self.processor = self._load_model()
        self.dtype = self.model.dtype
        self._overall_metrics = None
        self._samples_info = {}

    def _load_model(self):
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

    # ----------------------------
    # Audio chunking
    # ----------------------------
    @staticmethod
    def chunk_audio(waveform, sample_rate, chunk_duration=5.0, overlap=0.3):
        """Split waveform into overlapping chunks."""
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform)
        chunk_samples = int(chunk_duration * sample_rate)
        hop_samples = int(chunk_samples * (1 - overlap))
        chunks = []
        start = 0
        while start < waveform.shape[0]:
            end = min(start + chunk_samples, waveform.shape[0])
            chunks.append(waveform[start:end])
            if end == waveform.shape[0]:
                break
            start += hop_samples
        return chunks

    # ----------------------------
    # Inference with context-aware chunking
    # ----------------------------
    @decorators.Decorators.calculate_execution_time
    def run_inference_one_by_one(self, records, chunk_audio_flag=True, chunk_duration=5.0, overlap=0.3, context_words=15):
        all_refs_normalized = []
        all_hyps_normalized = []
        all_audio_paths = []

        if not isinstance(records, list):
            records = [records]

        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            waveform = record["waveform"]
            sample_rate = record["sample_rate"]
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            all_audio_paths.append(audio_path)

            try:
                # Chunk audio
                chunks = [waveform]
                if chunk_audio_flag:
                    chunks = self.chunk_audio(waveform, sample_rate, chunk_duration, overlap)

                # Sequential inference with context
                merged_prediction = ""
                prev_words = []

                for chunk in chunks:
                    if isinstance(chunk, np.ndarray):
                        chunk = torch.tensor(chunk)
                    inputs = self.processor(
                        audios=chunk,
                        sampling_rate=sample_rate,
                        return_tensors="pt"
                    ).to(self.device, dtype=self.dtype)

                    with torch.no_grad():
                        output = self.model.generate(**inputs, generate_speech=False, tgt_lang="arb")

                    chunk_pred = self.processor.decode(output[0][0].tolist(), skip_special_tokens=True)

                    # Prepend previous context words
                    if prev_words:
                        chunk_pred = " ".join(prev_words) + " " + chunk_pred

                    # Merge to remove duplication
                    if merged_prediction:
                        merged_prediction = merge_partial(merged_prediction, chunk_pred)
                    else:
                        merged_prediction = chunk_pred

                    # Update prev_words for next chunk
                    prev_words = chunk_pred.split()[-context_words:]

                # Normalize final prediction
                normalized_prediction = text_processing.StandardArabicTextProcessor.normalize_texts(merged_prediction)

                # Save results
                self._samples_info[audio_path] = {
                    "raw_transcription": transcription,
                    "normalized_transcription": normalized_transcription,
                    "raw_prediction": merged_prediction,
                    "normalized_prediction": normalized_prediction,
                }

                all_refs_normalized.append(normalized_transcription)
                all_hyps_normalized.append(normalized_prediction)

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
                continue

        # Compute overall metrics
        self._finalize_info(all_audio_paths, all_refs_normalized, all_hyps_normalized)
        self._overall_metrics = metrics.StandardSTTMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)

    # ----------------------------
    # Finalize per-sample metrics
    # ----------------------------
    def _finalize_info(self, all_audio_paths, all_refs_normalized, all_hyps_normalized):
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    sample_metrics = metrics.StandardSTTMetrics.evaluate(
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
                        "average_score": None,
                    }

    # ----------------------------
    # Summary and reset
    # ----------------------------
    def summary_of_evaluation(self):
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
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        LOGGER.info("SeamlessM4T instance has been reset. Model and processor remain loaded.")

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info