import nemo.collections.asr as nemo_asr
from tqdm import tqdm
import gc

from .. import LOGGER
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics,
    validate
)

class Fastconformer_hybridInference:
    def __init__(self):
        """
        Initialize the HubertArabicInference class.

        Args:
            device (str or torch.device): Device to run the model on ('cuda' or 'cpu')
        """
        self.model = self._load_model()
        self._overall_metrics = None
        self._samples_info = {}

    def _load_model(self):
        """
        Load the HuBERT Arabic model and processor.

        Returns:
            tuple: (model, processor)
        """
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0")
        LOGGER.info(f"Loaded model stt conformer hybrid")
        return model
    
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
            audio_path = record['audio_path']
            transcription = record['transcription']
            normalized_transcription = record['normalized_transcription']
            all_audio_paths.append(audio_path)
            try:
                output = self.model.transcribe([audio_path])
                raw_prediction = output[0].text
                if not raw_prediction or raw_prediction.strip() == "":
                   raise ValueError("Empty transcription result")

            except Exception as e:
                LOGGER.error(f"Sample {i+1}, Name: {audio_path}, failed: {e}")
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
        """
        Reset the inference results and metrics without reinitializing the model.
        This clears all stored results from previous inference runs while keeping
        the loaded model intact.
        """
        self._overall_metrics = None
        self._samples_info = {}
        
        # Optional: Clear GPU cache if CUDA is available
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    
        LOGGER.info("Fastconformer_hybridInference instance has been reset. Model remains loaded.")

    @property
    def samples_info(self):
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics