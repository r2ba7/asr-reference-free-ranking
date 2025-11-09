import gc
import time

import torch
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel

from . import LOGGER, NORMALIZER_OBJ
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics,
    helpers
)

class NvidiaInference:
    def __init__(self, device, model_id):
        """
        Initialize the HubertArabicInference class.

        Args:
            device (str or torch.device): Device to run the model on ('cuda' or 'cpu')
            model_id: either 'canary', 'parakeet', 'conformer'
        """
        self._overall_metrics = None
        self._samples_info = {}
        self.device = torch.device(device)
        self.model = self._load_model(model_id)
        # Add timing/memory tracking
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self, model_id):
        """
        Load the HuBERT Arabic model and processor.

        Returns:
            tuple: (model, processor)
        """
        if model_id == "canary":
            model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2", map_location=self.device)
        elif model_id == "parakeet":
            model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2", map_location=self.device)
        elif model_id == "conformer":
            model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/stt_en_conformer_ctc_large")
        else:
            raise ValueError("Wrong nvidia model, one of 'canary', 'parakeet', 'conformer'")
        LOGGER.info(f"Loaded model nvidia {model_id}")
        return model
    
    @decorators.Decorators.calculate_execution_time
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one and compute metrics.
        
        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_audio_paths = []
        if not isinstance(records, list): records = [records]
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            audio_path = record['audio_path']
            transcription = record['transcription']
            normalized_transcription = record['normalized_transcription']
            duration = record["audio_duration"]
            all_audio_paths.append(audio_path)
            try:
                # Time inference
                start_time = time.time()
                output = self.model.transcribe([audio_path], source_lang='en', target_lang='en', return_hypotheses=True, batch_size=64)
                hypothesis = output[0]
                raw_prediction = hypothesis.text
                normalized_prediction = NORMALIZER_OBJ(raw_prediction)
                inference_time = time.time() - start_time
                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration
                
                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"Sample {i+1}, Name: {audio_path}, failed: {e}")
                inference_time = None
                raw_prediction = ""
                normalized_prediction = ""

            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized_prediction,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if (inference_time and duration > 0) else None,
            }
                
        self._finalize_info(all_audio_paths=all_audio_paths)
        
    def _finalize_info(self, all_audio_paths):
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
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(
                        refs=self._samples_info[audio_path]["normalized_transcription"],
                        hyps=self._samples_info[audio_path]["normalized_prediction"],
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path] = {
                        "metrics":  helpers._empty_metrics()
                    }
            else:
                self._samples_info[audio_path] = {
                    "metrics":  helpers._empty_metrics()
                }

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)
                
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
        """
        Reset the inference results and metrics without reinitializing the model.
        This clears all stored results from previous inference runs while keeping
        the loaded model intact.
        """
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
    
        LOGGER.info("Fastconformer_hybridInference instance has been reset. Model remains loaded.")

    @property
    def samples_info(self):
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics