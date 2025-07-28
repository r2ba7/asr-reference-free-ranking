import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm

from . import LOGGER
from stt_benchmarking.utils import (
    postprocess,
    helpers, 
    decorators, 
    metrics
)


class RDIInference:

    URL = "http://34.57.97.217:6011/recognize"
    DATA = {"format": "json", "enable_ctm": "false", "model_version": "regular/Arabic/latest",}

    def __init__(self):
        self._overall_metrics = None
        self._samples_info = {}

    @decorators.Decorators.timeout_with_retry
    def process_single_file(self, record):
        try:
            transcription = record['transcription']
            normalized_transcription = record['normalized_transcription']
            audio_path = record["audio_path"]
            with open(audio_path, "rb") as audio_file:
                files = {"file": audio_file}
                response = requests.post(RDIInference.URL, data=RDIInference.DATA, files=files)
            
            if response.status_code == 200:
                result = response.json()
                raw_prediction = result['text']  # Keep raw output for now
                self._samples_info[audio_path] = {
                    "raw_transcription": transcription,
                    "normalized_transcription": normalized_transcription,
                    "raw_prediction": raw_prediction,
                    "normalized_prediction": None
                }
                
                return normalized_transcription, raw_prediction, audio_path
            
            else:
                raise Exception(f"HTTP {response.status_code} - retrying...")
                
        except Exception as e:
            raise e
    
    @decorators.Decorators.calculate_execution_time
    def run_inference(self, records):
        all_refs = []
        all_hyps_raw = []
        all_audio_paths = []

        if not isinstance(records, list):
            records = [records]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.process_single_file, record) for record in records]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing audio files"):
                ref, hyp_raw, audio_path = future.result()
                all_refs.append(ref)
                all_hyps_raw.append(hyp_raw)
                all_audio_paths.append(audio_path)
        
        all_hyps_normalized = postprocess.normalize_text(all_hyps_raw)
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                self._samples_info[audio_path]["normalized_prediction"] = all_hyps_normalized[i]
                sample_metrics = metrics.S2TMetrics.evaluate(
                    refs=all_refs[i],
                    hyps=all_hyps_normalized[i]
                )
                self._samples_info[audio_path]["metrics"] = sample_metrics
        
        self._overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps_normalized)

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

    @property
    def samples_info(self):
        return self._samples_info
    
    @property
    def overall_metrics(self):
        return self._overall_metrics


