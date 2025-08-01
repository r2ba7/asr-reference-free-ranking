import nemo.collections.asr as nemo_asr
from tqdm import tqdm

from . import LOGGER
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics
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
            try:
                audio_path = record['audio_path']
                transcription = record['transcription']
                normalized_transcription = record['normalized_transcription']
                all_audio_paths.append(audio_path)
                output = self.model.transcribe([audio_path])
                raw_prediction = output[0].text

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
                continue


            all_refs_normalized.append(normalized_transcription)
            all_hyps.append(raw_prediction)
            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": raw_prediction,
                "normalized_prediction": None
            }

        all_hyps_normalized = text_processing.ArabicTextProcessor.process_texts(all_hyps)     
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                self._samples_info[audio_path]["normalized_prediction"] = all_hyps_normalized[i]
                sample_metrics = metrics.FilteredS2TMetrics.evaluate(
                    refs=all_refs_normalized[i],
                    hyps=all_hyps_normalized[i],
                    single_sample=True
                )
                self._samples_info[audio_path]["metrics"] = sample_metrics

        self._overall_metrics = metrics.FilteredS2TMetrics.evaluate(refs=all_refs_normalized, hyps=all_hyps_normalized)

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