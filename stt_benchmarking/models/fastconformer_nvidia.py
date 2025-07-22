import nemo.collections.asr as nemo_asr
from tqdm import tqdm

from . import LOGGER
from stt_benchmarking.utils import metrics, helpers

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
    
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records one by one and compute metrics.
        
        Args:
            records (list): List of audio records containing waveform, sample_rate, 
                          transcription, and audio_path
        """
        all_refs = []
        all_hyps = []
        if isinstance(records, str):
            records = [records]
        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            try:
                transcription = record['transcription']
                audio_path = record["audio_path"]
                output = self.model.transcribe([audio_path])
                prediction = output[0].text
                predicted_sentence_clean = helpers.clean_arabic_text(prediction)
                all_refs.append(transcription)
                all_hyps.append(predicted_sentence_clean)
                sample_metrics = metrics.S2TMetrics.evaluate(refs=transcription, hyps=predicted_sentence_clean)
                self._samples_info[audio_path] = {
                    "transcription": transcription,
                    "prediction": predicted_sentence_clean,
                    "metrics": sample_metrics
                }

            except Exception as e:
                LOGGER.error(f"⚠️ Sample {i+1}, Name: {audio_path}, failed: {e}")
                LOGGER.warning(f"{transcription}, {predicted_sentence_clean}")
                continue

        self._overall_metrics = metrics.S2TMetrics.evaluate(refs=all_refs, hyps=all_hyps)

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