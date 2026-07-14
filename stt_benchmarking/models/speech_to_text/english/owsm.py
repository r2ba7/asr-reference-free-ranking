import gc
import time

import librosa
import numpy as np
import torch
from tqdm import tqdm

from . import LOGGER, NORMALIZER_OBJ
from stt_benchmarking.utils import (
    helpers,
    decorators,
    metrics
)

TARGET_SR = 16000
SHORT_FORM_DURATION = 30.0  # OWSM-CTC is trained on fixed 30s windows


class OWSMCTCFullInterface:
    """
    Loading and running inference with OWSM-CTC v3.2 via ESPnet's Speech2TextGreedySearch.
    Requires: espnet, espnet_model_zoo, librosa
    """

    def __init__(
        self,
        device,
        model_id="espnet/owsm_ctc_v3.2_ft_1B",
        lang_sym="<eng>",
        task_sym="<asr>",
        batch_size=16,
        context_len_in_secs=4,
        use_flash_attn=False,
    ):
        self.device = device if isinstance(device, str) else ("cuda" if device.type == "cuda" else "cpu")
        self.model_id = model_id
        self.lang_sym = lang_sym
        self.task_sym = task_sym
        self.batch_size = batch_size
        self.context_len_in_secs = context_len_in_secs
        self.use_flash_attn = use_flash_attn

        self.model = self._load_model()
        self._overall_metrics = None
        self._samples_info = {}

        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self):
        try:
            from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
        except ImportError as e:
            raise ImportError("espnet2 is required: pip install espnet espnet_model_zoo") from e

        model = Speech2TextGreedySearch.from_pretrained(
            self.model_id,
            device=self.device,
            use_flash_attn=self.use_flash_attn,
            generate_interctc_outputs=False,
            lang_sym=self.lang_sym,
            task_sym=self.task_sym,
        )
        LOGGER.info(f"Loaded OWSM-CTC model: {self.model_id} | lang={self.lang_sym} task={self.task_sym}")
        return model

    def _prepare_waveform(self, record):
        """
        Resample to 16kHz and return a float32 numpy array.
        OWSM-CTC batch_decode handles padding/chunking internally.
        """
        waveform = record["waveform"]
        sr = record["sample_rate"]
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        waveform = np.array(waveform, dtype=np.float32)
        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        return waveform

    @decorators.Decorators.calculate_execution_time
    def run_inference_one_by_one(self, records):
        """
        Run inference sample-by-sample using batch_decode with batch_size=1.
        Short audio (<30s) is padded to 30s by ESPnet internally.
        Long audio is chunked via buffered decoding.
        """
        all_audio_paths = []
        if not isinstance(records, list):
            records = [records]

        for i, record in tqdm(enumerate(records), total=len(records), desc="Processing Records"):
            audio_path = record["audio_path"]
            transcription = record["transcription"]
            normalized_transcription = record["normalized_transcription"]
            duration = record["audio_duration"]
            all_audio_paths.append(audio_path)

            try:
                waveform = self._prepare_waveform(record)
                start_time = time.time()
                raw_prediction = self.model.batch_decode(
                    waveform,
                    batch_size=1,
                    context_len_in_secs=self.context_len_in_secs,
                )
                # batch_decode returns str for single input
                if isinstance(raw_prediction, list):
                    raw_prediction = raw_prediction[0]
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

    @decorators.Decorators.calculate_execution_time
    def run_batch_inference(self, records):
        """
        Run batch inference using OWSM-CTC's native batch_decode.
        Passes a list of waveforms; ESPnet handles padding and chunking per sample.
        """
        all_audio_paths = []
        processed_records = []

        for record in records:
            waveform = self._prepare_waveform(record)
            processed_records.append({
                "waveform": waveform,
                "audio_path": record["audio_path"],
                "transcription": record["transcription"],
                "normalized_transcription": record["normalized_transcription"],
                "duration": record["audio_duration"],
            })

        waveforms = [r["waveform"] for r in processed_records]
        total_duration = sum(r["duration"] for r in processed_records)

        batch_start_time = time.time()
        try:
            predictions = self.model.batch_decode(
                waveforms,
                batch_size=self.batch_size,
                context_len_in_secs=self.context_len_in_secs,
            )
            # batch_decode returns list of str for list input
            if isinstance(predictions, str):
                predictions = [predictions]
        except Exception as e:
            LOGGER.error(f"Batch inference failed: {e}")
            predictions = ["" for _ in processed_records]

        batch_inference_time = time.time() - batch_start_time
        self._total_inference_time += batch_inference_time
        self._total_audio_duration += total_duration
        self._processed_count += len(records)

        per_sample_time = batch_inference_time / len(processed_records)

        for record, raw_prediction in zip(processed_records, predictions):
            audio_path = record["audio_path"]
            duration = record["duration"]
            all_audio_paths.append(audio_path)
            normalized_prediction = NORMALIZER_OBJ(raw_prediction)
            self._samples_info[audio_path] = {
                "raw_transcription": record["transcription"],
                "normalized_transcription": record["normalized_transcription"],
                "raw_prediction": raw_prediction,
                "normalized_prediction": normalized_prediction,
                "duration": duration,
                "inference_time": per_sample_time,
                "rtf": per_sample_time / duration if duration > 0 else None,
            }

        self._finalize_info(all_audio_paths=all_audio_paths)

    @decorators.Decorators.calculate_execution_time
    def run_inference_optimized(self, records, duration_threshold=30.0):
        """
        Route records to one-by-one or batch inference based on audio duration.
        For OWSM-CTC, batch_decode handles both short and long form,
        so the split is purely for RTF tracking granularity.

        Args:
            records (list): List of audio record dicts
            duration_threshold (float): Seconds above which batch path is used
        """
        if not isinstance(records, list):
            records = [records]

        short_records = [r for r in records if r["audio_duration"] < duration_threshold]
        long_records = [r for r in records if r["audio_duration"] >= duration_threshold]

        if short_records:
            LOGGER.info(f"Running one-by-one inference on {len(short_records)} short files (<{duration_threshold}s)")
            self.run_inference_one_by_one(short_records)

        if long_records:
            LOGGER.info(f"Running batch inference on {len(long_records)} long files (>={duration_threshold}s)")
            self.run_batch_inference(long_records)

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def _finalize_info(self, all_audio_paths):
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
                    self._samples_info[audio_path] = {"metrics": helpers._empty_metrics()}
            else:
                self._samples_info[audio_path] = {"metrics": helpers._empty_metrics()}

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def get_performance_summary(self):
        if self._total_audio_duration == 0:
            return None
        return {
            "average_rtf": self._total_inference_time / self._total_audio_duration,
            "total_inference_time": self._total_inference_time,
            "total_audio_duration": self._total_audio_duration,
            "processed_samples": self._processed_count,
        }

    def summary_of_evaluation(self):
        if not self._overall_metrics:
            LOGGER.warning("No evaluation metrics available. Run inference first.")
            return
        LOGGER.info("Overall Evaluation Summary:")
        for k, v in self._overall_metrics.items():
            LOGGER.info(f"{k}: {v}")
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
        LOGGER.info("OWSMCTCFullInterface instance has been reset. Model remains loaded.")

    @property
    def overall_metrics(self):
        return self._overall_metrics

    @property
    def samples_info(self):
        return self._samples_info