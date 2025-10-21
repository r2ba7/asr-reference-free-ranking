import gc
import time
import json
import random
from tqdm import tqdm
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

from .. import LOGGER
from stt_benchmarking.utils import (
    text_processing, 
    decorators, 
    metrics,
    helpers
)

ASR_EXAMPLE_PROMPTS = [
    "هل تقدر تعطيني نص هالصوت؟",
    "ممكن ترسل لي تفريغ هالتسجيل؟",
    "ماذا يقول المتحدث؟",
    "تعطيني نسخة مكتوبة من المقطع؟",
    "تكدر تحوّل هالصوت لنص؟",
    "سَوِ تحويل للصوت إلى نص مكتوب.",
    "أعطني تفريغ كلام المتكلّم بالصوت.",
    "هات لي نص التسجيل.",
    "أبغى نص الكلام الموجود بالصوت.",
    "حَوِّل هذا المقطع الصوتي إلى نص.",
]


class QwenOmniInference:
    def __init__(self, device="cuda"):
        """
        Initialize the QwenOmniInference class (GPU inference, no memory tracking).
        """
        self._overall_metrics = None
        self._samples_info = {}
        self.device = torch.device(device)
        self.model, self.processor = self._load_model()
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0

    def _load_model(self):
        """
        Load the Qwen Omni model and processor.
        """
        model_path = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, dtype="auto")
        processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        model.to(self.device)
        LOGGER.info("Loaded model Qwen3-Omni-30B-A3B-Instruct")
        return model, processor

    def _process_and_infer(self, audio, prompt):
        """
        Perform inference on a single audio file.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True
        )
        inputs = inputs.to(self.device).to(self.model.dtype)

        text_ids, _ = self.model.generate(
            **inputs,
            max_new_tokens=512,
            speaker="Ethan",
            thinker_return_dict_in_generate=True,
            use_audio_in_video=True
        )

        text = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return text[0]

    @decorators.Decorators.calculate_execution_time
    def run_inference_one_by_one(self, records):
        """
        Run inference on audio records sequentially and compute metrics.
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
                start_time = time.time()
                prompt = random.choice(ASR_EXAMPLE_PROMPTS) + " أَخْرِج التَّفْرِيغ الصَّوتي فَقَط."
                prediction = self._process_and_infer(audio_path, prompt)
                inference_time = time.time() - start_time

                if self._processed_count > 4:
                    self._total_inference_time += inference_time
                    self._total_audio_duration += duration

                self._processed_count += 1
            except Exception as e:
                LOGGER.error(f"Sample {i+1}, Name: {audio_path}, failed: {e}")
                inference_time = None
                prediction = ""

            self._samples_info[audio_path] = {
                "raw_transcription": transcription,
                "normalized_transcription": normalized_transcription,
                "raw_prediction": prediction,
                "normalized_prediction": None,
                "duration": duration,
                "inference_time": inference_time,
                "rtf": inference_time / duration if (inference_time and duration > 0) else None,
            }

        self._finalize_info(all_audio_paths=all_audio_paths)

    def _finalize_info(self, all_audio_paths):
        """
        Normalize predictions and compute metrics per sample.
        """
        for i, audio_path in enumerate(all_audio_paths):
            if audio_path in self._samples_info:
                try:
                    prediction = self._samples_info[audio_path]["raw_prediction"]
                    self._samples_info[audio_path]["normalized_prediction"] = text_processing.StandardArabicTextProcessor.main(
                        prediction, substitute=True
                    )
                    sample_metrics = metrics.BasicSTTMetrics.evaluate(
                        refs=self._samples_info[audio_path]["normalized_transcription"],
                        hyps=self._samples_info[audio_path]["normalized_prediction"],
                    )
                    self._samples_info[audio_path]["metrics"] = sample_metrics
                except Exception as e:
                    LOGGER.error(f"Error processing sample {i+1}, Name: {audio_path}, failed: {e}")
                    self._samples_info[audio_path]["normalized_prediction"] = None
                    self._samples_info[audio_path]["metrics"] = helpers._empty_metrics()
            else:
                self._samples_info[audio_path] = {
                    "normalized_prediction": None,
                    "metrics": helpers._empty_metrics()
                }

        refs = [v["normalized_transcription"] for v in self._samples_info.values()]
        hyps = [v["normalized_prediction"] for v in self._samples_info.values()]
        self._overall_metrics = metrics.BasicSTTMetrics.evaluate(refs=refs, hyps=hyps)

    def get_performance_summary(self):
        """Return performance metrics dict."""
        if self._total_audio_duration == 0:
            return None
        return {
            "average_rtf": self._total_inference_time / self._total_audio_duration if self._total_audio_duration > 0 else None,
            "total_inference_time": self._total_inference_time,
            "total_audio_duration": self._total_audio_duration,
            "processed_samples": self._processed_count
        }

    def summary_of_evaluation(self):
        """Display evaluation metrics summary."""
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
        """Reset results and metrics, keep model loaded."""
        self._overall_metrics = None
        self._samples_info = {}
        self._total_inference_time = 0.0
        self._total_audio_duration = 0.0
        self._processed_count = 0
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        LOGGER.info("QwenOmniInference instance has been reset. Model remains loaded.")

    @property
    def samples_info(self):
        return self._samples_info

    @property
    def overall_metrics(self):
        return self._overall_metrics