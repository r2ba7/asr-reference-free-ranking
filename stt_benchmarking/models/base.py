import torch

from . import LOGGER
from . import seamlessm4t_v2, whisper, xlsr_53


class ModelSelection:
    def __init__(self, device, model_type, **kwargs):
        if model_type == "whisper":
            if "model_version" not in kwargs:
                raise ValueError("`model_version` must be specified (e.g., 'v2' or 'v3')")

            model_version = kwargs["model_version"]
            self.model_object = whisper.WhisperInference(device=device, model_version=model_version)
        elif model_type == "seamlessm4t":
            self.model_object = seamlessm4t_v2.SeamlessM4TInference(device=device)
        else: #xlsr_53
            self.model_object = xlsr_53.XLSRInference(device=device)

    def evaluate_model(self, records):
        self.model_object.run_inference_one_by_one(records=records)

    def get_summary(self):
        self.model_object.summary_of_evaluation()

    def get_metrics(self):
        return self.model_object._overall_metrics