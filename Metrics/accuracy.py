from typing import Any

import numpy as np
from attrs import define

from .metric import Metric


@define
class Accuracy(Metric):
    @property
    def name(self) -> str:
        return "Accuracy"

    def calculate(self, model_outputs: np.ndarray, true: np.ndarray) -> float:
        predicted_labels = np.argmax(model_outputs, axis=1)
        same = predicted_labels == true
        accuracy = np.mean(same)
        assert isinstance(accuracy, float)
        return accuracy
