from typing import Any

import numpy as np
from attrs import define


@define
class Metric:

    @property
    def name(self) -> str:
        raise NotImplemented

    def __call__(self, model_outputs: np.ndarray, true: np.ndarray) -> float:
        return self.calculate(model_outputs, true)

    def calculate(self, model_outputs: np.ndarray, true: np.ndarray) -> float:
        raise NotImplemented
