import numpy as np
from attrs import define, field

from .activation import Activation


@define
class ReLU(Activation):
    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.maximum(input, 0)

    def backward(self, output_gradient) -> np.ndarray:
        assert self.input is not None
        return np.where(self.input > 0, output_gradient, 0)
