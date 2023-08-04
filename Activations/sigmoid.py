import numpy as np
from attrs import define, field

from .activation import Activation


def sigmoid(x) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


@define
class Sigmoid(Activation):
    input: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return sigmoid(input)

    def backward(self, output_gradient) -> np.ndarray:
        sig = sigmoid(self.input)
        return sig * (1 - sig) * output_gradient
