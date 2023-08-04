import numpy as np
from attrs import define, field

from .activation import Activation


@define
class Softmax(Activation):
    output: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray) -> np.ndarray:
        exp: np.ndarray = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, expected) -> np.ndarray:
        raise NotImplemented
