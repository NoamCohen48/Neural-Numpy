import numpy as np
import torch
from attrs import define, field
from torch import nn

from .loss import Loss


@define
class LogSoftmaxCrossEntropy(Loss):
    input_shape: tuple[int, ...] | None = field(init=False, default=None)

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        self.input_shape = input.shape
        return input

    def calc_loss(self, expected: np.ndarray) -> float:
        return 0

    def backward(self) -> np.ndarray:
        assert isinstance(self.input_shape, tuple)
        return np.zeros(self.input_shape)
