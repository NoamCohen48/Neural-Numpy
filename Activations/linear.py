import numpy as np
from attrs import define, field

from .activation import Activation


@define
class Linear(Activation):

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        return input

    def backward(self, output_gradient) -> np.ndarray:
        return output_gradient
