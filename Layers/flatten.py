from attrs import define, field
from typing import Callable

from Activations.activation import Activation
from .layer import Layer
import numpy as np


@define
class Flatten(Layer):
    input_shape: tuple[int, ...] | None = field(init=False, default=None)

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        self.input_shape = input.shape
        batch_size = input.shape[0]
        return input.reshape(batch_size, -1)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.input_shape is not None
        return output_gradient.reshape(*self.input_shape)

    def update(self, learning_rate: float) -> None:
        return

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return
