from typing import Callable

import numpy as np
from attrs import define

from Layers import Layer


@define
class Activation(Layer):
    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        raise NotImplemented

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        return input_shape

    def update(self, learning_rate: float) -> None:
        return

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        return

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        return
