from typing import Callable

import numpy as np
from attrs import define, field


@define
class Layer:
    input_shape: tuple | None = field(init=False, default=None)
    output_shape: tuple | None = field(init=False, default=None)

    def build(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        raise NotImplemented

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        raise NotImplemented

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def update(self, learning_rate: float) -> None:
        raise NotImplemented

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        raise NotImplemented

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        raise NotImplemented
