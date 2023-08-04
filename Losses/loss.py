import numpy as np
from attrs import define


@define
class Loss:
    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        raise NotImplemented

    def calc_loss(self, expected: np.ndarray) -> float:
        raise NotImplemented

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        raise NotImplemented

    def backward(self) -> np.ndarray:
        raise NotImplemented

    def __call__(self, expected: np.ndarray) -> float:
        return self.calc_loss(expected)
