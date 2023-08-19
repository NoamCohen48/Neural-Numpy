import numpy as np
from attrs import define


@define
class PreprocessingLayer:

    def build(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        raise NotImplemented

    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)
