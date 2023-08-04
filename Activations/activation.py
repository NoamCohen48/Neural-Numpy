import numpy as np
from attrs import define


@define
class Activation:
    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplemented

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplemented
