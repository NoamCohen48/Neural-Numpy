import numpy as np
from attrs import define, field

from PreprocessingLayers.preprocessing_layer import PreprocessingLayer


@define
class Standardization(PreprocessingLayer):
    axis: tuple[int, ...] = field(init=True, default=(0, ))

    mean: np.ndarray | None = field(init=True, kw_only=True, default=None)
    std: np.ndarray | None = field(init=True, kw_only=True, default=None)

    def build(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        assert isinstance(x, np.ndarray)
        self.mean = np.mean(x, axis=self.axis)
        self.std = np.std(x, axis=self.axis)
        self.std[self.std == 0] = 1.0

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert isinstance(self.mean, np.ndarray)
        assert isinstance(self.std, np.ndarray)

        assert self.mean.shape[-1] == input.shape[-1]
        assert self.std.shape[-1] == input.shape[-1]

        x = (input - self.mean) / self.std
        assert isinstance(x, np.ndarray)
        return x
