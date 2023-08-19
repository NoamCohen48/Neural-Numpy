import numpy as np
from attrs import define, field

from PreprocessingLayers.preprocessing_layer import PreprocessingLayer


@define
class Normalization(PreprocessingLayer):
    axis: tuple[int, ...] = field(init=True, default=(0, ))

    min_value: np.ndarray | None = field(default=None)
    max_value: np.ndarray | None = field(default=None)

    def build(self, x: np.ndarray, y: np.ndarray | None = None) -> None:
        assert isinstance(x, np.ndarray)
        self.min_value = np.min(x, axis=self.axis)
        self.max_value = np.max(x, axis=self.axis)
        self.max_value[np.isclose(self.max_value, self.min_value)] += 1

    def forward(self, input: np.ndarray) -> np.ndarray:
        assert isinstance(self.min_value, np.ndarray)
        assert isinstance(self.max_value, np.ndarray)

        assert self.min_value.shape[-1] == input.shape[-1]
        assert self.max_value.shape[-1] == input.shape[-1]

        x = (input - self.min_value) / (self.max_value - self.min_value)
        assert isinstance(x, np.ndarray)

        return x
