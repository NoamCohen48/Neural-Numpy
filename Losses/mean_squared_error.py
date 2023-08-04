import numpy as np
from attrs import define, field

from .loss import Loss


@define
class MeanSquaredError(Loss):
    expected: np.ndarray | None = field(init=False, default=None)
    predicted: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        self.predicted = input
        return self.predicted

    def calc_loss(self, expected: np.ndarray) -> float:
        assert self.predicted is not None
        self.expected = expected

        diff = self.predicted - expected
        assert isinstance(diff, np.ndarray)

        return np.square(diff).mean()

    def backward(self) -> np.ndarray:
        assert isinstance(self.expected, np.ndarray)

        diff = self.predicted - self.expected
        assert isinstance(diff, np.ndarray)
        batch_size = diff.shape[0]
        return 2 * diff / batch_size
