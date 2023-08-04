import numpy as np
import torch
from attrs import define, field
from torch import nn

from .loss import Loss


@define
class LogSoftmaxCrossEntropy(Loss):
    expected: np.ndarray | None = field(init=False, default=None)
    predicted: np.ndarray | None = field(init=False, default=None)

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        # log softmax(L) = L - log(np.sum(exp(L)))

        with np.errstate(divide='ignore'):
            # taking the max in each row
            x_max = np.max(input, axis=1, keepdims=True)
            # if infinite replace with 0
            x_max[~np.isfinite(x_max)] = 0
            # shifting the values
            shifted = input - x_max
            exp: np.ndarray = np.exp(shifted)
            self.predicted = shifted - np.log(np.sum(exp, axis=1, keepdims=True))
            assert isinstance(self.predicted, np.ndarray)

        return self.predicted

    def calc_loss(self, expected: np.ndarray) -> float:
        """

        :param expected: one dimensional vector of integers corresponding to the correct labels, starting at 0
        :return:
        """

        assert isinstance(self.predicted, np.ndarray)

        # each expected should be hot one vector so just pass list of int corresponding to the labels
        assert expected.ndim == 1
        assert np.issubdtype(expected.dtype, np.integer)
        assert np.max(expected) <= self.predicted.shape[-1] - 1
        self.expected = expected

        batch_size = self.predicted.shape[0]
        loss = -np.sum(self.predicted[np.arange(batch_size), expected])
        assert isinstance(loss, float)

        return loss

    def backward(self) -> np.ndarray:
        assert isinstance(self.expected, np.ndarray)
        assert isinstance(self.predicted, np.ndarray)

        batch_size = self.predicted.shape[0]
        gard = np.exp(self.predicted)
        gard[np.arange(batch_size), self.expected] -= 1
        # gard /= batch_size
        assert isinstance(gard, np.ndarray)

        return gard
