from attrs import define, field
from typing import Callable

from Activations.activation import Activation
from .layer import Layer
import numpy as np


@define
class FullyConnected(Layer):
    input_size: int
    output_size: int
    activation: Activation | None = field(default=None)

    weights: np.ndarray = field(init=False)
    biases: np.ndarray = field(init=False)

    weights_gradient: np.ndarray = field(init=False)
    biases_gradient: np.ndarray = field(init=False)

    input: np.ndarray | None = field(init=False, default=None)

    def __attrs_post_init__(self) -> None:
        self.weights = np.random.randn(self.input_size, self.output_size) / np.sqrt(self.output_size)
        self.biases = np.zeros((1, self.output_size))

        self._reset_gradients()

    def _reset_gradients(self) -> None:
        self.weights_gradient = np.zeros_like(self.weights)
        self.biases_gradient = np.zeros_like(self.biases)

    def forward(self, input: np.ndarray, *, training: bool) -> np.ndarray:
        if training:
            self.input = input
        else:
            self.input = None

        out = np.dot(input, self.weights) + self.biases

        if self.activation is None:
            return out
        else:
            return self.activation.forward(out)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.input is not None

        if self.activation is not None:
            output_gradient = self.activation.backward(output_gradient)

        batch_size = output_gradient.shape[0]
        self.weights_gradient += np.dot(self.input.T, output_gradient) / batch_size
        self.biases_gradient += np.sum(output_gradient, axis=0, keepdims=True) / batch_size
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.input = None
        return input_gradient

    def update(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self.weights_gradient
        self.biases -= learning_rate * self.biases_gradient

        self._reset_gradients()

    def save(self, push: Callable[[np.ndarray], None]) -> None:
        push(self.weights)
        push(self.biases)

    def load(self, pop: Callable[[], np.ndarray]) -> None:
        self.biases = pop()
        self.weights = pop()
