import numpy as np
import torch
from attrs import define, field
from torch import nn

from Layers import Layer, FullyConnected
from Activations import Activation
from Losses import Loss


@define
class Sequential:
    layers: list[Layer]
    loss: Loss

    def _forward(self, x: np.ndarray, *, training: bool) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        x = self.loss.forward(x, training=training)
        return x

    def _backward(self) -> None:
        grad = self.loss.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float, batch_size: int, epochs: int) -> None:
        for epoch in range(epochs):

            indices = np.random.permutation(x.shape[0])
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            loss = 0.
            for batch_start in range(0, x.shape[0], batch_size):
                # Divide the data into mini-batches
                batch_end = batch_start + batch_size
                x_batch = x_shuffled[batch_start:batch_end]
                y_batch = y_shuffled[batch_start:batch_end]

                predicted = self._forward(x_batch, training=True)

                loss += self.loss(y_batch)

                self._backward()

                self._update(learning_rate)

            print(f"epoch average loss = {loss / x.shape[0]}")

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        predicted = self._forward(x, training=False)
        loss = self.loss(y) / x.shape[0]
        return predicted, loss

    def _accuracy(self, predicted: np.ndarray, y: np.ndarray) -> float:
        predicted_lebales = np.argmax(predicted, axis=1)
        same = predicted_lebales == y
        accuracy = np.mean(same)
        assert isinstance(accuracy, float)
        return accuracy

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x, training=False)
