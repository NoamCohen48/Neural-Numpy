import numpy as np
from attrs import define, field

from Metrics import Metric
from PreprocessingLayers import PreprocessingLayer
from Layers import Layer
from Activations import Activation
from Losses import Loss

from typing import Self


@define
class Sequential:
    preprocessing_layers: list[PreprocessingLayer] = field(init=False, factory=list)
    layers: list[Layer] = field(init=False, factory=list)
    loss_layer: Loss | None = field(init=False, default=None)
    metrics: list[Metric] = field(init=False, factory=list)

    def add_preprocessing_layer(self, preprocessing_layer: PreprocessingLayer) -> Self:
        self.preprocessing_layers.append(preprocessing_layer)
        return self

    def add_layer(self, layer: Layer) -> Self:
        self.layers.append(layer)
        return self

    def set_loss(self, loss_layer: Loss | None) -> Self:
        self.loss_layer = loss_layer
        return self

    def add_metric(self, metric: Metric) -> Self:
        self.metrics.append(metric)
        return self

    def _forward(self, x: np.ndarray, *, training: bool) -> np.ndarray:
        for preprocessing_layer in self.preprocessing_layers:
            x = preprocessing_layer.forward(x)

        for layer in self.layers:
            x = layer.forward(x, training=training)

        if self.loss_layer is not None:
            x = self.loss_layer.forward(x, training=training)

        return x

    def _backward(self, *, output_gradient: np.ndarray | None = None) -> np.ndarray:
        if self.loss_layer is None:
            assert output_gradient is not None
            grad = output_gradient
        else:
            grad = self.loss_layer.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad

    def _update(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.update(learning_rate)

    def build(self, x: np.ndarray, y: np.ndarray | None = None) -> Self:
        for preprocessing_layer in self.preprocessing_layers:
            preprocessing_layer.build(x, y)

        input_shape = x.shape[1:]
        for layer in self.layers:
            input_shape = layer.build(input_shape)
        return self

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

                if self.loss_layer is not None:
                    loss += self.loss_layer(y_batch)

                self._backward()

                self._update(learning_rate)

            print(f"epoch average loss = {loss / x.shape[0]}")

    def run_metrics(self, model_outputs: np.ndarray, true: np.ndarray) -> dict[str, float]:
        # results: dict[str, float] = dict()
        # for metric in self.metrics:
        #     result = metric(model_outputs, true)
        #     results[metric.name] = result
        results: dict[str, float] = {metric.name: metric(model_outputs, true) for metric in self.metrics}

        return results

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x, training=False)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
        assert isinstance(self.loss_layer, Loss)
        predicted = self.predict(x)
        loss = self.loss_layer(y) / x.shape[0]
        return predicted, loss
