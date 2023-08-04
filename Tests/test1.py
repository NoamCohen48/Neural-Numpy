from tensorflow import keras
from keras import layers
import torch
import numpy as np

from Activations.relu import ReLU
from Layers.fully_connected import FullyConnected
from Losses.mean_squared_error import MeanSquaredError
from Models import Sequential
from utils import int_to_one_hot


def function(x, y, z):
    # return (np.square(x - y) - np.abs(z))
    return (np.square(x - y) * np.sqrt(np.abs(z)) + x + y)


def split_given_size(arr, chunk_size, axis=0):
    return np.split(arr, np.arange(chunk_size, arr.shape[axis], chunk_size), axis=axis)


def data_loader(N):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # x = np.random.uniform(-1, 1, (N, 1))
    # y = np.random.uniform(-1, 1, (N, 1))
    # z = np.random.uniform(-1, 1, (N, 1))
    # x_train = np.hstack([x, y, z])
    # y_train = function(x, y, z)

    return x_train.reshape(-1, 28 * 28), int_to_one_hot(y_train)


def keras_model(x_train, y_train, epoch, lr, batch_size, x_test):
    model = keras.Sequential(
        [
            keras.Input(shape=28*28),
            layers.Dense(6, activation="relu"),
            layers.Dense(6, activation="relu"),
            layers.Dense(6, activation="relu"),
            layers.Dense(10),
        ]
    )
    loss = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer)

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0)

    if x_test is None:
        return
    return model.predict(x_test)


def numpy_model(x_train, y_train, epochs, lr, batch_size, x_test):
    layers = [
        FullyConnected(28*28, 6, ReLU()),
        FullyConnected(6, 6, ReLU()),
        FullyConnected(6, 6, ReLU()),
        FullyConnected(6, 10)
    ]
    loss_function = MeanSquaredError()
    model = Sequential(layers, loss_function)
    model.train(x_train, y_train, lr, batch_size, epochs)
    if x_test is None:
        return
    return model.predict(x_test)


def main():
    N = 1_000_000
    x_train, y_train = data_loader(N)
    x_test = np.array([
        [0.5, 0.3, -0.15],
        [0.7, -0.2, 0.6],
    ])
    x_test = None

    lr = 0.01
    batch_size = 64
    epochs = 10

    keras_predictions = keras_model(x_train, y_train, epochs, lr, batch_size, x_test)
    numpy_predictions = numpy_model(x_train, y_train, epochs, lr, batch_size, x_test)

    if keras_predictions is None or numpy_predictions is None:
        return

    for x, keras, numpy in zip(x_test, keras_predictions, numpy_predictions):
        print(f"x={x}, keras={keras}, numpy={numpy}, expected={function(*x)}")
    return

    # y_true = np.array([1.0, 2.0, 1.0])
    # y_pred = np.array([-0.15, 0.5, 0.])
    # y_pred_softmax = keras.layers.Softmax()(y_pred)
    # print(y_pred_softmax)
    # cce1 = keras.losses.CategoricalCrossentropy(from_logits=False)
    # print(cce1(y_true, y_pred))
    # cce2 = keras.losses.CategoricalCrossentropy(from_logits=True)
    # print(cce2(y_true, y_pred))

    input = torch.randn(3, requires_grad=True)
    m = torch.nn.Softmax(dim=0)
    output = m(input)
    s = torch.sum(output)
    s.backward()
    g = input.grad
    print()


if __name__ == '__main__':
    main()
