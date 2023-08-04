from tensorflow import keras
from keras import layers
import torch
import numpy as np

from Activations.relu import ReLU
from Layers.flatten import Flatten
from Layers.fully_connected import FullyConnected
from Losses.mean_squared_error import MeanSquaredError
from Losses.log_softmax_cross_entropy import LogSoftmaxCrossEntropy
from Models import Sequential


def split_given_size(arr, chunk_size, axis=0):
    return np.split(arr, np.arange(chunk_size, arr.shape[axis], chunk_size), axis=axis)


def keras_model(train, test, epoch, lr, batch_size):
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = keras.optimizers.SGD(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    x_train, y_train = train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, validation_split=0)

    x_test, y_test = test
    model.evaluate(x_test, y_test)

    print(model.predict(x_test[0].reshape(1, 28, 28)))


def numpy_model(train, test, epochs, lr, batch_size, ):
    layers = [
        Flatten(),
        FullyConnected(28 * 28, 512, ReLU()),
        FullyConnected(512, 256, ReLU()),
        FullyConnected(256, 64, ReLU()),
        FullyConnected(64, 10)
    ]
    loss_function = LogSoftmaxCrossEntropy()
    model = Sequential(layers, loss_function)

    x_train, y_train = train
    model.train(x_train, y_train, lr, batch_size, epochs)

    x_test, y_test = test
    predicted, loss = model.evaluate(x_test, y_test)
    accuracy = model._accuracy(predicted, y_test)
    print(f"loss on test is ={loss}, accuracy={accuracy}")

    print(np.exp(model.predict(x_test[0].reshape(1, 28, 28))))


def main():
    train, test = keras.datasets.mnist.load_data()

    lr = 0.001
    batch_size = 64
    epochs = 10

    keras_model(train, test, epochs, lr, batch_size)
    numpy_model(train, test, epochs, lr, batch_size)


if __name__ == '__main__':
    # np.seterr(all="raise")
    np.random.seed(10)
    main()
