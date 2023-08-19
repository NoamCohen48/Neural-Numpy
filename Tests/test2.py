import numpy as np
from keras import layers
from tensorflow import keras

from Activations import Relu
from Layers import Flatten
from Layers import FullyConnected
from Losses import LogSoftmaxCrossEntropy
from Metrics import Accuracy
from Models import Sequential
from PreprocessingLayers import Normalization


def split_given_size(arr, chunk_size, axis=0):
    return np.split(arr, np.arange(chunk_size, arr.shape[axis], chunk_size), axis=axis)


def keras_model(train, test, epoch, lr, batch_size):
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Normalization(),
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


def numpy_model(train, test, epochs, lr, batch_size):
    x_train, y_train = train

    model = (
        Sequential()
        .add_preprocessing_layer(Normalization())

        .add_layer(Flatten())

        .add_layer(FullyConnected(512))
        .add_layer(Relu())

        .add_layer(FullyConnected(256))
        .add_layer(Relu())

        .add_layer(FullyConnected(64))
        .add_layer(Relu())

        .add_layer(FullyConnected(10))
        .set_loss(LogSoftmaxCrossEntropy())
        
        .build(x_train, y_train)
        .add_metric(Accuracy())
    )
    model.train(x_train, y_train, lr, batch_size, epochs)

    x_test, y_test = test
    predicted, loss = model.evaluate(x_test, y_test)
    print(f"loss on test is ={loss}")
    metrics = model.run_metrics(predicted, y_test)
    for metric, value in metrics.items():
        print(f"{metric}={value}", end=", ")

    print(np.exp(model.predict(x_test[0].reshape(1, 28, 28))))


def main():
    train, test = keras.datasets.mnist.load_data()

    lr = 0.001
    batch_size = 64
    epochs = 10

    # keras_model(train, test, epochs, lr, batch_size)
    numpy_model(train, test, epochs, lr, batch_size)


if __name__ == '__main__':
    # np.seterr(all="raise")
    np.random.seed(10)
    main()
