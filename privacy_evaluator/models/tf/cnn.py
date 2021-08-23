import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Flatten,
)
from typing import Tuple


def ConvNet(
    num_classes: int = 2,
    input_shape: Tuple[int, ...] = (1, 28, 28),
    num_channels: Tuple[int, ...] = (1, 16, 32, 64),
) -> keras.Model:
    """
    Provide a convolutional neural network for image classification.

    Note: This method is just aimed at fetching a model for developers' test when
    a target model is required. Since only `MNIST` and `CIFAR10` datasets are
    our concern, this method is compatible only with these two corresponding
    image sizes (28*28 and 32*32*3)

    Args:
        num_classes: number of classes during prediction, serving as the size of
        the last fully-connected layer.
        input_shape: either (28, 28) or (32, 32, 3) or their variations (because
        of position for channel-dimension).
        num_channels: Number of input channels.
    Returns:
        An MNIST-classifier if `input_shape` corresponds to 28*28 or a CIFAR10-classifier
        if corresponds to 32*32*3. Otherwise raise an Error.
    """

    if input_shape in [(1, 28, 28), (28, 28, 1)]:
        return ConvNetMNIST(num_classes, num_channels)

    elif input_shape in [(3, 32, 32), (32, 32, 3)]:
        num_channels = (3, 64, 128)
        return ConvNetCIFAR10(num_classes, num_channels)

    else:
        raise ValueError(
            "The input_shape must be one of the followings: "
            "(1, 28, 28), (28, 28, 1), (3, 32, 32), (32, 32, 3)"
        )


class ConvNetMNIST(keras.Model):
    """
    Provide a convolutional neural network for MNIST classification.

    Note: This method is just aimed at fetching a model for developers' test when
    a target model is required. Since only the `MNIST` dataset is our concern, this 
    method is compatible only with the corresponding image size (28*28).

    Args:
        num_classes: Number of classes during prediction, serving as the size of
        the last fully-connected layer.
        input_shape: Either (28, 28) or its variations (because of possible positions \
            for the channel-dimension).
        num_channels: Number of channels.
    """

    def __init__(
        self, num_classes: int = 2, num_channels: Tuple[int, ...] = (1, 16, 32, 64)
    ):
        super(ConvNetMNIST, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels

        # define the architecture
        self.conv1 = keras.Sequential(
            [
                Conv2D(
                    filters=num_channels[1],
                    kernel_size=3,
                    activation="relu",
                    input_shape=(28, 28, 1),
                ),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.conv2 = keras.Sequential(
            [
                Conv2D(filters=num_channels[2], kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.conv3 = keras.Sequential(
            [
                Conv2D(filters=num_channels[3], kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.conv3 = keras.Sequential(
            [
                Conv2D(
                    filters=num_channels[3],
                    kernel_size=3,
                    activation="relu",
                ),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.fc = keras.Sequential(
            [
                Flatten(),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )

    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)
        return out


class ConvNetCIFAR10(keras.Model):
    """
    Provide a convolutional neural network for CIFAR10 classification.

    Note: This method is just aimed at fetching a model for developers' test when
    a target model is required. Since only the `CIFAR10` dataset is our concern, this 
    method is compatible only with the corresponding image size (32*32*3).

    Args:
        num_classes: Number of classes during prediction, serving as the size of
        the last fully-connected layer.
        input_shape: Either (32, 32, 3) or its variations (because of possible positions \
            for the channel-dimension).
        num_channels: Number of input channels.
    """

    def __init__(
        self, num_classes: int = 2, num_channels: Tuple[int, ...] = (3, 16, 32, 64)
    ):
        super(ConvNetCIFAR10, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels

        # define the architecture
        self.conv1 = keras.Sequential(
            [
                Conv2D(
                    filters=num_channels[1],
                    kernel_size=3,
                    activation="relu",
                    input_shape=(32, 32, 3),
                ),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.conv2 = keras.Sequential(
            [
                Conv2D(filters=num_channels[2], kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.conv3 = keras.Sequential(
            [
                Conv2D(filters=num_channels[3], kernel_size=3, activation="relu"),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.conv3 = keras.Sequential(
            [
                Conv2D(
                    filters=num_channels[3],
                    kernel_size=3,
                    activation="relu",
                ),
                BatchNormalization(),
                MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.fc = keras.Sequential(
            [
                Flatten(),
                Dense(128, activation="relu"),
                Dense(num_classes, activation="softmax"),
            ]
        )

    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out)
        return out
