import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPool2D,
    BatchNormalization,
    Flatten,
)
from typing import Tuple


def ConvNet(
    num_classes: int = 2, input_shape: Tuple[int, ...] = (1, 28, 28)
) -> keras.Model:
    if input_shape in [(1, 28, 28), (28, 28, 1)]:
        num_channels = (1, 64, 128)
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
    def __init__(
        self, num_classes: int = 2, num_channels: Tuple[int, ...] = (1, 64, 128)
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
                Conv2D(
                    filters=num_channels[2],
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
        out = self.fc(out)
        return out


class ConvNetCIFAR10(keras.Model):
    def __init__(
        self, num_classes: int = 2, num_channels: Tuple[int, ...] = (3, 64, 128)
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
                Conv2D(
                    filters=num_channels[2],
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
        out = self.fc(out)
        return out
