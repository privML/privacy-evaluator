import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten


class FCNeuralNet(tf.Model):
    """A simple fully-connected network for multi-classification.

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate in the fully-connected layer.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0):
        super(FCNeuralNet, self).__init__()
        self.flatten = Flatten()
        self.fc = keras.Sequential(
            [
                Dense(512, activation="relu"),
                Dropout(dropout),
                Dense(64, activation="relu"),
                Dropout(dropout),
                Dense(num_classes),
            ]
        )

    def call(self, x):
        out = self.flatten(x)
        out = self.fc(out)
        return out
