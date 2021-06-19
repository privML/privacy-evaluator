import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten


class FCNeuralNet(keras.Model):
    """Provide a simple fully-connected network for multi-classification in TF.

    Note: This method is just aimed at fetching a model for developers' test when
    a target model is required. Since only `MNIST` and `CIFAR10` datasets are
    our concern, this method is compatible only with these two corresponding
    image sizes (28*28 and 32*32*3)

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate between the fully-connected layers.
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
