import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, Flatten


class ResNet50(keras.Model):
    """An adapted residual network for multi-classification.

    The backbone is the pretrained (on `imagenet`) `ResNet50` model. We freeze all
    the convolutional layers and only change the last fully connected layer to do
    classification.

    Args:
        num_classes: The number of classes involved in the classification.
        dropout: Drop-out rate in the fully-connected layer.
    """

    def __init__(self, num_classes=2, dropout=0):
        super(ResNet50, self).__init__()

        self.resnet = tf.keras.applications.ResNet50(
            include_top=False, input_shape=(32, 32, 3)
        )
        for layer in self.resnet.layers:
            layer.trainable = False

        self.avgpool = layers.GlobalAveragePooling2D()
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
        out = self.resnet(x)
        out = self.avgpool(out)
        out = self.fc(out)
        return out


class FCNeuralNet(keras.Model):
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

    def forward(self, x):
        x = self.flatten(x)
        out = self.fc(x)
        return out
