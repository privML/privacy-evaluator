import tensorflow as tf

from .tf import TFDataset


class TFMNIST(TFDataset):
    """`TFMNIST` class.

    Represents a MNIST dataset class for TensorFlow.
    """

    TF_MODULE = tf.keras.datasets.mnist
    DATASET_SIZE = {"train": 60000, "test": 10000}
    INPUT_SHAPE = (28, 28)
    N_CLASSES = 10
