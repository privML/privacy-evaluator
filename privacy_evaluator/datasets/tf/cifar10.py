import tensorflow as tf

from .tf import TFDataset


class TFCIFAR10(TFDataset):
    """`TFCIFAR10` class.

    Represents a CIFAR10 dataset class for TensorFlow.
    """

    TF_MODULE = tf.keras.datasets.cifar10
    DATASET_SIZE = {"train": 50000, "test": 10000}
    INPUT_SHAPE = (32, 32, 3)
    N_CLASSES = 10
