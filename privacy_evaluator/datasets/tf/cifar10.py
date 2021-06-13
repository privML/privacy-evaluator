import tensorflow as tf

from .tf import TFDataset


class TFCIFAR10(TFDataset):
    """TensorFlow CIFAR10 dataset class."""

    TF_MODULE = tf.keras.datasets.cifar10
    DATASET_SIZE = {"train": 50000, "test": 10000}
    INPUT_SHAPE = (3, 32, 32)
    N_CLASSES = 10
