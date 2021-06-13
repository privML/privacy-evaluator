import tensorflow as tf

from .tf import TFDataset


class TFMNIST(TFDataset):
    """TensorFlow MNIST dataset class."""

    TF_MODULE = tf.keras.datasets.mnist
    DATASET_SIZE = {"train": 60000, "test": 10000}
    INPUT_SHAPE = (28, 28)
    N_CLASSES = 10
