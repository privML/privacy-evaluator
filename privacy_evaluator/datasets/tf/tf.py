import numpy as np
import tensorflow as tf
from typing import Tuple


class TFDataset:
    """TensorFlow Dataset base class."""

    @classmethod
    def numpy(
        cls,
        one_hot_encode: bool = True,
        normalize: bool = True,
        take: int = 100,
    ) -> Tuple[np.ndarray, ...]:
        """Loads train and test dataset for given model type as a numpy arrays.

        :param one_hot_encode: If data should be one-hot-encoded.
        :param normalize: If data should be normalized.
        :param take: Percentage of the data set to use.
        :return: Train and Test data and labels as numpy arrays.
        """
        (x_train, y_train), (x_test, y_test) = cls.TF_MODULE.load_data()

        n_train = round(cls.DATASET_SIZE["train"] / 100 * take)
        n_test = round(cls.DATASET_SIZE["test"] / 100 * take)

        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
        x_test = x_test[:n_test]
        y_test = y_test[:n_test]

        if normalize:
            x_train = cls.normalize(x_train)
            x_test = cls.normalize(x_test)

        if one_hot_encode:
            y_train = cls.one_hot_encode(y_train)
            y_test = cls.one_hot_encode(y_test)

        cls.validate(x_train, y_train, n_train, one_hot_encoded=one_hot_encode)
        cls.validate(x_test, y_test, n_test, one_hot_encoded=one_hot_encode)

        return x_train, y_train, x_test, y_test

    @classmethod
    def one_hot_encode(cls, y: np.ndarray) -> np.ndarray:
        """On-hot-encode labels.

        :param y: Labels to be one-hot-encoded.
        :return: One-hot-encoded labels.
        """
        return (
            tf.one_hot(y, depth=cls.N_CLASSES)
            .numpy()
            .reshape(y.shape[0], cls.N_CLASSES)
        )

    @classmethod
    def validate(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        n: int,
        one_hot_encoded: bool = True,
    ):
        """Validates the data.

        :param x: Data to be validated.
        :param y: Labels for `x` to be validated.
        :param n: Expected number of data points.
        :param one_hot_encoded: If data is one-hot-encoded or not.
        """
        assert x.shape == (n, *cls.INPUT_SHAPE)
        if one_hot_encoded:
            assert y.shape == (n, cls.N_CLASSES)
        else:
            assert y.shape == (n)

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        """Normalize the data.

        :param x: Data to be normalized.
        :return: Normalized dataset.
        """
        return tf.image.per_image_standardization(x / 255)
