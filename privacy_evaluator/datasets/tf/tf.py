from typing import Tuple
import tensorflow as tf
import numpy as np


class TFDataset:
    """TensorFlow Dataset base class."""

    @classmethod
    def numpy(
        cls, one_hot_encode: bool = True, normalize: bool = True
    ) -> Tuple[np.ndarray, ...]:
        """Loads train and test dataset for given model type as a numpy arrays.

        :param one_hot_encode: If data should be one-hot-encoded.
        :param normalize: If data should be normalized.
        :return: Train and Test data and labels as numpy arrays.
        """
        (x_train, y_train), (x_test, y_test) = cls.TF_MODULE.load_data()

        if normalize:
            x_train = cls.normalize(x_train)
            x_test = cls.normalize(x_test)

        if one_hot_encode:
            y_train = cls.one_hot_encode(y_train)
            y_test = cls.one_hot_encode(y_test)

        cls.validate(x_train, y_train, one_hot_encoded=one_hot_encode)
        cls.validate(x_test, y_test, dataset="test", one_hot_encoded=one_hot_encode)

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
        dataset: str = "train",
        one_hot_encoded: bool = True,
    ):
        """Validates the data.

        :param x: Data to be validated.
        :param y: Labels for `x` to be validated.
        :param one_hot_encoded: If data is one-hot-encoded or not.
        :param dataset: Dataset to be validated; either `train` or `test`.
        """
        assert x.shape == (cls.DATASET_SIZE[dataset], *cls.INPUT_SHAPE)
        if one_hot_encoded:
            assert y.shape == (cls.DATASET_SIZE[dataset], cls.N_CLASSES)
        else:
            assert y.shape == (cls.DATASET_SIZE[dataset])

    @staticmethod
    def normalize(x: np.ndarray) -> np.ndarray:
        """Normalize the data.

        :param x: Data to be normalized.
        :return: Normalized dataset.
        """
        return tf.image.per_image_standardization(x / 255)
