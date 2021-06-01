import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple


class Dataset:
    """Dataset base class."""

    DATA_ROOT = "./data"

    @classmethod
    def numpy(
        cls, model_type: str, one_hot_encode: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """Loads train and test dataset for given model type as a numpy arrays.

        :param one_hot_encode: If data should be one-hot-encoded.
        :param model_type: Type of the model for which the data is.
        :return: Train and Test data and labels as numpy arrays.
        """
        raise NotImplementedError(
            "Method 'numpy()' needs to be implemented in subclass."
        )

    @classmethod
    def pytorch_loader(
        cls,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        one_hot_encode: bool = False,
    ) -> Tuple[DataLoader, DataLoader]:
        """Loads the dataset as Pytorch train and test data loader.

        :param train_batch_size: Batch size of the train data loader.
        :param test_batch_size: Batch size of the test data loader.
        :param one_hot_encode: If data should be one-hot-encoded.
        :return: Train and test data loaders.
        """
        raise NotImplementedError(
            "Method 'pytorch_loader()' needs to be implemented in subclass."
        )

    @classmethod
    def tensorflow_loader(
        cls,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        one_hot_encode: bool = False,
    ):
        """Loads the dataset as Tensorflow train and test data loader.

        :param train_batch_size: Batch size of the train data loader.
        :param test_batch_size: Batch size of the test data loader.
        :param one_hot_encode: If data should be one-hot-encoded.
        :return: Train and test data loaders.
        """
        raise NotImplementedError(
            "Method 'tensorflow_loader()' needs to be implemented in subclass."
        )

    @classmethod
    def _one_hot_encode(cls, y: np.ndarray, n_classes: int) -> np.ndarray:
        """On-hot-encode labels.

        :param y: Labels to be one-hot-encoded.
        :param n_classes: Number of classes that exist.
        :return: One-hot-encoded label.
        """
        y_one_hot_encoded = np.zeros((y.shape[0], n_classes))
        y_one_hot_encoded[np.arange(y.shape[0]), y] = 1
        return y_one_hot_encoded
