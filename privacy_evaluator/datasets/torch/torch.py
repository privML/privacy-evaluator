import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple


class TorchDataset:
    """PyTorch Dataset base class."""

    DATA_ROOT = "./data"

    @classmethod
    def numpy(
        cls, one_hot_encode: bool = True, transformers: str = "default"
    ) -> Tuple[np.ndarray, ...]:
        """Loads train and test dataset for given model type as a numpy arrays.

        :param one_hot_encode: If data should be one-hot-encoded.
        :param transformers: Transformers for the dataset; either `default` or `training`
        :return: Train and Test data and labels as numpy arrays.
        """
        train_loader, test_loader = cls.data_loader(
            train_batch_size=cls.DATASET_SIZE["train"],
            test_batch_size=cls.DATASET_SIZE["test"],
            one_hot_encode=one_hot_encode,
            transformers=transformers,
            shuffle=False,
        )

        x_train, y_train = next(iter(train_loader))
        x_train, y_train = x_train.numpy(), y_train.numpy()
        x_test, y_test = next(iter(test_loader))
        x_test, y_test = x_test.numpy(), y_test.numpy()

        cls.validate(x_train, y_train)
        cls.validate(x_test, y_test, dataset="test")

        return x_train, y_train, x_test, y_test

    @classmethod
    def data_loader(
        cls,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        one_hot_encode: bool = True,
        transformers: str = "default",
        shuffle: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Loads the dataset as train and test DataLoader.

        :param train_batch_size: Batch size of the train data loader.
        :param test_batch_size: Batch size of the test data loader.
        :param one_hot_encode: If data should be one-hot-encoded.
        :param transformers: Transformers for the dataset; either `default` or `training`
        :param shuffle: If data should be shuffled.
        :return: Train and test data loaders.
        """
        train_set = cls.TORCH_MODULE(
            root=cls.DATA_ROOT,
            train=True,
            download=True,
            transform=cls.TRANSFORMERS[transformers]["train"],
        )

        test_set = cls.TORCH_MODULE(
            root=cls.DATA_ROOT,
            train=False,
            download=True,
            transform=cls.TRANSFORMERS[transformers]["test"],
        )

        if one_hot_encode:
            train_set.targets = cls.one_hot_encode(np.array(train_set.targets))
            test_set.targets = cls.one_hot_encode(np.array(test_set.targets))

        return (
            DataLoader(
                train_set, batch_size=train_batch_size, shuffle=shuffle, num_workers=4
            ),
            DataLoader(
                test_set, batch_size=test_batch_size, shuffle=shuffle, num_workers=4
            ),
        )

    @classmethod
    def one_hot_encode(cls, y: np.ndarray) -> np.ndarray:
        """On-hot-encode labels.

        :param y: Labels to be one-hot-encoded.
        :return: One-hot-encoded labels.
        """
        y_one_hot_encoded = np.zeros((y.shape[0], cls.N_CLASSES))
        y_one_hot_encoded[np.arange(y.shape[0]), y] = 1
        return y_one_hot_encoded

    @classmethod
    def validate(cls, x: np.ndarray, y: np.ndarray, dataset: str = "train", one_hot_encoded: bool = True):
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
