import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
from typing import Tuple

from privacy_evaluator.datasets.dataset import Dataset


class CIFAR10(Dataset):
    """CIFAR10 dataset class."""

    TRAIN_SET_SIZE = 50000
    TEST_SET_SIZE = 10000
    INPUT_SHAPE = (3, 32, 32)
    N_CLASSES = 10

    @classmethod
    def pytorch_loader(
        cls,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        one_hot_encode: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """Loads the dataset as pytorch train and test data loader.

        :param train_batch_size: Batch size of the train data loader.
        :param test_batch_size: Batch size of the test data loader.
        :param one_hot_encode: If data should be one-hot-encoded.
        :return: Train and test data loaders.
        """
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                # The first tuple contains the mean for every RGB channel
                # computed over the training set.
                # The second tuple contains the standard deviation for every
                # RGB channel computed over the training set.
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_set = datasets.CIFAR10(
            root=cls.DATA_ROOT, train=True, download=True, transform=transform_train
        )

        test_set = datasets.CIFAR10(
            root=cls.DATA_ROOT, train=False, download=True, transform=transform_test
        )

        if one_hot_encode:
            train_set.targets = cls._one_hot_encode(
                np.array(train_set.targets), cls.N_CLASSES
            )
            test_set.targets = cls._one_hot_encode(
                np.array(test_set.targets), cls.N_CLASSES
            )

        return (
            DataLoader(
                train_set, batch_size=train_batch_size, shuffle=True, num_workers=4
            ),
            DataLoader(
                test_set, batch_size=test_batch_size, shuffle=False, num_workers=4
            ),
        )

    @classmethod
    def numpy(cls, one_hot_encode: bool = True) -> Tuple[np.ndarray, ...]:
        """Loads train and test dataset as a numpy arrays.

        :param one_hot_encode: If data should be one-hot-encoded.
        :return: Train and Test data and labels as numpy arrays.
        """
        train_loader, test_loader = cls.pytorch_loader(
            train_batch_size=cls.TRAIN_SET_SIZE,
            test_batch_size=cls.TEST_SET_SIZE,
            one_hot_encode=one_hot_encode,
        )

        x_train, y_train = next(iter(train_loader))
        x_train, y_train = x_train.numpy(), y_train.numpy()
        x_test, y_test = next(iter(test_loader))
        x_test, y_test = x_test.numpy(), y_test.numpy()

        return x_train, y_train, x_test, y_test

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
            "Method 'tensorflow_loader()' needs to be implemented."
        )
