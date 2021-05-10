import numpy as np
from tensorflow.keras.datasets import cifar10
from typing import Tuple, Optional, Dict


def dataset_downloader(dataset_name: str = "CIFAR10") -> Tuple[np.ndarray, np.ndarray]:
    """
    Download the corresponding dataset.

    Args:
        dataset_name: Name of the dataset.\n
    Returns:
        Train and test dataset, both of type `np.ndarray`.
    """
    if dataset_name == "CIFAR10":
        train_dataset, test_dataset = cifar10.load_data()
    return train_dataset, test_dataset


def subset(
    dataset: Tuple[np.ndarray, np.ndarray],
    class_id: int = 0,
    num_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take a subset from the whole dataset.

    First, fetch all the data points of the given `class_id` as subset,
    then randomly select `num_samples` many points from this class.

    Args:
        dataset: The dataset, usually containing multiple classes.
        class_id: The id for the target class we want to filter.
        num_samples: Size of the result dataset.

    Returns:
        A subset from `dataset` with samples all in class `class_id` and
        of size `num_samples`.
    """
    data_x, data_y = dataset[0], dataset[1]
    idx = (data_y == class_id).reshape(data_x.shape[0])
    subset_x, subset_y = data_x[idx], data_y[idx]

    if num_samples:
        assert num_samples <= len(subset_y)
        idx = np.random.choice(len(subset_y), num_samples, replace=False)
        subset_x, subset_y = subset_x[idx], subset_y[idx]
    return subset_x, subset_y


def new_dataset_from_size_dict(
    train_dataset: np.ndarray,
    test_dataset: np.ndarray,
    size_dict: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build new dataset from original, using the `size_dict` to set the size of
    each wanted class.

    To create new (unbalanced) training set, we subset the original training
    set according to  `size_dict` which gives an unbalanced distribution. The
    new training set then only covers the classes appeared in `size_dict` and
    its size shrinks to the sum of values in `size_dict`.\n
    To create new test set, we only filter the classes mentioned and keep the
    original size. Therefore, if a class is mentioned, all its test samples
    will stay.

    Args:
        train_dataset: The original train set.
        test_dataset: The original test set.
        size_dict: A list `(class_1: size_1, class_2: size_2, ..., class_n:
        size_n)` where `class_i` denotes the class id and `size_i` denotes
        the sample size of `class_i`.

    Returns:
        The new training and test set consisting of mentioned classes, both of
        type `np.ndarray`.
    """
    # prepare train set according to given (imbalanced) sizes
    new_train_x, new_train_y = [], []
    for _, (class_id, size) in enumerate(size_dict.items()):
        new_train_x.append(subset(train_dataset, class_id, size)[0])
        new_train_y.append(subset(train_dataset, class_id, size)[1])
    new_train_x = np.vstack(new_train_x)
    new_train_y = np.vstack(new_train_y).flatten()
    new_train_set = (new_train_x, new_train_y)

    # prepare test set of the original (balanced) sizes
    new_test_x, new_test_y = [], []
    for _, (class_id, _) in enumerate(size_dict.items()):
        new_test_x.append(subset(test_dataset, class_id)[0])
        new_test_y.append(subset(test_dataset, class_id)[1])
    new_test_x = np.vstack(new_test_x)
    new_test_y = np.vstack(new_test_y).flatten()
    new_test_set = (new_test_x, new_test_y)
    return new_train_set, new_test_set
