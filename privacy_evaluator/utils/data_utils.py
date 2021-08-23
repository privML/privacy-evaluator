import numpy as np
from tensorflow.keras.datasets import cifar10, mnist
from typing import Tuple, Optional, Dict
from .data_adaptation import images_adaptation


def dataset_downloader(
    dataset_name: str = "CIFAR10",
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Download the corresponding dataset.
    Args:
        dataset_name: Name of the dataset.
    Returns:
        Train and test dataset, both of type `np.ndarray`.
    """
    if dataset_name == "CIFAR10":
        train_dataset, test_dataset = cifar10.load_data()
    elif dataset_name == "MNIST":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # convert image shape from (28, 28) to (28, 28, 1) if color channel is missing
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        train_dataset = (X_train, y_train)
        test_dataset = (X_test, y_test)
    else:
        raise ValueError("This dataset not supported!")
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
        dataset: The training and test dataset in a tuple.
        class_id: The id for the target class we want to filter.
        num_samples: Sampling size of the specified class.
    Returns:
        A subset from `dataset` with samples all in class `class_id` and
        of size `num_samples`. If `num_samples` is not specified, keep all
        samples in this class, which is the usual practice for test set.
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
    dataset: Tuple[np.ndarray, np.ndarray], size_dict: Dict[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build new dataset from original, using the `size_dict` to set the size of
    each wanted class.
    To create new (unbalanced) training set, we subset the original training
    set according to `size_dict` which gives an unbalanced distribution. The
    new training set then only covers the classes appeared in `size_dict` and
    its size shrinks to the sum of values in `size_dict`.\n
    To create new test set, we only filter the classes mentioned and keep the
    original size. Therefore, if a class is mentioned, all its test samples
    will stay.
    Args:
        train_dataset: The original train set.
        test_dataset: The original test set.
        size_dict: A dictionary `{class_1: size_1, class_2: size_2, ..., class_n:
        size_n}` where `class_i` denotes the class id and `size_i` denotes
        the sample size of `class_i`. `n` is the number of classes involved.
    Returns:
        The new training and test set consisting of mentioned classes, both of
        type `np.ndarray`.
    """
    # prepare train set according to given (imbalanced) sizes
    new_data_x, new_data_y = [], []
    for _, (class_id, size) in enumerate(size_dict.items()):
        new_data_x.append(subset(dataset, class_id, size)[0])
        new_data_y.append(subset(dataset, class_id, size)[1])
    new_data_x = np.vstack(new_data_x)
    new_data_y = np.concatenate(new_data_y, axis=0)
    new_dataset = (new_data_x, new_data_y)

    return new_dataset


def split_data_set_with_ratio(
    data_set: Tuple[np.ndarray, np.ndarray], ratio: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Splits a data set in two datsets according to given ratio.
    :param data_set: Input data set.
    :param ratio: A ratio how to split the data set. Describes size of new_data_set1.
    :return: split datasets
    """
    assert ratio <= 1 and ratio >= 0
    num_samples = len(data_set[0])

    idx = np.random.choice(num_samples, int(num_samples * ratio), replace=False)
    new_data_set1 = (data_set[0][idx], data_set[1][idx])

    mask = np.ones(num_samples, np.bool)
    mask[idx] = 0
    new_data_set2 = (data_set[0][mask], data_set[1][mask])

    return new_data_set1, new_data_set2


def create_new_dataset_with_adaptation(
    data_set: Tuple[np.ndarray, np.ndarray], ratio: float, adaptation: str, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a new data set with adapted samples according to given ratio.
    :param data_set: Input data set.
    :param ratio: A ratio for the number of adapted samples.
    :param adaptation: The type of adaptation. ('mask', 'random_noise', 'brightness')
    :params **kwargs: Optional parameters for the specified adaptation.

    Optional params:
    :params box_len: Involved when `adaptation` is "mask", the side length of masking boxes.
    :params brightness: Involved when `adaptation` is "brightness", the amount the brightness should be raised or lowered
    :params mean: Involved when `adaptation` is "random_noise", the mean of the added noise.
    :params mean: Involved when `adaptation` is "random_noise", the standard deviation of the added noise.

    :return: data set with adapted samples according to given ratio
    """
    # size of data1 is according to ratio, size of data2 is according to (1-ratio)
    data1, data2 = split_data_set_with_ratio(data_set, ratio)
    adapted_dataset = (images_adaptation(data1[0], adaptation, **kwargs), data1[1])
    new_data_set = (
        np.concatenate((adapted_dataset[0], data2[0])),
        np.concatenate((adapted_dataset[1], data2[1])),
    )
    return new_data_set
