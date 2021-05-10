import os
import torch
import torchvision
import numpy as np
from typing import Tuple, Dict, Union, Optional


def dataset_downloader(
    dataset_name: str = "CIFAR10",
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Download the corresponding dataset, skip if already downloaded.

    Args:
        dataset_name: Name of the dataset.\n
    Returns:
        Train and test dataset, both of type `torch.utils.data.Dataset`.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))

    if dataset_name == "CIFAR10":
        # check if already downloaded
        data_path = os.path.join(dir_path, "../../../", dataset_name)
        downloaded = os.path.exists(os.path.join(data_path, "cifar-10-python.tar.gz"))
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=not downloaded,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=not downloaded,
        )
    return train_dataset, test_dataset


def subset(
    dataset: torch.utils.data.Dataset,
    class_id: int = 0,
    num_samples: Optional[int] = None,
) -> torch.utils.data.Dataset:
    """
    Take a subset from the whole dataset.

    First, fetch all the data points of the given `class_id` as subset,
    then randomly select `num_samples` many points from this class.

    Args:
        dataset: The dataset, usually containing multiple classes.
        class_id: The id for the target class we want to filter.
        num_samples: Sampling size of the class `class_id`.

    Returns:
        A subset from `dataset` with samples all in class `class_id` and of 
        size `num_samples`. If `num_samples` is not specified, then keep all 
        the samples in this class, which is the usual practice for test set.
    """
    idx = torch.tensor(dataset.targets) == class_id
    subset = torch.utils.data.dataset.Subset(
        dataset=dataset, indices=np.where(idx == True)[0]
    )

    if num_samples:
        assert num_samples <= len(subset)
        idx = np.random.choice(len(subset), num_samples, replace=False)
        subset = torch.utils.data.dataset.Subset(dataset=subset, indices=idx)
    return subset


def new_dataset_from_size_dict(
    train_dataset: torch.utils.data.Dataset,
    test_dataset: torch.utils.data.Dataset,
    size_dict: Dict[int, int],
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
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
        type `torch.utils.data.Dataset`.
    """
    # prepare train set
    subsets = []
    for _, (class_id, size) in enumerate(size_dict.items()):
        subsets.append(subset(train_dataset, class_id, size))
    new_train_dataset = torch.utils.data.ConcatDataset(subsets)

    # prepare test set
    subsets = []
    for _, (class_id, _) in enumerate(size_dict.items()):
        subsets.append(subset(test_dataset, class_id))
    new_test_dataset = torch.utils.data.ConcatDataset(subsets)
    return new_train_dataset, new_test_dataset
