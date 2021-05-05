import os
import torch
from torchvision import datasets, transforms
import numpy as np


def dataset_downloader(dataset_name='CIFAR10'):
    """
    Download the corresponding dataset, skip if already downloaded.

    :param dataset_name: Name of the dataset.\n
    :type dataset_name: str\n
    
    :return: Train&Test dataset, both of type `torch.utils.data.Dataset`.
    :rtype: tuple\n
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))

    if dataset_name == 'CIFAR10':
        # check if already downloaded        
        data_path = os.path.join(dir_path, dataset_name)
        downloaded = os.path.exists(
            os.path.join(data_path, 'cifar-10-python.tar.gz'))
        train_dataset = datasets.CIFAR10(root=data_path, 
                                         train=True, 
                                         transform=transforms.ToTensor(), 
                                         download=not downloaded)
        test_dataset = datasets.CIFAR10(root=data_path, 
                                        train=False, 
                                        transform=transforms.ToTensor(), 
                                        download=not downloaded)
    return train_dataset, test_dataset


def subset(dataset, class_id, num_samples=None):
    """
    Take a subset from the whole dataset.

    First, fetch all the data points of the given `class_id` as subset,
    then randomly select `num_samples` many points from this class.

    :param dataset: The dataset, usually containing multiple classes.\n
    :type dataset: torch.utils.data.Dataset\n
    :param class_id: The id for the target class we want to filter.\n
    :type class_id: int\n
    :param num_samples: Size of the result dataset.\n
    :type num_samples: int, Nonetype

    :return: A subset from `dataset` with samples all in class \
        `class_id` and of size `num_samples`.\n
    :rtype: torch.utils.data.Dataset
    """
    idx = (torch.tensor(dataset.targets) == class_id)
    subset = torch.utils.data.dataset.Subset(
        dataset=dataset, 
        indices=np.where(idx==True)[0])

    if num_samples:
        idx = np.random.choice(len(subset), num_samples, replace=False)
        subset = torch.utils.data.dataset.Subset(dataset=subset, 
                                                 indices=idx)
    return subset


def new_dataset(train_dataset, test_dataset, size_dict):
    """
    Build new dataset from original, using the `size_dict` to set the \
    size of each wanted class.

    To create new (unbalanced) training set, we subset the original \
        training set according to  `size_dict` which gives an unbalanced \
        distribution. The new training set then only covers the classes \
        appeared in `size_dict` and its size shrinks to the sum of values \
        in `size_dict`.\n
    To create new test set, we only filter the classes mentioned and keep the\
        original size. Therefore, if a class is mentioned, all its test \
        samples will stay.
    
    :param train_dataset: The original train set.\n
    :type train_dataset: torch.utils.data.Dataset\n
    :param test_dataset: The original test set.\n
    :type test_dataset: torch.utils.data.Dataset\n
    :param size_dict: A list `(class_1: size_1, class_2: size_2, ..., \
        class_n: size_n)` where `class_i` denotes the class id and `size_i`\
        denotes the sample size of `class_i`.\n
    :type size_dict: dict\n

    :return: The new training and test set consisting of mentioned classes, \
        both of type `torch.utils.data.Dataset`.\n
    :rtype: tuple
    """
    # prepare train set
    subsets = []
    for i, (class_id, size) in enumerate(size_dict.items()):
        subsets.append(subset(train_dataset, class_id, size))
    new_train_dataset = torch.utils.data.ConcatDataset(subsets)
    
    # prepare test set
    subsets = []
    for i, (class_id, _) in enumerate(size_dict.items()):
        subsets.append(subset(test_dataset, class_id))
    new_test_dataset = torch.utils.data.ConcatDataset(subsets)
    return new_train_dataset, new_test_dataset


def accuracy(outputs, labels):
    """
    Calculate the accuracy given the predicted probability distribution and label.

    :param outputs: Model output given as tensor of shape `[batch_size, num_classes]`.
    :type outputs: torch.Tensor
    :param labels: True class given as tensor of shape `[batch_size,]`.
    :type labels: torch.Tensor

    :return: The accuracy for this batch.
    :rtype: float
    """
    assert outputs.size(0) == labels.size(0)
    _, pred = torch.max(outputs.data, 1)
    total = labels.size(0)
    hit = (pred == labels).sum()           
    return 1.0 * hit / total
