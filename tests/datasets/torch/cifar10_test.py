import pytest
import numpy as np

from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10


def test_cifar10():
    TorchCIFAR10.numpy()
    TorchCIFAR10.data_loader()

    actual = TorchCIFAR10.one_hot_encode(np.arange(10))
    expected = np.eye(10)

    assert actual.shape == expected.shape
    assert (actual == expected).sum() == 100

