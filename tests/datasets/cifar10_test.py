import pytest

from privacy_evaluator.datasets.cifar10 import CIFAR10


def test_cifar10():
    CIFAR10.numpy()
    CIFAR10.pytorch_loader()
    with pytest.raises(NotImplementedError):
        CIFAR10.tensorflow_loader()
