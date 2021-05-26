import pytest
import numpy as np

from privacy_evaluator.datasets.dataset import Dataset


def test_dataset():
    with pytest.raises(NotImplementedError):
        Dataset.numpy()
        Dataset.pytorch_loader()
        Dataset.tensorflow_loader()

    actual = Dataset._one_hot_encode(np.arange(10), 10)
    expected = np.eye(10)

    assert actual.shape == expected.shape
    assert (actual == expected).sum() == 100
