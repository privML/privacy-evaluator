import pytest

from privacy_evaluator.metrics.basics import *


def test_accuracy():
    assert pytest.approx(accuracy(np.ones((10, 10)), np.ones((10, 10)))) == 1.0
    with pytest.raises(ValueError):
        accuracy(np.ones(10), np.ones((10, 10)))


def test_train_to_test_accuracy_gap():
    assert pytest.approx(train_to_test_accuracy_gap(0.80, 0.75)) == 0.05


def test_train_to_test_accuracy_ratio():
    assert pytest.approx(train_to_test_accuracy_ratio(0.80, 0.40)) == 2.0
