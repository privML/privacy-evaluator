import pytest

from privacy_evaluator.metrics.basics import *


def test_accuracy():
    assert pytest.approx(accuracy(np.ones((10, 10)), np.ones((10, 10)))) == 1.0