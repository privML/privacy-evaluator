import pytest

from privacy_evaluator.metrics.accuracy import accuracy_difference
from privacy_evaluator.metrics.accuracy import accuracy_ratio


def test_accuracy_difference():
    assert pytest.approx(accuracy_difference(0.95, 0.91)) == 0.04


def test_accuracy_ratio():
    assert pytest.approx(accuracy_ratio(0.9, 0.8)) == 1.125
