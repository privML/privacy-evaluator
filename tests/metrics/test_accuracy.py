import pytest

from privacy_evaluator.metrics.accuracy import accuracy_difference
from privacy_evaluator.metrics.accuracy import accuracy_proportion


def test_accuracy_difference():
    assert pytest.approx(accuracy_difference(0.95, 0.91)) == 0.04


def test_accuracy_proportion():
    assert pytest.approx(accuracy_proportion(0.9, 0.8)) == 1.125
