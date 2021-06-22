import pytest
import numpy as np

from privacy_evaluator.validators import *


def test_validate_number_of_features():
    validate_number_of_features(np.ones((10, 10)), "test", 10)
    with pytest.raises(ValueError):
        validate_number_of_features(np.ones((10, 10)), "test", 5)


def test_validate_number_of_dimensions():
    validate_number_of_dimensions(np.ones((10, 10)), "test", 2)
    with pytest.raises(ValueError):
        validate_number_of_dimensions(np.ones((10, 10)), "test", 1)


def test_validate_matching_number_of_samples():
    validate_matching_number_of_samples(
        np.ones((10, 10)), "test", np.ones((10, 10)), "test"
    )
    with pytest.raises(ValueError):
        validate_matching_number_of_samples(
            np.ones((10, 10)), "test", np.ones((5, 10)), "test"
        )


def test_validate_one_hot_encoded():
    validate_one_hot_encoded(np.ones((10, 10)), "test", 10)
    with pytest.raises(ValueError):
        validate_one_hot_encoded(np.ones(10), "test", 10)
