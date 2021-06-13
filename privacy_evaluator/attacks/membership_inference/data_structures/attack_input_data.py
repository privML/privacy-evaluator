from dataclasses import dataclass
import numpy as np


@dataclass
class AttackInputData:
    """Input data for running an attack analysis."""

    # Data samples from the training set.
    x_train: np.ndarray

    # Labels for the data samples from the training set.
    y_train: np.ndarray

    # Data samples from the test set.
    x_test: np.ndarray

    # Labels for the data samples from the test set.
    y_test: np.ndarray
