from typing import Tuple
import numpy as np

from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier


class SampleAttack(Attack):
    def __init__(self, target_model: Classifier, x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray):
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        return np.ones(10), np.zeros(10)
