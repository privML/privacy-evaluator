from typing import Tuple
import numpy as np

from privacy_evaluator.classifiers.classifier import Classifier


class Attack:

    def __init__(self, target_model: Classifier, x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray):
        """Initializes the Attack Class.
        :param target_model: the target model to be attacked.
        :param x_train: data that was used to train the target model.
        :param y_train: labels for the data that was used to train the target model.
        :param x_test: data that was not used to train the target model.
        :param y_test: labels for the data that was not used to train the target model.
        """
        self.target_model = target_model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        """Performs the attack on the target model.

        :param args: The arguments of the attack.
        :param kwargs: The keyword arguments of the attack.
        :return: Result of the attack.
        """
        raise NotImplementedError("Method 'attack()' needs to be implemented in subclass")