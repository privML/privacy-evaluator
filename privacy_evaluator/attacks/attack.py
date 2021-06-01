import numpy as np

from privacy_evaluator.classifiers.classifier import Classifier


class Attack:
    """Attack base class."""

    def __init__(
        self,
        target_model: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """Initializes a Attack class.

        :param target_model: Target model to be attacked.
        :param x_train: Data which was used to train the target model.
        :param y_train: True labels for `x_train`.
        :param x_test: Data that was not used to train the target model.
        :param y_test: True labels for `x_test`.
        """
        self.target_model = target_model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def attack(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Performs the attack on the target model.

        :param x: Data to be attacked.
        :param y: True labels for `x`.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the results of the attack.
        """
        raise NotImplementedError(
            "Method `attack()` needs to be implemented in subclass"
        )
