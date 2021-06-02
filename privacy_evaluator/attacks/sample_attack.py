import numpy as np

from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier


class SampleAttack(Attack):
    """SampleAttack class."""

    def __init__(
        self,
        target_model: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """Initializes a SampleAttack class.

        :param target_model: Target model to be attacked.
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def attack(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the attack on the target model.

        :param x: Input data to attack.
        :param y: True labels for x.
        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the result of the attack.
        """
        return np.ones(10)
