import numpy as np

from privacy_evaluator.classifiers.classifier import Classifier


# todo: remove x_train, ... from init and move it to fit() and attack()
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
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        """
        self.target_model = target_model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def attack(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the attack on the target model.

        :param x: Input data to attack.
        :param y: True labels for x.
        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the result of the attack.
        """
        raise NotImplementedError(
            "Method `attack()` needs to be implemented in subclass"
        )
