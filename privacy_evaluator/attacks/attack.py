from typing import Tuple
import numpy as np

from ..classifiers.classifier import Classifier


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

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        """Performs the attack on the target model.

        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: Two arrays holding the results of the attack. The first array includes the results for the train data
        and the second includes the results for the test data.
        """
        raise NotImplementedError(
            "Method 'attack()' needs to be implemented in subclass"
        )
