import numpy as np

from .attack import Attack
from ..classifiers.classifier import Classifier
from ..validators.attack import validate_parameters


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
        :param x_train: Data which was used to train the target model.
        :param y_train: True, one-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model.
        :param y_test: True, one-hot encoded labels for `x_test`.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def attack(self, x: np.ndarray, y: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Performs the attack on the target model.

        :param x: Data to be attacked.
        :param y: True, one-hot encoded labels for `x`.
        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the result of the attack.
        """

        validate_parameters(
            "attack",
            target_model=self.target_model,
            x=x,
            y=y,
        )
        return np.ones(10)
