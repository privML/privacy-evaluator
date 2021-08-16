import numpy as np

from ..classifiers.classifier import Classifier


class Attack:
    """`Attack` class."""

    def __init__(
        self,
        target_model: Classifier,
    ):
        """Initializes an `Attack` class.

        :param target_model: Target model to be attacked.
        """
        self.target_model = target_model

    def attack(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """Performs the attack on the target model.

        :param x: Data to be attacked.
        :param y: One-hot encoded labels for `x`.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the results of the attack.
        """
        raise NotImplementedError(
            "Method `attack()` needs to be implemented in subclass"
        )
