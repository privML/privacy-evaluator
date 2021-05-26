from typing import Tuple
import numpy as np

from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier


<<<<<<< HEAD
    def perform_attack(params):
        pass
=======
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

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, ...]:
        """Performs the attack on the target model.

        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: Two arrays holding the results of the attack. The first array includes the results for the train data
        and the second includes the results for the test data.
        """
        return np.ones(10), np.zeros(10)
>>>>>>> a2c92363fa47ad8e5e5a52176753621a11ebe57b
