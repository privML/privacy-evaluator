from art.attacks.inference.membership_inference import (
    MembershipInferenceBlackBoxRuleBased,
)
from typing import Tuple
import numpy as np

from privacy_evaluator.attacks.membership_inference.membership_inference import (
    MembershipInferenceAttack,
)
from privacy_evaluator.classifiers.classifier import Classifier


class MembershipInferenceBlackBoxRuleBasedAttack(MembershipInferenceAttack):
    """MembershipInferenceBlackBoxRuleBasedAttack class."""

    def __init__(
        self,
        target_model: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """Initializes a MembershipInferenceBlackBoxRuleBasedAttack class.

        :param target_model: The target model to be attacked.
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def infer(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Alias method for attack().

        :param args: The arguments of the attack.
        :param kwargs: The keyword arguments of the attack.
        :return: Result of the attack.
        """
        attack = MembershipInferenceBlackBoxRuleBased(self.target_model.art_classifier)

        inferred_train_data = attack.infer(self.x_train, self.y_train)
        inferred_test_data = attack.infer(self.x_test, self.y_test)

        return inferred_train_data, inferred_test_data
