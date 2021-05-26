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

        :param target_model: Target model to be attacked.
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def infer(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Alias method for attack().

        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: Two arrays holding the inferred membership status. The first array includes the results for the
        inferred membership status of the train data and the second includes the results for the test data, where 1
        indicates a member and 0 indicates non-member. The optimal attack would return only ones for the first array and
        only zeros for the second.
        """
        attack = MembershipInferenceBlackBoxRuleBased(self.target_model.art_classifier)

        inferred_train_data = attack.infer(self.x_train, self.y_train)
        inferred_test_data = attack.infer(self.x_test, self.y_test)

        return inferred_train_data, inferred_test_data
