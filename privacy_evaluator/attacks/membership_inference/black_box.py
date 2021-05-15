from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from typing import Tuple
import numpy as np

from privacy_evaluator.attacks.membership_inference.membership_inference import (
    MembershipInferenceAttack,
)
from privacy_evaluator.classifiers.classifier import Classifier


class MembershipInferenceBlackBoxAttack(MembershipInferenceAttack):
    """MembershipInferenceBlackBoxAttack class."""

    def __init__(
        self,
        target_model: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        attack_train_ratio: float = 0.5,
    ):
        """Initializes a MembershipInferenceBlackBoxAttack class.

        :param target_model: The target model to be attacked.
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        :param attack_train_ratio: Ratio between used test and train data from the target model to train the attack model.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

        self.attack_train_ratio = attack_train_ratio
        self.attack_train_size = int(len(x_train) * attack_train_ratio)
        self.attack_test_size = int(len(x_test) * attack_train_ratio)

    def infer(
        self, attack_model_type: str = "nn", *args, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Alias method for attack().

        :param attack_model_type: Type of the attack model. On of "rf", "gb", "nn".
        :param args: The arguments of the attack.
        :param kwargs: The keyword arguments of the attack.
        :return: Result of the attack.
        """
        assert attack_model_type in ["rf", "gb", "nn"]

        attack = MembershipInferenceBlackBox(
            self.target_model.art_classifier, attack_model_type=attack_model_type
        )

        attack.fit(
            self.x_train[: self.attack_train_size],
            self.y_train[: self.attack_train_size],
            self.x_test[: self.attack_test_size],
            self.y_test[: self.attack_test_size],
        )

        inferred_train_data = attack.infer(
            self.x_train[self.attack_train_size :],
            self.y_train[self.attack_train_size :],
        )
        inferred_test_data = attack.infer(
            self.x_test[self.attack_test_size :], self.y_test[self.attack_test_size :]
        )

        return inferred_train_data, inferred_test_data
