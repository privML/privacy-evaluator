import numpy as np

from .membership_inference import (
    MembershipInferenceAttack,
)
from ...classifiers.classifier import Classifier


class MembershipInferenceBlackBoxRuleBasedAttack(MembershipInferenceAttack):
    """MembershipInferenceBlackBoxRuleBasedAttack class."""

    _ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS = "MembershipInferenceBlackBoxRuleBased"

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
        self._art_attack_model_fitted = True

    @MembershipInferenceAttack._fit_decorator
    def fit(self, **kwargs):
        """Fits the attack model.

        :param kwargs: Keyword arguments for the fitting.
        """
        pass
