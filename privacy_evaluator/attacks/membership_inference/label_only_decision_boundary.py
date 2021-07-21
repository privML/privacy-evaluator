import numpy as np

from .membership_inference import MembershipInferenceAttack
from ...classifiers.classifier import Classifier


class MembershipInferenceLabelOnlyDecisionBoundaryAttack(MembershipInferenceAttack):
    """MembershipInferenceLabelOnlyDecisionBoundaryAttack class."""

    _ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS = "LabelOnlyDecisionBoundary"

    def __init__(
        self,
        target_model: Classifier,
    ):
        """Initializes a MembershipInferenceLabelOnlyDecisionBoundaryAttack class.

        :param target_model: Target model to be attacked.
        :param x_train: Data which was used to train the target model.
        :param y_train: One-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model.
        :param y_test: One-hot encoded labels for `x_test`.
        """
        super().__init__(target_model)

    @MembershipInferenceAttack._fit_decorator
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ):
        """Fits the attack model.

        :param x_train: Data which was used to train the target model and will be used for training the attack model.
        :param y_train: True, one-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model and will be used for training the attack model.
        :param y_test: True, one-hot encoded labels for `x_test`.
        :param kwargs: Keyword arguments for the fitting.
        """
        self._art_attack.calibrate_distance_threshold(
            x_train, y_train, x_test, y_test, **kwargs
        )
