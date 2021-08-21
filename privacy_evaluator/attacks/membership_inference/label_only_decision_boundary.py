import logging
import numpy as np

from .membership_inference import MembershipInferenceAttack
from ...classifiers.classifier import Classifier


class MembershipInferenceLabelOnlyDecisionBoundaryAttack(MembershipInferenceAttack):
    """`MembershipInferenceLabelOnlyDecisionBoundaryAttack` class.

    For information about this attacks outcome, please see to `membership_inference.py`.
    """

    _ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS = "LabelOnlyDecisionBoundary"

    def __init__(
        self,
        target_model: Classifier,
    ):
        """Initializes a `MembershipInferenceLabelOnlyDecisionBoundaryAttack` class.

        :param target_model: Target model to be attacked.
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
        :param y_train: One-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model and will be used for training the attack model.
        :param y_test: One-hot encoded labels for `x_test`.
        :param kwargs: Keyword arguments for fitting the attack model. Possible kwargs are:
        :kwargs norm: Order of the norm for HopSkipJump. Possible values: "inf", np.inf or 2.
        :kwargs max_iter: Maximum number of iterations for HopSkipJump.
        :kwargs max_eval: Maximum number of evaluations for estimating gradient for HopSkipJump.
        :kwargs init_eval: Initial number of evaluations for estimating gradient for HopSkipJump.
        :kwargs init_size: Maximum number of trials for initial generation of adversarial examples for HopSkipJump.

        For more details about the HopSkipJump parameters, please read the following paper:
        https://arxiv.org/pdf/1904.02144.pdf
        """
        logger = logging.getLogger(__name__)
        logger.info("fiting MembershipInferenceLabelOnlyDecisionBoundaryAttack")
        self._art_attack.calibrate_distance_threshold(
            x_train, y_train, x_test, y_test, **kwargs
        )
