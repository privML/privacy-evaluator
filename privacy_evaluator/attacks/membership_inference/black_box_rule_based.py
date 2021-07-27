import logging

from .membership_inference import MembershipInferenceAttack
from ...classifiers.classifier import Classifier


class MembershipInferenceBlackBoxRuleBasedAttack(MembershipInferenceAttack):
    """MembershipInferenceBlackBoxRuleBasedAttack class.

    For information about this attacks outcome, please see to membership_inference.py.
    """

    _ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS = "MembershipInferenceBlackBoxRuleBased"

    def __init__(
        self,
        target_model: Classifier,
    ):
        """Initializes a MembershipInferenceBlackBoxRuleBasedAttack class.

        :param target_model: Target model to be attacked.
        """
        super().__init__(target_model)
        self._art_attack_model_fitted = True

    @MembershipInferenceAttack._fit_decorator
    def fit(self, *args, **kwargs):
        """Fits the attack model.

        :param args: Arguments for the fitting. Currently, there are no additional arguments provided.
        :param kwargs: Keyword arguments for fitting the attack model. Currently, there are no additional keyword
        arguments provided.
        """
        logger = logging.getLogger(__name__)
        logger.debug(
            "Trying to fit MembershipInferenceBlackBoxRuleBasedAttack, nothing to fit."
        )
        pass
