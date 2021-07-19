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

        :param args: Arguments for the fitting.
        :param kwargs: Keyword arguments for the fitting.
        """
        pass
