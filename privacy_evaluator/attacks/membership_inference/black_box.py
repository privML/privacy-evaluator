import numpy as np

from .membership_inference import MembershipInferenceAttack
from ...classifiers.classifier import Classifier


class MembershipInferenceBlackBoxAttack(MembershipInferenceAttack):
    """MembershipInferenceBlackBoxAttack class.

    For information about this attacks outcome, please see to membership_inference.py.
    """

    _ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS = "MembershipInferenceBlackBox"

    def __init__(
        self,
        target_model: Classifier,
        attack_model_type: str = "nn",
    ):
        """Initializes a MembershipInferenceBlackBoxAttack class.

        :param target_model: Target model to be attacked.
        :param x_train: Data which was used to train the target model.
        :param y_train: One-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model.
        :param y_test: One-hot encoded labels for `x_test`.
        :param attack_model_type: Type of the attack model. On of "rf", "gb", "nn".
        :raises TypeError: If `attack_model_type` is of invalid type.
        :raises ValueError: If `attack_model_type` is none of `rf`, `gb`, `nn`.
        """
        if not isinstance(attack_model_type, str):
            raise TypeError(
                f"Expected `attack_model_type` to be an instance of {str(str)}, received {str(type(attack_model_type))} instead."
            )
        if attack_model_type not in ["rf", "gb", "nn"]:
            raise ValueError(
                f"Expected `attack_model_type` to be one of `rf`, `gb`, `nn`, received {attack_model_type} instead."
            )
        super().__init__(
            target_model,
            attack_model_type=attack_model_type,
        )

    @MembershipInferenceAttack._fit_decorator
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs,
    ):
        """Fits the attack model.

        :param x_train: Data which was used to train the target model and will be used for training the attack model.
        :param y_train: True, one-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model and will be used for training the attack model.
        :param y_test: True, one-hot encoded labels for `x_test`.
        :param kwargs: Keyword arguments for the fitting.
        """
        self._art_attack.fit(x_train, y_train, x_test, y_test)
