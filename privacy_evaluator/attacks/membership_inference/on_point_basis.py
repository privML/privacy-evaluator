from .membership_inference import MembershipInferenceAttack
from ...classifiers.classifier import Classifier
import numpy as np
from typing import Tuple


class MembershipInferenceAttackOnPointBasis(MembershipInferenceAttack):
    """MembershipInferenceBlackBoxAttack class."""
    def __init__(
        self,
        target_model: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        attack_model_type: str = "nn",
    ):
        """Initializes a MembershipInferenceOnPointBasis class.

        :param target_model: Target model to be attacked.
        :param x_train: Data which was used to train the target model.
        :param y_train: True, one-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model.
        :param y_test: True, one-hot encoded labels for `x_test`.
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
            x_train,
            y_train,
            x_test,
            y_test,
            attack_model_type=attack_model_type,
        )

    @MembershipInferenceAttack._fit_decorator
    def fit(self, **kwargs):
        """Fits the attack model.

        :param kwargs: Keyword arguments for the fitting.
        """
        pass


    def attack(
        self,
        num_bins=15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes each individual point's likelihood of being a member
        (denoted as privacy risk score in https://arxiv.org/abs/2003.10595).

        For an individual sample, its privacy risk score is computed as the posterior
        probability of being in the training set
        after observing its prediction output by the target machine learning model.

        (Helper method and description taken from
        https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L217)


        :param num_bins: the number of bins used to compute the training/test histogram
        :return: membership probability results
        """
        loss_train = self.target_model.art_classifier.compute_loss(self.x_train, self.y_train)
        loss_test = self.target_model.art_classifier.compute_loss(self.x_test, self.y_test)
        return self._compute_membership_probability(loss_train, loss_test, num_bins)


    def _compute_membership_probability(
        self, loss_train, loss_test, num_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Has been taken from https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L217
        Helper function to compute_privacy_risk_score
        :param loss_train: the loss of the target classifier on train data
        :param loss_test: the loss of the target classifier on test data
        :param num_bins: the number of bins used to compute the training/test histogram
        :return: membership probability results
        """

        train_values = loss_train
        test_values = loss_test
        # Compute the histogram in the log scale
        small_value = 1e-10
        train_values = np.maximum(train_values, small_value)
        test_values = np.maximum(test_values, small_value)

        min_value = min(train_values.min(), test_values.min())
        max_value = max(train_values.max(), test_values.max())
        bins_hist = np.logspace(np.log10(min_value), np.log10(max_value), num_bins + 1)

        train_hist, _ = np.histogram(train_values, bins=bins_hist)
        train_hist = train_hist / (len(train_values) + 0.0)
        train_hist_indices = (
            np.fmin(np.digitize(train_values, bins=bins_hist), num_bins) - 1
        )

        test_hist, _ = np.histogram(test_values, bins=bins_hist)
        test_hist = test_hist / (len(test_values) + 0.0)
        test_hist_indices = np.fmin(np.digitize(test_values, bins=bins_hist), num_bins) - 1

        combined_hist = train_hist + test_hist
        combined_hist[combined_hist == 0] = small_value
        membership_prob_list = train_hist / (combined_hist + 0.0)

        train_membership_probs = membership_prob_list[train_hist_indices]
        test_membership_probs = membership_prob_list[test_hist_indices]

        return train_membership_probs, test_membership_probs
