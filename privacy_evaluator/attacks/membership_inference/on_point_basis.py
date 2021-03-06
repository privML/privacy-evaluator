import logging
import numpy as np
from typing import Tuple

from .membership_inference import MembershipInferenceAttack
from ...classifiers.classifier import Classifier


class MembershipInferenceAttackOnPointBasis(MembershipInferenceAttack):
    """`MembershipInferenceAttackOnPointBasis` class."""

    def __init__(
        self,
        target_model: Classifier,
    ):
        """Initializes a `MembershipInferenceAttackOnPointBasis` class.

        :param target_model: Target model to be attacked.
        """
        super().__init__(target_model, init_art_attack=False)

    @MembershipInferenceAttack._fit_decorator
    def fit(self, *args, **kwargs):
        """Fits the attack model.

        :param args: Arguments for the fitting. Currently, there are no additional arguments provided.
        :param kwargs: Keyword arguments for fitting the attack model. Currently, there are no additional keyword
            arguments provided.
        """
        logger = logging.getLogger(__name__)
        logger.debug(
            "Trying to fit MembershipInferenceAttackOnPointBasis, nothing to fit."
        )

    def attack(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        num_bins: int = 15,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes each individual point's likelihood of being a member (denoted as privacy risk score in
        https://arxiv.org/abs/2003.10595).

        For an individual sample, its privacy risk score is computed as the posterior probability of being in the
        training set after observing its prediction output by the target machine learning model.

        (Helper method and description taken from
        https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L217)

        :param x_train: Data which was used to train the target model.
        :param y_train: One-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model.
        :param y_test: One-hot encoded labels for `x_test`.
        :param num_bins: The number of bins used to compute the training/test histogram.
        :return: Membership probability results.
        """
        logger = logging.getLogger(__name__)
        logger.info("Running MembershipInferenceAttackOnPointBasis")
        logger.debug("Computing Loss for target Model")
        loss_train = self.target_model.art_classifier.compute_loss(x_train, y_train)
        loss_test = self.target_model.art_classifier.compute_loss(x_test, y_test)

        logger.debug(
            "Computing membership probability per point with %d bins" % num_bins
        )
        return self._compute_membership_probability(loss_train, loss_test, num_bins)

    @staticmethod
    def _compute_membership_probability(
        loss_train, loss_test, num_bins: int = 15
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Has been taken from https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L217

        Helper function to compute_privacy_risk_score.

        :param loss_train: The loss of the target classifier on train data.
        :param loss_test: The loss of the target classifier on test data.
        :param num_bins: The number of bins used to compute the training/test histogram.
        :return: Membership probability results.
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
        test_hist_indices = (
            np.fmin(np.digitize(test_values, bins=bins_hist), num_bins) - 1
        )

        combined_hist = train_hist + test_hist
        combined_hist[combined_hist == 0] = small_value
        membership_prob_list = train_hist / (combined_hist + 0.0)

        train_membership_probs = membership_prob_list[train_hist_indices]
        test_membership_probs = membership_prob_list[test_hist_indices]

        return train_membership_probs, test_membership_probs
