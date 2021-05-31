from privacy_evaluator.classifiers.classifier import Classifier
import numpy as np


def compute_privacy_risk_score(
    target: Classifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_bins=15,
):
    """
    Computes each individual point's likelihood of being a member
    (denoted as privacy risk score in https://arxiv.org/abs/2003.10595).

    For an individual sample, its privacy risk score is computed as the posterior
    probability of being in the training set
    after observing its prediction output by the target machine learning model.

    (Helper method and description taken from
    https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L217)

    Args:
        attack_input: input data for compute membership probability
        num_bins: the number of bins used to compute the training/test histogram
    Returns:
        membership probability results
    """
    loss_train = target.art_classifier.compute_loss(x_train, y_train)
    loss_test = target.art_classifier.compute_loss(x_test, y_test)
    return _compute_membership_probability(loss_train, loss_test, num_bins)


def _compute_membership_probability(loss_train, loss_test, num_bins: int = 15):
    """
    Has been taken from https://github.com/tensorflow/privacy/blob/master/tensorflow_privacy/privacy/membership_inference_attack/membership_inference_attack.py#L217
    Helper function to compute_privacy_risk_score
    Args:
        loss_train: the loss of the target classifier on train data
        loss_test: the loss of the target classifier on test data
        num_bins: the number of bins used to compute the training/test histogram
    Returns:
        membership probability results
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
