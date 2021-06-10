import numpy as np

from privacy_evaluator.metrics.basics import *
from privacy_evaluator.output.user_output_privacy_score import UserOutputPrivacyScore
from privacy_evaluator.output.user_output_inference_attack import (
    UserOutputInferenceAttack,
)


def test_output_priv_score_function():
    data_y = np.array(
        ["blue", "orange", "red", "orange", "red", "red", "blue", "red", "orange"]
    )
    priv_risk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    user_output = UserOutputPrivacyScore(data_y, priv_risk)
    labels, count = user_output.histogram_top_k(
        np.array(["green", "blue", "red", "orange", "white"]), 4, show_diagram=False
    )
    assert (labels == np.array(["green", "blue", "red", "orange", "white"])).all()
    assert (count == np.array([0, 1, 2, 1, 0])).all()
    assert (
        user_output._to_json()
        == '{"attack_data_y": ["blue", "orange", "red", "orange", "red", "red", "blue", "red", "orange"], "privacy_risk": [1, 2, 3, 4, 5, 6, 7, 8, 9]}'
    )
    assert (
        user_output._to_json(["privacy_risk"])
        == '{"privacy_risk": [1, 2, 3, 4, 5, 6, 7, 8, 9]}'
    )
    data_y = np.array(
        ["blue", "orange", "red", "orange", "red", "red", "blue", "red", "orange"]
    )
    priv_risk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    user_output = UserOutputPrivacyScore(data_y, priv_risk)
    labels, count = user_output.histogram_top_k_relative(
        np.array(["green", "blue", "red", "orange", "white"]),
        4,
        show_diagram=False,
    )
    assert (labels == np.array(["green", "blue", "red", "orange", "white"])).all()
    np.testing.assert_array_almost_equal(count, np.array([0, 0.5, 0.5, 0.333333, 0]))


def test_output_inference_attack_function():
    user_output = UserOutputInferenceAttack(0.9, 0.8, 0.1, 1.125, 0.75)
    assert (
        user_output._to_json()
        == '{"target_model_train_accuracy": 0.9, "target_model_test_accuracy": 0.8, "target_model_train_to_test_accuracy_gap": 0.1, "target_model_train_to_test_accuracy_ratio": 1.125, "attack_model_accuracy": 0.75}'
    )
    assert (
        user_output._to_json(
            ["target_model_train_accuracy", "target_model_test_accuracy"]
        )
        == '{"target_model_train_accuracy": 0.9, "target_model_test_accuracy": 0.8}'
    )
