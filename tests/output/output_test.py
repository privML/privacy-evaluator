import numpy as np

from privacy_evaluator.metrics.basics import *
from privacy_evaluator.output.user_output_privacy_score import UserOutputPrivacyScore


def test_output_function():
    data_y = np.array(
        ["blue", "orange", "red", "orange", "red", "red", "blue", "red", "orange"]
    )
    priv_risk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    user_output = UserOutputPrivacyScore(data_y, priv_risk)
    labels, count = user_output.histogram_top_k(
        np.array(["green", "blue", "red", "orange", "white"]),
        4,
        show_diagram=False,
    )


def test_output_function_relative():
    data_y = np.array(
        ["blue", "orange", "red", "orange", "red", "red", "blue", "red", "orange"]
    )
    priv_risk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    user_output = UserOutputPrivacyScore(data_y, priv_risk)
    labels, count = user_output.histogram_top_k_relative(
        np.array([5, 40, 6, 35, 4]),
        np.array(["green", "blue", "red", "orange", "white"]),
        4,
        show_diagram=False,
    )
