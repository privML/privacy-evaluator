import numpy as np

from privacy_evaluator.metrics.basics import *
from privacy_evaluator.output.user_output import UserOutput


def test_output_function():
    data_y = np.array(
        ["blue", "orange", "red", "orange", "red", "red", "blue", "red", "orange"]
    )
    priv_risk = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    user_output = UserOutput(
        data_y, priv_risk, ["green", "blue", "red", "orange", "white"]
    )
    labels, count = user_output.histogram_top_k(4, show_diagram=False)
    assert (labels == np.array(["green", "blue", "red", "orange", "white"])).all()
    assert (count == np.array([0, 1, 2, 1, 0])).all()
