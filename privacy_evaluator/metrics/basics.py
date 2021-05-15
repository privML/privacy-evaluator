import numpy as np


def accuracy(y: np.ndarray, y_prediction: np.ndarray) -> np.float32:
    """Calculates accuracy for true labels and predicted labels.

    :params y: True labels.
    :params y_prediction: Predicted lables.
    :returns: Accuracy
    """
    return (np.argmax(y, axis=1) == np.argmax(y_prediction, axis=1)).sum() / y.shape[0]
