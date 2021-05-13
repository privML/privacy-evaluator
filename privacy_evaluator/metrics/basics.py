import numpy as np


def accuracy(y: np.ndarray, y_prediction: np.ndarray) -> np.float32:
    tp = true_positive(y, y_prediction)
    tn = true_negative(y, y_prediction)
    return (tp + tn) / (tp + tn + false_positive(y, y_prediction) + false_negative(y, y_prediction))


def true_positive(y: np.ndarray, y_prediction):
    return (y[np.argwhere(y_prediction == 1)] == y_prediction[np.argwhere(y_prediction == 1)]).sum()


def true_negative(y: np.ndarray, y_prediction: np.ndarray):
    return (y[np.argwhere(y_prediction == 0)] == y_prediction[np.argwhere(y_prediction == 0)]).sum()


def false_positive(y: np.ndarray, y_prediction: np.ndarray):
    return (y[np.argwhere(y_prediction == 1)] != y_prediction[np.argwhere(y_prediction == 1)]).sum()


def false_negative(y: np.ndarray, y_prediction: np.ndarray):
    return (y[np.argwhere(y_prediction == 0)] != y_prediction[np.argwhere(y_prediction == 0)]).sum()


def precision(y: np.ndarray, y_prediction: np.ndarray):
    tp = true_positive(y, y_prediction)
    return tp / (tp + false_positive(y, y_prediction) + np.nextafter(0, 1))


def recall(y: np.ndarray, y_prediction: np.ndarray):
    tp = true_positive(y, y_prediction)
    return tp / (tp + false_negative(y, y_prediction) + np.nextafter(0, 1))


def f1_score(y: np.ndarray, y_prediction: np.ndarray):
    p = precision(y, y_prediction)
    r = recall(y, y_prediction)
    return 2 * p * r / (p + r + np.nextafter(0, 1))
