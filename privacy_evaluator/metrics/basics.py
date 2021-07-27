import numpy as np


def accuracy(y: np.ndarray, y_prediction: np.ndarray) -> np.float32:
    """Calculates accuracy for true labels and predicted labels.

    :params y: True labels.
    :params y_prediction: Predicted labels.
    :return: Calculated accuracy.
    :raises ValueError: If `y` and `y_prediction` are not of the same shape.
    """
    if y.shape != y_prediction.shape:
        raise ValueError(
            f"Expected `y` and `y_prediction` to be of same shape, received `y` with shape {str(y.shape)} and "
            f"`y_prediction` with shape {str(y_prediction.shape)} instead."
        )

    if len(y.shape) != 1:
        y = np.argmax(y, axis=1)
        y_prediction = np.argmax(y_prediction, axis=1)

    return (y == y_prediction).sum() / y.shape[0]


def train_to_test_accuracy_gap(
    train_accuracy: np.float32, test_accuracy: np.float32
) -> np.float32:
    """Calculates the gap between the train and test accuracy of a classifier.

    The gap is calculated by subtracting the test accuracy from the train accuracy.

    :params train_accuracy: The train accuracy.
    :params test_accuracy: The test accuracy.
    :return: The gap between the train and test accuracy.
    """
    return train_accuracy - test_accuracy


def train_to_test_accuracy_ratio(
    train_accuracy: np.float32, test_accuracy: np.float32
) -> np.float32:
    """Calculates the ratio between the train and test accuracy of a classifier.

    The ratio is calculated by dividing the test accuracy by the train accuracy.

    :params train_accuracy: The train accuracy.
    :params test_accuracy: The test accuracy.
    :return: The ratio between the train and test accuracy.
    """
    return train_accuracy / test_accuracy
