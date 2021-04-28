def accuracy_difference(train_accuracy, test_accuracy):
    """Calculate the difference between the train  and test accuracy.

    The difference is calculated by subtracting the test accuracy from thee train accuracy. The result is a positive
    number if the train accuracy is higher than the test accuracy and a negative number if the train accuracy is smaller
    than the test accuracy.

    :param train_accuracy: Accuracy of the train phase.
    :type train_accuracy: int, float
    :param test_accuracy: Accuracy of the test phase.
    :type test_accuracy: int, float

    :return: The difference between the train  and test accuracy.
    :rtype: int, float
    """
    return train_accuracy - test_accuracy


def accuracy_proportion(train_accuracy, test_accuracy):
    """Calculate the proportion between the train  and test accuracy.

    The proportion is calculated by dividing the test accuracy from thee train accuracy. The result is higher than 1 if
    the train accuracy is higher than the test accuracy and smaller than 1 if the train accuracy is smaller.
    than the test accuracy.

    :param train_accuracy: Accuracy of the train phase.
    :type train_accuracy: int, float
    :param test_accuracy: Accuracy of the test phase. Must be non-zero.
    :type test_accuracy: int, float

    :return: The proportion between the train  and test accuracy.
    :rtype: float
    """
    return train_accuracy / test_accuracy
