import numpy as np


def validate_number_of_features(data: np.ndarray, label: str, number_of_features: int):
    """Validates the number of features of given data.

    :param data: Data to be validated.
    :param label: Label for `data` to provide an understandable exception message.
    :param number_of_features: Number of features `data` should have.
    :raises ValueError: If `data` do not have `number_of_features`.
    """
    if data.shape[1] != number_of_features:
        raise ValueError(
            f"Expected `{label}` to have {number_of_features} number of features, instead got {data.shape[1]}."
        )


def validate_number_of_dimensions(
    data: np.ndarray, label: str, number_of_dimensions: int
):
    """Validates the number of dimensions of given data.

    :param data: Data to be validated.
    :param label: Label for `data` to provide an understandable exception message.
    :param number_of_dimensions: Number of dimensions `data` should have.
    :raises ValueError: If `data` do not have `number_of_dimensions`.
    """
    if len(data.shape) != number_of_dimensions:
        raise ValueError(
            f"Expected `{label}` to have {number_of_dimensions} number of dimensions, instead got {len(data.shape)}."
        )


def validate_matching_number_of_samples(
    data_a: np.ndarray, label_data_a: str, data_b: np.ndarray, label_data_b: str
):
    """Validates the number of samples of two given datasets.

    :param data_a: Dataset a to be validated.
    :param label_data_a: Label for `data_a` to provide an understandable exception message.
    :param data_b: Dataset b to be validated.
    :param label_data_b: Label for `data_b` to provide an understandable exception message.
    :raises ValueError: If `data_a` and `data_b` do not have the same number of samples.
    """
    if data_a.shape[0] != data_b.shape[0]:
        raise ValueError(
            f"Number of samples in `{label_data_a}` and `{label_data_b}` do not match"
        )


def validate_one_hot_encoded(data: np.ndarray, label: str, number_of_classes: int):
    """Validates if given data is one-hot encoded.

    :param data: Data to be validated.
    :param label: Label for `data` to provide an understandable exception message.
    :param number_of_classes: Number of classes `data` should be one-hot encoded for.
    :raises ValueError: If `data` is not properly one-hot encoded.
    """
    try:
        validate_number_of_dimensions(data, label, 2)
        validate_number_of_features(data, label, number_of_classes)
    except ValueError:
        raise ValueError(
            f"Expected `{label}` to be one-hot encoded and of shape ({data.shape[0]}, {number_of_classes}), instead got ({data.shape})."
        )
