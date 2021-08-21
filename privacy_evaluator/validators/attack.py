import sys

from .basic import *
from ..classifiers import Classifier


def validate_parameters(method: str, *args, **kwargs):
    """Validates parameters for given method.

    :param method: Method for which parameters are validated.
    :param args: Arguments to be validated.
    :param kwargs: Keyword arguments to be validated.
    """
    getattr(sys.modules[__name__], f"_validate_{method}_parameters")(*args, **kwargs)


def _validate_fit_parameters(
    target_model: Classifier,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    **_,
):
    """Validates parameters for `fit` method.

    :param target_model: Target model to be validated.
    :param x_train: Data which was used to train the target model to be validated.
    :param y_train: True labels for `x_train` to be validated.
    :param x_test: Data that was not used to train the target model to be validated.
    :param y_test: True labels for `x_test` to be validated.
    """
    validate_number_of_features(
        x_train, "x_train", target_model.art_classifier.input_shape[0]
    )
    validate_matching_number_of_samples(x_train, "x_train", y_train, "y_train")
    validate_one_hot_encoded(y_train, "y_train", target_model.nb_classes)

    validate_number_of_features(
        x_test, "x_test", target_model.art_classifier.input_shape[0]
    )
    validate_matching_number_of_samples(x_test, "x_test", y_test, "y_test")
    validate_one_hot_encoded(y_test, "y_test", target_model.nb_classes)


def _validate_attack_parameters(target_model: Classifier, x: np.ndarray, y: np.ndarray):
    """Validates parameters for `attack` method.

    :param target_model: Target model to be validated.
    :param x: Data to be validated.
    :param y: True labels for `x` to be validated.
    """
    validate_number_of_features(x, "x", target_model.art_classifier.input_shape[0])
    validate_matching_number_of_samples(x, "x", y, "y")
    validate_one_hot_encoded(y, "y", target_model.nb_classes)


def _validate_attack_output_parameters(
    target_model: Classifier, x: np.ndarray, y: np.ndarray, y_attack: np.ndarray
):
    """Validates parameters for `attack_output` method.

    :param target_model: Target model to be validated.
    :param x: Data to be validated.
    :param y: True labels for `x` to be validated.
    :param y_attack: True labels for the attack model to be validated.
    """
    validate_number_of_features(x, "x", target_model.art_classifier.input_shape[0])
    validate_matching_number_of_samples(x, "x", y, "y")
    validate_one_hot_encoded(y, "y", target_model.nb_classes)
    validate_matching_number_of_samples(x, "x", y_attack, "y_attack")
    validate_number_of_dimensions(y_attack, "y_attack", 1)
