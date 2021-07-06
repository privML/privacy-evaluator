import sys

from .basic import *
from ..classifiers import Classifier


def validate_parameters(method: str, **kwargs):
    """Validates parameters for given method.

    :param method: Method for which parameters are validated.
    :param kwargs: Parameters to be validated as keyword arguments.
    """
    getattr(sys.modules[__name__], f"_validate_{method}_parameters")(**kwargs)


def _validate_fit_parameters(**kwargs):
    """Validates parameters for `fit` method.

    :param kwargs: Parameters to be validated as keyword arguments.
    """
    if ["target_model", "x_train", "y_train", "x_test", "y_test"] in kwargs:
        validate_number_of_features(
            kwargs["x_train"],
            "x_train",
            kwargs["target_model"].art_classifier.input_shape[0],
        )
        validate_matching_number_of_samples(
            kwargs["x_train"], "x_train", kwargs["y_train"], "y_train"
        )
        validate_one_hot_encoded(
            kwargs["y_train"], "y_train", kwargs["target_model"].nb_classes
        )

        validate_number_of_features(
            kwargs["x_test"],
            "x_test",
            kwargs["target_model"].art_classifier.input_shape[0],
        )
        validate_matching_number_of_samples(
            kwargs["x_test"], "x_test", kwargs["y_test"], "y_test"
        )
        validate_one_hot_encoded(
            kwargs["y_test"], "y_test", kwargs["target_model"].nb_classes
        )


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
