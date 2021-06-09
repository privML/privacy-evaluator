from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from typing import Union, Tuple
import numpy as np
import tensorflow as tf
import torch


class Classifier:
    """Classifier base class."""

    def __init__(
        self,
        classifier: Union[tf.keras.Model, torch.nn.Module],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ):
        """Initializes a Classifier class.

        :param classifier: The classifier. Either a Pytorch or Tensorflow classifier.
        :param nb_classes: Number of classes that were used to train the classifier.
        :param input_shape: Input shape of a data point of the classifier.
        """
        self.art_classifier = self._to_art_classifier(
            classifier, nb_classes, input_shape
        )

    def predict(self, x: np.ndarray):
        """Predicts labels for given data.

        :param x: Data which labels should be predicted for.
        :return: Predicted labels.
        """
        return self.art_classifier.predict(x)

    @staticmethod
    def _to_art_classifier(
        classifier: Union[tf.keras.Model, torch.nn.Module],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ) -> Union[TensorFlowV2Classifier, PyTorchClassifier]:
        """Converts a classifier to an ART classifier.

        :param classifier: Classifier to be converted. Either a Pytorch or Tensorflow classifier.
        :param nb_classes: Number of classes that were used to train the classifier.
        :param input_shape: Input shape of a data point of the classifier.
        :return: Given classifier converted to an ART classifier.
        :raises TypeError: If the given classifier is of an invalid type.
        """
        if isinstance(classifier, torch.nn.Module):
            return PyTorchClassifier(
                model=classifier,
                loss=None,
                nb_classes=nb_classes,
                input_shape=input_shape,
            )
        if isinstance(classifier, tf.keras.Model):
            return TensorFlowV2Classifier(
                model=classifier, nb_classes=nb_classes, input_shape=input_shape,
            )
        else:
            raise TypeError(
                f"Expected classifier to be an instance of {str(torch.nn.Module)} or {str(tf.keras.Model)}, received {str(type(classifier))} instead."
            )
