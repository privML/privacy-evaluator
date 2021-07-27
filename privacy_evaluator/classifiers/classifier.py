from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
import numpy as np
import tensorflow as tf
import torch
from typing import Union, Tuple


class Classifier:
    """`Classifier` class."""

    def __init__(
        self,
        classifier: Union[tf.Module, torch.nn.Module],
        loss: Union[tf.losses.Loss, torch.nn.modules.loss._Loss],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ):
        """Initializes a `Classifier` class.

        :param classifier: The classifier. Either a Pytorch or TensorFlow classifier.
        :param nb_classes: Number of classes that were used to train the classifier.
        :param input_shape: Input shape of a data point of the classifier.
        """
        self.loss = loss
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.art_classifier = self._to_art_classifier(
            classifier, loss, nb_classes, input_shape
        )

    def predict(self, x: np.ndarray):
        """Predicts labels for given data.

        :param x: Data which labels should be predicted for.
        :return: Predicted labels.
        """
        return self.art_classifier.predict(x)

    def to_art_classifier(self):
        """Converts the classifier to an ART classifier.

        :return: Converted ART classifier.
        """
        return self.art_classifier

    @staticmethod
    def _to_art_classifier(
        classifier: Union[tf.Module, torch.nn.Module],
        loss: Union[tf.losses.Loss, torch.nn.modules.loss._Loss],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ) -> Union[TensorFlowV2Classifier, PyTorchClassifier]:
        """Initializes an ART classifier.

        :param classifier: Original classifier, either Pytorch or TensorFlow.
        :param nb_classes: Number of classes that were used to train the classifier.
        :param input_shape: Shape of a input data point of the classifier.
        :return: Instance of an ART classifier.
        :raises TypeError: If `classifier` is of invalid type.
        """
        if isinstance(classifier, torch.nn.Module):
            return PyTorchClassifier(
                model=classifier,
                loss=loss,
                nb_classes=nb_classes,
                input_shape=input_shape,
            )
        if isinstance(classifier, tf.Module):
            return TensorFlowV2Classifier(
                model=classifier,
                loss_object=loss,
                nb_classes=nb_classes,
                input_shape=input_shape,
            )
        else:
            raise TypeError(
                f"Expected `classifier` to be an instance of {str(torch.nn.Module)} or {str(tf.Module)}, received {str(type(classifier))} instead."
            )
