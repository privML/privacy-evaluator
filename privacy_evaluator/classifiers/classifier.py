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
        loss: Union[tf.keras.losses.Loss, torch.nn.modules.loss._Loss],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ):
        """Initializes a Classifier class.

        :param classifier: The classifier. Either a Pytorch or Tensorflow classifier.
        :param nb_classes: Number of classes that were used to train the classifier.
        :param input_shape: Input shape of a data point of the classifier.
        """
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
        classifier: Union[tf.keras.Model, torch.nn.Module],
        loss: Union[tf.keras.losses.Loss, torch.nn.modules.loss._Loss],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ) -> Union[TensorFlowV2Classifier, PyTorchClassifier]:
        """Initializes an ART classifier.

        :param classifier: Original classifier, either Pytorch or Tensorflow.
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
        if isinstance(classifier, tf.keras.Model):
            return TensorFlowV2Classifier(
                model=classifier,
                loss_object=loss,
                nb_classes=nb_classes,
                input_shape=input_shape,
            )
        else:
            raise TypeError(
                f"Expected `classifier` to be an instance of {str(torch.nn.Module)} or {str(tf.keras.Model)}, received {str(type(classifier))} instead."
            )
