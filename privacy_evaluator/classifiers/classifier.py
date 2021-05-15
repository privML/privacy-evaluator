from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from typing import Union, Callable, Tuple
import torch
import numpy as np


class Classifier:
    """Classifier base class."""

    def __init__(
        self,
        classifier: Union[Callable, torch.nn.Module],
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
        classifier: Union[Callable, torch.nn.Module],
        nb_classes: int,
        input_shape: Tuple[int, ...],
    ):
        """Converts a classifier to an ART classifier.

        :param classifier: Classifier to be converted. Either a Pytorch or Tensorflow classifier.
        :param nb_classes: Number of classes that were used to train the classifier.
        :param input_shape: Input shape of a data point of the classifier.
        """
        if isinstance(classifier, torch.nn.Module):
            return PyTorchClassifier(
                model=classifier,
                loss=None,
                nb_classes=nb_classes,
                input_shape=input_shape,
            )
        else:
            return TensorFlowV2Classifier(
                model=classifier,
                nb_classes=nb_classes,
                input_shape=input_shape,
            )
