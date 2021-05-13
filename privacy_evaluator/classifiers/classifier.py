from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from typing import Union, Callable, Tuple
import torch
import numpy as np


class Classifier:

    def __init__(self, classifier: Union[Callable, torch.nn.Module], nb_classes: int, input_shape: Tuple[int, ...]):
        self._art_classifier = self._convert_to_art_classifier(classifier, nb_classes, input_shape)

    def predict(self, x: np.ndarray):
        return self._art_classifier.predict(x)

    @staticmethod
    def _convert_to_art_classifier(classifier: Union[Callable, torch.nn.Module], nb_classes: int, input_shape: Tuple[int, ...]):
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
