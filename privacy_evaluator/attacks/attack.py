from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from typing import Union, Callable, Tuple, Dict
import numpy as np
import torch


class Attack:

    """Initilizes the Attack Class.
    :param model: the model to be attacked.
    :type model: ART-Classifier
    :param x_train: training data of the model.
    :type x_train: np.ndarray
    :param y_train: labels of the training data.
    :type y_train: np.ndarray
    :param x_test: test data of the model.
    :type x_test: np.ndarray
    :param y_test: labels of the test data.
    :type y_test: np.ndarray
    """

    def __init__(
        self,
        target_model: Union[Callable, torch.nn.Module],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):

        if isinstance(target_model, torch.nn.Module):
            self.target_model = PyTorchClassifier(
                model=target_model,
                loss=None,
                nb_classes=y_train.shape[1],
                input_shape=x_train.shape[1:],
            )
        else:
            self.target_model = TensorFlowV2Classifier(
                model=target_model,
                nb_classes=y_train.shape[1],
                input_shape=x_train.shape[1:],
            )

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    """ Performs the actual attack on the model
    :param prams: The prarameters of the Attack
    :type params: dict
    return: result of the attack
    """

    def perform_attack(self, params):
        raise NotImplementedError
