from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from typing import Union, Callable, Tuple, Dict
import numpy as np
import torch

from privacy_evaluator.metrics.basics import *


# todo: add AttackInterface as soon as other PR merged
# todo: test why test and train accuracies are so bad
class MembershipInferenceAttack:

    def __init__(self, target_model: Union[Callable, torch.nn.Module], x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray):
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

    def infer(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Method 'infer()' needs to be implemented in subclass")

    def target_model_train_accuracy(self) -> np.float32:
        y_prediction = self.target_model.predict(self.x_train)
        y_prediction_ = np.zeros_like(y_prediction)
        y_prediction_[np.arange(y_prediction.shape[0]), np.argmax(y_prediction, axis=1)] = 1

        return accuracy(self.y_train, y_prediction)

    def target_model_test_accuracy(self) -> np.float32:
        y_prediction = self.target_model.predict(self.x_test)
        y_prediction_ = np.zeros_like(y_prediction)
        y_prediction_[np.arange(y_prediction.shape[0]), np.argmax(y_prediction, axis=1)] = 1

        return accuracy(self.y_test, y_prediction)

    def target_model_train_to_test_accuracy_gap(self) -> np.float32:
        """Calculates the gap between the train and test accuracy.

        The gap is calculated by subtracting the test accuracy from the train accuracy.

        :return: The gap between the train and test accuracy.
        :rtype: np.float32
        """
        return self.target_model_train_accuracy() - self.target_model_test_accuracy()

    def target_model_train_to_test_accuracy_ratio(self) -> np.float32:
        """Calculates the ratio between the train and test accuracy.

        The ratio is calculated by dividing the test accuracy by the train accuracy.

        :return: The ratio between the train and test accuracy.
        :rtype: np.float32
        """
        return self.target_model_train_accuracy() / self.target_model_test_accuracy()

    def attack_model_overall_accuracy(self, **kwargs) -> np.float32:
        inferred_train_data, inferred_test_data = self.infer(**kwargs)

        train_accuracy = np.sum(inferred_train_data) / len(inferred_train_data)
        test_accuracy = 1 - (np.sum(inferred_test_data) / len(inferred_test_data))

        return (train_accuracy * len(inferred_train_data) + test_accuracy * len(inferred_test_data)) / (len(inferred_train_data) + len(inferred_test_data))

    def model_card_info(self, **kwargs) -> Dict:
        return {
            'attack_model_overall_accuracy': self.attack_model_overall_accuracy(**kwargs),
            'target_model_train_test_accuracy_gap': self.target_model_train_test_accuracy_gap(),
            'target_model_train_test_accuracy_ratio': self.target_model_train_test_accuracy_ratio()
        }
