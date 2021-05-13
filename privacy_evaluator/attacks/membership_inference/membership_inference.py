from typing import Tuple, Dict
import numpy as np

from privacy_evaluator.metrics.basics import accuracy
from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier


# todo: doc/test/examples
class MembershipInferenceAttack(Attack):

    def __init__(self, target_model: Classifier, x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray):
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return self.infer(*args, **kwargs)

    def infer(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
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
            'target_model_train_test_accuracy_gap': self.target_model_train_to_test_accuracy_gap(),
            'target_model_train_test_accuracy_ratio': self.target_model_train_to_test_accuracy_ratio()
        }
