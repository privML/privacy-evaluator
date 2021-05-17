from typing import Tuple, Dict
import numpy as np

from privacy_evaluator.metrics.basics import accuracy
from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier


class MembershipInferenceAttack(Attack):
    """MembershipInferenceAttack base class."""

    def __init__(
        self,
        target_model: Classifier,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        """Initializes a MembershipInferenceAttack class.

        :param target_model: The target model to be attacked.
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the attack on the target model.

        :param args: The arguments of the attack.
        :param kwargs: The keyword arguments of the attack.
        :return: Result of the attack.
        """
        return self.infer(*args, **kwargs)

    def infer(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Alias method for attack().

        :param args: The arguments of the attack.
        :param kwargs: The keyword arguments of the attack.
        :return: Result of the attack.
        """
        raise NotImplementedError(
            "Method 'infer()' needs to be implemented in subclass"
        )

    def target_model_train_accuracy(self) -> np.float32:
        """Calculates the train accuracy of the target model.

        :return: Train accuracy.
        """
        return accuracy(self.y_train, self.target_model.predict(self.x_train))

    def target_model_test_accuracy(self) -> np.float32:
        """Calculates the test accuracy of the target model.

        :return: Test accuracy.
        """
        return accuracy(self.y_test, self.target_model.predict(self.x_test))

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
        """Calculates the overall accuracy of the attack model.

        :return: Overall accuracy.
        """
        inferred_train_data, inferred_test_data = self.infer(**kwargs)

        train_accuracy = np.sum(inferred_train_data) / len(inferred_train_data)
        test_accuracy = 1 - (np.sum(inferred_test_data) / len(inferred_test_data))

        return (
            train_accuracy * len(inferred_train_data)
            + test_accuracy * len(inferred_test_data)
        ) / (len(inferred_train_data) + len(inferred_test_data))

    def model_card_info(self, **kwargs) -> Dict:
        """Creates model card info in an extractable format.

        :return: Model card info with the target model train and test accuracy, target model train to test accuracy
        gap and ratio and the privacy risk score.
        """
        return {
            "target_model_train_accuracy": self.target_model_train_accuracy(),
            "target_model_test_accuracy": self.target_model_test_accuracy(),
            "target_model_train_to_test_accuracy_gap": self.target_model_train_to_test_accuracy_gap(),
            "target_model_train_to_test_accuracy_ratio": self.target_model_train_to_test_accuracy_ratio(),
            "privacy_risk_score": self.attack_model_overall_accuracy(**kwargs),
        }
