from typing import Tuple, Dict
import numpy as np

from privacy_evaluator.metrics.basics import (
    accuracy,
    train_to_test_accuracy_gap,
    train_to_test_accuracy_ratio,
)
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

        :param target_model: Target model to be attacked.
        :param x_train: Data that was used to train the target model.
        :param y_train: Labels for the data that was used to train the target model.
        :param x_test: Data that was not used to train the target model.
        :param y_test: Labels for the data that was not used to train the target model.
        """
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def attack(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Performs the attack on the target model.

        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: Two arrays holding the inferred membership status. The first array includes the results for the
        inferred membership status of the train data and the second includes the results for the test data, where 1
        indicates a member and 0 indicates non-member. The optimal attack would return only ones for the first array and
        only zeros for the second.
        """
        return self.infer(*args, **kwargs)

    def infer(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Alias method for attack().

        :param args: Arguments of the attack.
        :param kwargs: Keyword arguments of the attack.
        :return: Two arrays holding the inferred membership status. The first array includes the results for the
        inferred membership status of the train data and the second includes the results for the test data, where 1
        indicates a member and 0 indicates non-member. The optimal attack would return only ones for the first array and
        only zeros for the second.
        """
        raise NotImplementedError(
            "Method 'infer()' needs to be implemented in subclass"
        )

    def attack_output(self, **kwargs) -> Dict:
        """Creates attack output metrics in an extractable format.

        :return: Attack output metrics including the target model train and test accuracy, target model train to test
        accuracy gap and ratio and the attack model accuracy.
        """

        train_accuracy = accuracy(self.y_train, self.target_model.predict(self.x_train))
        test_accuracy = accuracy(self.y_test, self.target_model.predict(self.x_test))
        attack_train_result, attack_test_result = self.attack(**kwargs)

        return {
            "target_model_train_accuracy": train_accuracy,
            "target_model_test_accuracy": test_accuracy,
            "target_model_train_to_test_accuracy_gap": train_to_test_accuracy_gap(
                train_accuracy, test_accuracy
            ),
            "target_model_train_to_test_accuracy_ratio": train_to_test_accuracy_ratio(
                train_accuracy, test_accuracy
            ),
            "attack_model_accuracy": accuracy(
                np.stack([attack_train_result, attack_test_result]),
                np.stack(
                    [
                        np.ones(attack_train_result.shape),
                        np.zeros(attack_test_result.shape),
                    ]
                ),
            ),
        }
