from .user_output import UserOutput


class UserOutputInferenceAttack(UserOutput):

    """User Output for Inference Attacks Class"""

    def __init__(
        self,
        target_model_train_accuracy: float,
        target_model_test_accuracy: float,
        target_model_train_to_test_accuracy_gap: float,
        target_model_train_to_test_accuracy_ratio: float,
        attack_model_accuracy: float,
    ):
        """
        Initilaizes the Class with values
        """
        self.target_model_train_accuracy = target_model_train_accuracy
        self.target_model_test_accuracy = target_model_test_accuracy
        self.target_model_train_to_test_accuracy_gap = (
            target_model_train_to_test_accuracy_gap
        )
        self.target_model_train_to_test_accuracy_ratio = (
            target_model_train_to_test_accuracy_ratio
        )
        self.attack_model_accuracy = attack_model_accuracy
