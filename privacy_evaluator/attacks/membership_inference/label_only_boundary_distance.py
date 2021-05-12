from art.attacks.inference.membership_inference import LabelOnlyDecisionBoundary
from typing import Union, Callable, Tuple
import numpy as np
import torch

from privacy_evaluator.attacks.membership_inference.membership_inference import MembershipInferenceAttack


class MembershipInferenceLabelOnlyDecisionBoundaryAttack(MembershipInferenceAttack):
    
    def __init__(self, target_model: Union[Callable, torch.nn.Module], x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray):
        super().__init__(target_model, x_train, y_train, x_test, y_test)

    def infer(self) -> Tuple[np.ndarray, np.ndarray]:
        attack = LabelOnlyDecisionBoundary(self.target_model)

        attack.calibrate_distance_threshold(
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
        )

        inferred_train_data = attack.infer(
            self.x_train,
            self.y_train
        )
        inferred_test_data = attack.infer(
            self.x_test,
            self.y_test
        )
        return inferred_train_data, inferred_test_data
