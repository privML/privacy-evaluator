from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from typing import Union, Callable, Tuple
import numpy as np
import torch

from privacy_evaluator.attacks.membership_inference.membership_inference import MembershipInferenceAttack


class MembershipInferenceBlackBoxAttack(MembershipInferenceAttack):
    
    def __init__(self, target_model: Union[Callable, torch.nn.Module], x_train: np.ndarray, y_train: np.ndarray,
                 x_test: np.ndarray, y_test: np.ndarray, attack_train_ratio: float = 0.5):
        super().__init__(target_model, x_train, y_train, x_test, y_test)

        self.attack_train_ratio = attack_train_ratio
        self.attack_train_size = int(len(x_train) * attack_train_ratio)
        self.attack_test_size = int(len(x_test) * attack_train_ratio)
        
    def infer(self, attack_model_type: str = "nn",) -> Tuple[np.ndarray, np.ndarray]:
        assert attack_model_type in ["rf", "gb", "nn"]

        attack = MembershipInferenceBlackBox(self.target_model, attack_model_type=attack_model_type)

        attack.fit(
            self.x_train[:self.attack_train_size],
            self.y_train[:self.attack_train_size],
            self.x_test[:self.attack_test_size],
            self.y_test[:self.attack_test_size]
        )
        
        inferred_train_data = attack.infer(
            self.x_train[self.attack_train_size:],
            self.y_train[self.attack_train_size:]
        )
        inferred_test_data = attack.infer(
            self.x_test[self.attack_test_size:],
            self.y_test[self.attack_test_size:]
        )

        return inferred_train_data, inferred_test_data
