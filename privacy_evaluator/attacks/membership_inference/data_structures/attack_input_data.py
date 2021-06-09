from dataclasses import dataclass
import numpy as np

@dataclass
class AttackInputData:
    x_train: np.ndarray
    y_train: np.ndarray
    
    x_test: np.ndarray
    y_test: np.ndarray
