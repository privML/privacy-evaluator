from typing import Tuple

import numpy as np

from .user_output import UserOutput


class UserOutpurInferenceAttackAnalysis(UserOutput):
    """Result of the membership inference attack analysis for a single slice."""

    def __init__(
        self,
        slice_desc: str,
        slice_data_points: np.ndarray,
        advantage: float,
    ):
        self.slice_desc = (slice_desc,)
        self.slice_data_points = (slice_data_points,)
        self.advantage = advantage
