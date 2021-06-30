from .user_output import UserOutput
from typing import Dict


class UserOutputPropertyInferenceAttack(UserOutput):

    """User Output for Inference Attacks Class"""

    def __init__(
        self,
        max_message: str,
        output: Dict[str, float]
    ):
        """
        Initilaizes the Class with values
        """
        self.max_message = max_message
        self.output = output
