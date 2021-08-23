from .user_output import UserOutput
from typing import Dict


class UserOutputPropertyInferenceAttack(UserOutput):

    """User Output for Inference Attacks Class"""

    def __init__(self, max_message: str, output: Dict[str, float]):
        """
        :param max_message: message with prediction of attack (most probable property)
        :param output: whole output, includes: max_message, predictions for all subattacks
        """
        self.max_message = max_message
        self.output = output
