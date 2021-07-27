from textwrap import indent

from .user_output import UserOutput
from ..attacks.membership_inference.data_structures.slicing import Slice


class UserOutputInferenceAttackAnalysis(UserOutput):
    """Result of the membership inference attack analysis for a single slice."""

    def __init__(
        self,
        slice: Slice,
        advantage: float,
        accuracy: float,
    ):
        self.slice = slice
        self.advantage = advantage
        self.accuracy = accuracy

    def to_json(self, include_indices=False) -> str:
        """Serialize the output to JSON.

        :param include_indices: If True, slice indices will be included in the output.
        """
        if include_indices:
            slice = Slice(
                indices=self.slice.indices.tolist(),
                desc=self.slice.desc,
            )
        else:
            slice = Slice(
                indices=[],
                desc=self.slice.desc,
            )

        output = UserOutputInferenceAttackAnalysis(
            slice=slice.__dict__,
            advantage=self.advantage,
            accuracy=self.accuracy,
        )

        return UserOutput._to_json(
            output,
        )

    def __str__(self) -> str:
        return "\n".join(
            (
                "UserOutputInferenceAttackAnalysis(",
                indent(str(self.slice), "  "),
                f"  advantage: {self.advantage:.3f}",
                f"  accuracy: {self.accuracy:.3f}",
                ")",
            )
        )
