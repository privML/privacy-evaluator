from dataclasses import dataclass
import numpy as np
from typing import Any


@dataclass
class Slicing:
    """Slicing specification for `MembershipInferenceAttackAnalysis` and `MembershipInferencePointAnalysis`."""

    # If true, the analysis will produce a slice of the whole dataset.
    entire_dataset: bool = True

    # If true, the analysis will produce two slices.
    # First for the correct classified data and second for incorrect classified data.
    by_classification_correctness: bool = False

    # If true, a slice will be created for each class/label.
    by_class: bool = False


@dataclass
class Slice:
    """Single slice that is created by `MembershipInferenceAttackAnalysis`."""

    # Indices of the data samples that are part of this slice.
    indices: Any[list, np.ndarray]

    # Human-readable description of the slice.
    desc: str

    def __str__(self):
        """Returns a humand-readable representation of the slice."""

        return "\n".join(
            (
                "Slice(",
                "  indices: "
                + np.array2string(self.indices, threshold=10, edgeitems=2)
                + f" ({len(self.indices)} items)",
                "  desc: " + self.desc,
                ")",
            )
        )


@dataclass
class SlicePoints:
    """Single slice that is created by `MembershipInferencePointAnalysis`."""

    # Indices of the data samples that are part of this slice that belong to the train set.
    indices_train: np.ndarray

    # Indices of the data samples that are part of this slice that belong to the test set.
    indices_test: np.ndarray

    # Human-readable description of the slice.
    desc: str

    def __str__(self):
        """Returns a humand-readable representation of the slice."""

        return "\n".join(
            (
                "Slice(",
                "  indices_train: "
                + np.array2string(self.indices_train, threshold=10, edgeitems=2)
                + f" ({len(self.indices_train)} items)",
                "  indices_test: "
                + np.array2string(self.indices_train, threshold=10, edgeitems=2)
                + f" ({len(self.indices_test)} items)",
                "  desc: " + self.desc,
                ")",
            )
        )
