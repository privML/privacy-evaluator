from dataclasses import dataclass
import numpy as np


@dataclass
class Slicing:
    """Slicing specification for the membership inference attack analysis."""

    # If true, the analysis will produce a slice of the whole dataset.
    entire_dataset: bool = True

    # If true, the analysis will produce two slices.
    # First for the correct classified data and second for incorrect classified data.
    by_classification_correctness: bool = False

    # If true, a slice will be created for each class/label.
    by_class: bool = False


@dataclass
class Slice:
    """Single slice that is created by the membership inference attack analysis."""

    # Indices of the data samples that are part of this slice.
    indices: np.ndarray

    # Human-readable description of the slice.
    desc: str

    def __str__(self):
        """Returns a humand-readable representation of the slice."""

        return "\n".join(
            (
                "Slice(",
                "  indices: "
                + np.array2string(self.indices, threshold=10, edgeitems=2),
                "  desc: " + self.desc,
                ")",
            )
        )