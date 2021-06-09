from dataclasses import dataclass
from typing import Union
from typing import Iterable
import numpy as np

@dataclass
class Slicing:
    entire_dataset: bool = True
    by_classification_correctness: bool = False
    by_class: bool = False


@dataclass
class Slice:
    indices: np.ndarray
    desc: str

    def __str__(self):
        return "\n".join((
            "Slice(",
            "  indices: " + np.array2string(self.indices, threshold=10, edgeitems=2),
            "  desc: " + self.desc,
            ")",
        ))