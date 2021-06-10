from dataclasses import dataclass
from typing import Iterable, Type
import numpy as np
from . import MembershipInferenceAttack
from .data_structures.attack_input_data import AttackInputData
from ...classifiers import Classifier
from .data_structures.slicing import Slicing
from .data_structures.slicing import Slice
from sklearn import metrics
from textwrap import indent


@dataclass
class MembershipInferenceAttackAnalysisSliceResult:
    """Result of the membership inference attack analysis for a single slice."""

    # The slice for which this result was produced.
    slice: Slice

    # Advantage score calculated by the membership inference attack analysis.
    advantage: float

    def __str__(self):
        """Human-readable representation of the result."""

        return "\n".join(
            (
                "MembershipInferenceAttackAnalysisSliceResult(",
                indent(str(self.slice), "  "),
                f"  advantage: {self.advantage:.4f}",
                ")",
            )
        )


class MembershipInferenceAttackAnalysis:
    """Represents the membership inference attack analysis class."""

    def __init__(
        self, attack_type: Type[MembershipInferenceAttack], input_data: AttackInputData
    ) -> None:
        """Initializes a MembershipInferenceAttackAnalysis class.

        :param attack_type: Type of membership inference attack to analyse.
        :param input_data: Data for the membership inference attack.
        """
        self.attack_type = attack_type
        self.input_data = input_data

    def analyse(
        self,
        target_model: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        membership: np.ndarray,
        slicing: Slicing = Slicing(entire_dataset=True),
    ) -> Iterable[MembershipInferenceAttackAnalysisSliceResult]:
        """Runs the membership inference attack and calculates attacker's advantage for each slice.

        :param target_model: Target model to attack.
        :param x: Input data to attack.
        :param y: True labels for `x`.
        :param membership: Labels representing the membership for each data sample in `x`. 1 for member and 0 for non-member.
        :param slicing: Slicing specification. The slices will be created according to the specification and the attack will be run on each slice.
        """

        # Instantiate an object of the given attack type.
        attack = self.attack_type(
            target_model=target_model,
            x_train=self.input_data.x_train,
            y_train=self.input_data.y_train,
            x_test=self.input_data.x_test,
            y_test=self.input_data.y_test,
        )

        # TODO: fit on the whole thing or should we fit for each slice?
        attack.fit()

        results = []
        for slice in slices(x, y, target_model, slicing):
            membership_prediction = attack.attack(x[slice.indices], y[slice.indices])

            # Calculate the advantage score as in tensorflow privacy package.
            tpr, fpr, _ = metrics.roc_curve(
                membership[slice.indices],
                membership_prediction,
                drop_intermediate=False,
            )
            advantage = max(np.abs(tpr - fpr))

            results.append(
                MembershipInferenceAttackAnalysisSliceResult(
                    slice=slice,
                    advantage=advantage,
                )
            )

        return results


def slices(x: np.ndarray, y: np.ndarray, target_model: Classifier, slicing: Slicing):
    """Generates slices according to the specification.

    :param x: Input data to attack.
    :param y: True labels for `x`.
    :param target_model: Target model to attack.
    :param slicing: Slicing specification.
    """

    if slicing.entire_dataset:
        yield Slice(indices=np.arange(len(x)), desc="Entire dataset")

    if slicing.by_classification_correctness:
        # Use the target model to predict the classes for given samples
        prediction = target_model.predict(x).argmax(axis=1)
        result = prediction == y.argmax(axis=1)
        yield Slice(
            indices=np.argwhere(result == True).flatten(), desc="Correctly classified"
        )

        yield Slice(
            indices=np.argwhere(result == False).flatten(),
            desc="Incorrectly classified",
        )

    if slicing.by_class:
        for label in range(target_model.to_art_classifier().nb_classes):
            yield Slice(
                indices=np.argwhere((label == y.argmax(axis=1)) == True).flatten(),
                desc=f"Class={label}",
            )
