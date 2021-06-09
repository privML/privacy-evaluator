from dataclasses import dataclass
from typing import Iterable, Type
import numpy as np
from . import MembershipInferenceAttack
from .data_structures.attack_input_data import AttackInputData
from ...classifiers import Classifier
from .data_structures.slicing import Slicing
from.data_structures.slicing import Slice
from sklearn import metrics
from textwrap import indent


@dataclass
class MembershipInferenceAttackAnalysisSliceResult:
    slice: Slice
    advantage: float

    def __str__(self):
        return "\n".join((
            "MembershipInferenceAttackAnalysisSliceResult(",
            indent(str(self.slice), "  "),
            f"  advantage: {self.advantage:.4f}",
            ")",
        ))


class MembershipInferenceAttackAnalysis:
    def __init__(
            self, 
            attack_type: Type[MembershipInferenceAttack], 
            input_data: AttackInputData) -> None:
        self.attack_type = attack_type
        self.input_data = input_data

    def analyse(
            self,
            target_model: Classifier, 
            x: np.ndarray, 
            y: np.ndarray, 
            membership: np.ndarray,
            slicing: Slicing) -> Iterable[MembershipInferenceAttackAnalysisSliceResult]:
        attack = self.attack_type(
            target_model=target_model, 
            x_train=self.input_data.x_train, 
            y_train=self.input_data.y_train, 
            x_test=self.input_data.x_test,
            y_test=self.input_data.y_test)

        # TODO: fit on the whole thing or should we fit for each slice?
        attack.fit()

        results = []
        for slice in slices(x, y, target_model, slicing):
            membership_prediction = attack.attack(x[slice.indices], y[slice.indices])
            tpr, fpr, _ = metrics.roc_curve(
                membership[slice.indices], membership_prediction, drop_intermediate=False)
            advantage = max(np.abs(tpr - fpr))
            results.append(
                MembershipInferenceAttackAnalysisSliceResult(
                    slice=slice,
                    advantage=advantage,
                )
            )

        return results


def slices(
        x: np.ndarray, 
        y: np.ndarray,
        target_model: Classifier, 
        slicing: Slicing):

    if slicing.entire_dataset:
        yield Slice(
            indices=np.arange(len(x)),
            desc="Entire dataset"
        )

    if slicing.by_classification_correctness:
        prediction = target_model.predict(x).argmax(axis=1)
        result = (prediction == y.argmax(axis=1))
        yield Slice(
            indices=np.argwhere(result == True).flatten(),
            desc="Correctly classified"
        )

        yield Slice(
            indices=np.argwhere(result == False).flatten(),
            desc="Incorrectly classified"
        )
    
    if slicing.by_class:
        for label in range(target_model.to_art_classifier().nb_classes):
            yield Slice(
                indices=np.argwhere((label == y.argmax(axis=1)) == True).flatten(),
                desc=f"Class={label}"
            )
