import numpy as np
from typing import Iterable

from .data_structures.attack_input_data import AttackInputData
from .data_structures.slicing import SlicePoints, Slicing
from ..membership_inference.on_point_basis import MembershipInferenceAttackOnPointBasis
from ...classifiers import Classifier
from ...output.user_output_privacy_score import UserOutputPrivacyScore


class MembershipInferencePointAnalysis:
    """`MembershipInferencePointAnalysis` class.

    `MembershipInferencePointAnalysis` makes it possible to apply slicing to `MembershipInferenceAttackOnPointBasis`.
    """

    def __init__(
        self,
        input_data: AttackInputData,
    ) -> None:
        """Initializes a `MembershipInferencePointAnalysis` class.

        :param input_data: Data for the membership inference attack on point basis.
        """
        self.input_data = input_data

    def analyse(
        self,
        target_model: Classifier,
        slicing: Slicing = Slicing(entire_dataset=True),
        num_bins: int = 15,
    ) -> Iterable[UserOutputPrivacyScore]:
        """Runs the membership inference on point basis attack.

        :param target_model: Target model to attack.
        :param slicing: Slicing specification. The slices will be created according to the specification and the attack
            will be run on each slice.
        :param num_bins: See `MembershipInferenceAttackOnPointBasis.attack` for details.
        """

        attack = MembershipInferenceAttackOnPointBasis(
            target_model=target_model,
        )

        results = []
        slices_names = []
        slcies_avg_train = []
        slcies_avg_test = []
        for slice in self.slices_point(target_model, slicing):
            membership_train_probs, membership_test_probs = attack.attack(
                self.input_data.x_train[slice.indices_train],
                self.input_data.y_train[slice.indices_train],
                self.input_data.x_test[slice.indices_test],
                self.input_data.y_test[slice.indices_test],
                num_bins,
            )

            slices_names.append(slice.desc)
            slcies_avg_train.append(membership_train_probs.mean())
            slcies_avg_test.append(membership_test_probs.mean())

            output = UserOutputPrivacyScore(
                np.argmax(self.input_data.y_train[slice.indices_train], axis=1),
                membership_train_probs,
            )
            output.histogram_distribution(class_name=slice.desc)
            results.append(output)

        output.histogram_slices(slices_names, slcies_avg_train, name="train")
        output.histogram_slices(slices_names, slcies_avg_test, name="test")

        return results

    def slices_point(self, target_model: Classifier, slicing: Slicing):
        """Generates slices according to the specification.

        :param target_model: Target model to attack.
        :param slicing: Slicing specification.
        """

        if slicing.entire_dataset:
            yield SlicePoints(
                indices_train=np.arange(len(self.input_data.y_train)),
                indices_test=np.arange(len(self.input_data.y_test)),
                desc="Entire dataset",
            )

        if slicing.by_class:
            for label in range(target_model.to_art_classifier().nb_classes):
                yield SlicePoints(
                    indices_train=np.argwhere(
                        (label == self.input_data.y_train.argmax(axis=1)) == True
                    ).flatten(),
                    indices_test=np.argwhere(
                        (label == self.input_data.y_test.argmax(axis=1)) == True
                    ).flatten(),
                    desc=f"Class={label}",
                )
