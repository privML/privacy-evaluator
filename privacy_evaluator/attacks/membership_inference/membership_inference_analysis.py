from ...output.user_output_inference_attack_analysis import (
    UserOutputInferenceAttackAnalysis,
)
from typing import Iterable, Type
import numpy as np
from . import MembershipInferenceAttack
from .data_structures.attack_input_data import AttackInputData
from ...classifiers import Classifier
from .data_structures.slicing import Slicing
from .data_structures.slicing import Slice
from sklearn import metrics


class MembershipInferenceAttackAnalysis:
    """Represents the membership inference attack analysis class.

    Interpretation of Outcome:

    Advantage Score:
    The attacker advantage is a score that relies on comparing the model output on member and non-member data points.
    The model outputs are probability values over all classes, and they are often different on member and non-member
    data points. Usually, the model is more confident on member data points, because it has seen them during training.
    When trying to find a threshold value to tell apart member and non-member samples by their different model outputs,
    the attacker has interest in finding the best ratio between false positives “fpr” (non-members that are classified
    as members) and true positives “tpr” (members that are correctly identifies as members).
    This best ratio is calculated as the max(tpr-fpr) over all threshold values and represents the attacker advantage.

    Slicing: Incorrectly classified:
    It is normal that the attacker is more successful to deduce membership on incorrectly classified samples than on
    correctly classified ones. This results from the fact, that model predictions are often better on training than on
    test data points, whereby your attack model might learn to predict incorrectly classified samples as non-members.
    If your model overfits the training data, this assumption might hold true often enough to make the attack seem more
    successful on this slice. If you wish to reduce that, pay attention to reducing your model’s overfitting.

    Slicing: Specific classes:
    Specific classes can be differently vulnerable. It may seem that the membership inference attack is more successful
    on some classes than on the other classes. Research has shown that the class distribution (and also the distribution
    of data points within one class) are factors that influence the vulnerability of a class for membership inference
    attacks [1]. Also, small classes (belonging to minority groups) can be more prone to membership inference
    attacks [2]. One reason for this could be, that there is less data for that class, and therefore, the model overfits
    within this class. It might make sense to look into the vulnerable classes of your model again, and maybe add more
    data to them, use private synthetic data, or introduce privacy methods like Differential Privacy [2]. Attention, the
    use of Differential Privacy could have a negative influence on the performance of your model for the minority
    classes.

    References:
    [1] Stacey Truex, Ling Liu, Mehmet Emre Gursoy, Lei Yu, and Wenqi Wei. 2019.Demystifying Membership Inference
    Attacks in Machine Learning as a Service.IEEE Transactions on Services Computing(2019)
    [2] Suriyakumar, Vinith M., Nicolas Papernot, Anna Goldenberg, and Marzyeh Ghassemi. "Chasing Your Long Tails:
    Differentially Private Prediction in Health Care Settings." In Proceedings of the 2021 ACM Conference on Fairness,
    Accountability, and Transparency, pp. 723-734. 2021.
    """

    def __init__(
        self,
        attack_type: Type[MembershipInferenceAttack],
        input_data: AttackInputData,
        **kwargs,
    ) -> None:
        """Initializes a MembershipInferenceAttackAnalysis class.

        :param attack_type: Type of membership inference attack to analyse.
        :param input_data: Data for the membership inference attack.
        :param attack_kwargs: kwargs passed to the attack.
        """
        self.attack_type = attack_type
        self.input_data = input_data
        self.attack_kwargs = kwargs

    def analyse(
        self,
        target_model: Classifier,
        x: np.ndarray,
        y: np.ndarray,
        membership: np.ndarray,
        slicing: Slicing = Slicing(entire_dataset=True),
    ) -> Iterable[UserOutputInferenceAttackAnalysis]:
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
            **self.attack_kwargs,
        )

        attack.fit(
            x_train=self.input_data.x_train,
            y_train=self.input_data.y_train,
            x_test=self.input_data.x_test,
            y_test=self.input_data.y_test,
        )

        results = []
        for slice in slices(x, y, target_model, slicing):
            membership_prediction = attack.attack(
                x[slice.indices], y[slice.indices], probabilities=True
            )

            # Calculate the advantage score as in tensorflow privacy package.
            tpr, fpr, _ = metrics.roc_curve(
                membership[slice.indices],
                membership_prediction,
                drop_intermediate=False,
            )
            advantage = max(np.abs(tpr - fpr))
            accuracy = (
                membership_prediction.round() == membership[slice.indices]
            ).sum() / len(slice.indices)

            results.append(
                UserOutputInferenceAttackAnalysis(
                    slice=slice,
                    advantage=advantage,
                    accuracy=accuracy,
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
