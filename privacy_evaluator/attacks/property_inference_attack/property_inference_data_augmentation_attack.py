from torch.nn.modules import adaptive
from ...classifiers.classifier import Classifier
from ...utils import data_utils
from ...output.user_output_property_inference_attack import (
    UserOutputPropertyInferenceAttack,
)
from .property_inference_attack import PropertyInferenceAttack

import numpy as np
import logging
from tqdm.auto import tqdm
from typing import Tuple, Dict, List
from collections import OrderedDict

# count of shadow training sets, must be even
AMOUNT_SETS = 2
# ratio and size for unbalanced data sets
SIZE_SHADOW_TRAINING_SET = 1000
# ratios for different properties in sub-attacks
RATIOS_FOR_ATTACK = [
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
]

# number of epochs for training the meta classifier
NUM_EPOCHS_META_CLASSIFIER = 20

# The type of adaptation. ('mask', 'random_noise', 'brightness')
#   1. 'mask'-adaptation: A white-colored box is added at a random location for each image. The side length of the masking
#      box can be further specified by parameter `box_len`.
#   2. 'random_noise'-adaptation: Normal-distributed noise is added to each image. The mean and deviation of the noise
#      can be further specified by `mean` and `std` respectively.
#   3. 'brightness'-adaptation: A fixed value given by parameter `brightness` is added to each pixel of each image. If
#      `brightness > 0` then brighter; if `brightness < 0`, then darker.
ADAPTATION = "mask"

# ratio of negation of property
NEGATIVE_RATIO = 0.5


class PropertyInferenceDataAugmentationAttack(PropertyInferenceAttack):
    def __init__(
        self,
        target_model: Classifier,
        dataset: Tuple[np.ndarray, np.ndarray],
        amount_sets: int = AMOUNT_SETS,
        size_shadow_training_set: int = SIZE_SHADOW_TRAINING_SET,
        ratios_for_attack: List[int] = RATIOS_FOR_ATTACK,
        negative_ratio: int = NEGATIVE_RATIO,
        verbose: int = 0,
        num_epochs_meta_classifier: int = NUM_EPOCHS_META_CLASSIFIER,
        adaptation: str = ADAPTATION,
        **kwargs,
    ):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        :param dataset: dataset for training of shadow classifiers, test_data from dataset
        :param amount_sets: count of shadow training sets, must be even
        :param size_shadow_training_set: ratio and size for unbalanced data sets
        :param ratios_for_attack: ratios for different properties in sub-attacks
        :param verbose: 0: no information; 1: backbone (most important) information; 2: utterly detailed information will be printed
        :param num_epochs_meta_classifier: number of epochs for training the meta classifier
        :params adaptation: The type of adaptation.
        :params **kwargs: Optional parameters for the specified adaptation.

        Optional params:
        :params box_len: Involved when `adaptation` is "mask", the side length of masking boxes.
        :params brightness: Involved when `adaptation` is "brightness", the amount the brightness should be raised or lowered.
        :params mean: Involved when `adaptation` is "random_noise", the mean of the added noise.
        :params std: Involved when `adaptation` is "random_noise", the standard deviation of the added noise.
        """
        self.adaptation = adaptation
        self.negative_ratio = negative_ratio
        self.kwargs = kwargs

        super().__init__(
            target_model,
            dataset,
            amount_sets,
            size_shadow_training_set,
            ratios_for_attack,
            num_epochs_meta_classifier,
            verbose,
        )

    def create_shadow_training_sets(
        self, ratio: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create the shadow training sets with given ratio.
        The function works for the specific binary case that the ratio is a fixed distribution
        specified in the input.
        :params adaptation: The type of adaptation.
        :params **kwargs: Optional parameters for the specified adaptation.
        :return: The adapted images.

        Optional params:
        :params box_len: Involved when `adaptation` is "mask", the side length of masking boxes.
        :params brightness: Involved when `adaptation` is "brightness", the amount the brightness should be raised or lowered
        :params mean: Involved when `adaptation` is "random_noise", the mean of the added noise.
        :params std: Involved when `adaptation` is "random_noise", the standard deviation of the added noise.
        :return: shadow training sets for given ratio
        """

        training_sets = []

        # Creation of shadow training sets with the size dictionaries
        # amount_sets divided by 2 because amount_sets describes the total amount of shadow training sets.
        # In this function however only all shadow training sets of one type (follow property OR negation of property) are created, hence amount_sets / 2.
        self.logger.info("Creating shadow training sets")

        for _ in range(int(self.amount_sets / 2)):
            idx = np.random.choice(
                len(self.dataset[0]), self.size_shadow_training_set, replace=False
            )
            smaller_data_set = (self.dataset[0][idx], self.dataset[1][idx])
            shadow_training_set = data_utils.create_new_dataset_with_adaptation(
                smaller_data_set, ratio, self.adaptation, **self.kwargs
            )

            training_sets.append(shadow_training_set)

        return training_sets

    def create_shadow_classifier_from_training_set(self, ratio: int) -> list:
        """
        Creates and trains shadow classifiers from shadow training sets with specific ratio (= for one subattack).
        :param ratio: distribution of property for shadow training sets
        :return: list of shadow classifiers, accuracies for the classifiers
        """
        # create training sets
        shadow_training_sets = self.create_shadow_training_sets(ratio)

        num_elements_per_classes = dict(
            zip(*np.unique(self.dataset[1], return_counts=True))
        )
        # create classifiers with trained models based on given data set
        shadow_classifiers = self.train_shadow_classifiers(
            shadow_training_sets, num_elements_per_classes
        )
        return shadow_classifiers

    def output_attack(self, predictions_ratios) -> UserOutputPropertyInferenceAttack:
        """
        Determination of prediction with highest probability.
        :param predictions_ratios: Prediction values from meta-classifier for different subattacks (different properties)
        :type predictions_ratios: OrderedDict[float, np.ndarray]
        :return: Output message for the attack
        """

        # get key & value of ratio with highest property probability
        max_property = max(predictions_ratios.items(), key=lambda item: item[1][0][0])

        output = dict()
        # rounding because calculation creates values like 0.499999999 when we expected 0.5
        for ratio in predictions_ratios:
            key = "{} of the samples are adapted with type {}".format(
                round(ratio, 5), self.adaptation
            )
            output[key] = predictions_ratios[ratio][0][0]

        if len(self.ratios_for_attack) >= 2:
            max_message = "The most probable property with a probability of {} has {} samples adapted with type {}" " and {} samples unmodified.".format(
                predictions_ratios[max_property[0]][0][0],
                round(max_property[0], 5),
                self.adaptation,
                round(1 - max_property[0], 5),
            )
        else:
            if list(predictions_ratios.values())[0][0][0] > 0.5:
                max_message = "The given distribution is more likely than a balanced distribution. " "The given distribution has {} samples adapted with type {} and {} unmodified samples.".format(
                    round(self.ratios_for_attack[0], 5),
                    self.adaptation,
                    round(1 - self.ratios_for_attack[0], 5),
                )
            else:
                max_message = "A balanced distribution is more likely than the given distribution. " "The given distribution has {} samples adapted with type {} and {} unmodified samples.".format(
                    round(self.ratios_for_attack[0], 5),
                    self.adaptation,
                    round(1 - self.ratios_for_attack[0], 5),
                )
            if abs(list(predictions_ratios.values())[0][0][0] - 0.5) <= 0.05:
                self.logger.warning(
                    "The probabilities are very close to each other. The prediction is likely to be a random guess."
                )

        return UserOutputPropertyInferenceAttack(max_message, output)

    def prediction_on_specific_property(
        self,
        feature_extraction_target_model: np.ndarray,
        shadow_classifiers_neg_property: list,
        ratio: float,
    ) -> np.ndarray:
        """
        Perform prediction for a subattack (specific property)
        :param feature_extraction_target_model: extracted features of target model
        :param shadow_classifiers_neg_property: balanced shadow classifiers negation property
        :param ratio: distribution for the property
        :return: Prediction of meta-classifier for property and negation property
        """

        # create shadow classifiers with trained models with unbalanced data set
        shadow_classifiers_property = self.create_shadow_classifier_from_training_set(
            ratio
        )

        # create meta training set
        meta_features, meta_labels = self.create_meta_training_set(
            shadow_classifiers_property, shadow_classifiers_neg_property
        )

        # create meta classifier
        meta_classifier = self.train_meta_classifier(meta_features, meta_labels)

        # get prediction
        prediction = self.perform_prediction(
            meta_classifier, feature_extraction_target_model
        )

        return prediction

    def attack(self) -> UserOutputPropertyInferenceAttack:
        """
        Perform Property Inference attack.
        :return: message with most probable property, dictionary with all properties
        """
        self.logger.info("Initiating Property Inference Attack ... ")
        self.logger.info("Extracting features from target model ... ")
        # extract features of target model
        feature_extraction_target_model = self.feature_extraction(self.target_model)

        self.logger.info(
            "{} --- features extracted from the target model.".format(
                feature_extraction_target_model.shape
            )
        )

        self.logger.info(
            "Creating set of {} balanced shadow classifier(s) ... ".format(
                int(self.amount_sets / 2)
            )
        )

        # create shadow classifiers negation property
        shadow_classifiers_neg_property = (
            self.create_shadow_classifier_from_training_set(self.negative_ratio)
        )

        self.ratios_for_attack.sort()
        predictions = OrderedDict.fromkeys(self.ratios_for_attack, 0)
        # iterate over unbalanced ratios in 0.05 steps (0.05-0.45, 0.55-0.95)
        # (e.g. 0.55 means: class 0: 0.45 of all samples, class 1: 0.55 of all samples)

        self.logger.info(
            f"Performing PIA for the following ratios: {self.ratios_for_attack}."
        )

        for ratio in tqdm(
            self.ratios_for_attack,
            disable=(self.logger.level > logging.INFO),
            desc=f"Performing {len(self.ratios_for_attack)} sub-attack(s)",
        ):
            self.logger.info(f"Sub-attack for ratio {ratio} ... ")
            predictions[ratio] = self.prediction_on_specific_property(
                feature_extraction_target_model, shadow_classifiers_neg_property, ratio
            )
        self.logger.info("PIA completed!")
        return self.output_attack(predictions)
