from unicodedata import decimal
from ..attacks.attack import Attack
from ..classifiers.classifier import Classifier
from ..utils import data_utils
from ..utils.trainer import trainer
from ..models.tf.conv_net_meta_classifier import ConvNetMetaClassifier
from ..utils.model_utils import copy_and_reset_model
from ..output.user_output_property_inference_attack import (
    UserOutputPropertyInferenceAttack,
)

import numpy as np
import torch
import tensorflow as tf
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.auto import tqdm
from typing import Tuple, Dict, List, Union
from art.estimators.classification import TensorFlowV2Classifier, PyTorchClassifier
from collections import OrderedDict


# count of shadow training sets, must be even
AMOUNT_SETS = 2
# ratio and size for unbalanced data sets
SIZE_SET = 1000
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
# classes the attack should be performed on
CLASSES = [0, 1]

# number of epochs for training the meta classifier
NUM_EPOCHS_META_CLASSIFIER = 20


class PropertyInferenceAttack(Attack):
    def __init__(
        self,
        target_model: Classifier,
        dataset: Tuple[np.ndarray, np.ndarray],
        amount_sets: int = AMOUNT_SETS,
        size_set: int = SIZE_SET,
        ratios_for_attack: List[int] = RATIOS_FOR_ATTACK,
        classes: List[int] = CLASSES,
        verbose: int = 0,
        num_epochs_meta_classifier: int = NUM_EPOCHS_META_CLASSIFIER,
    ):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        :param dataset: dataset for training of shadow classifiers, test_data from dataset
        :param amount_sets: count of shadow training sets, must be even
        :param size_set: ratio and size for unbalanced data sets
        :param ratios_for_attack: ratios for different properties in sub-attacks
        with concatenation [test_features, test_labels]
        :param classes: classes the attack should be performed on
        :param verbose: 0: no information; 1: backbone (most important) information; 2: utterly detailed information will be printed
        :param num_epochs_meta_classifier: number of epochs for training the meta classifier
        """
        logging.info("Property Inference Attack initialization.")
        self.logger = logging.getLogger(__name__)
        if verbose == 2:
            level = logging.DEBUG
        elif verbose == 1:
            level = logging.INFO
        else:
            level = logging.WARNING

        self.logger.setLevel(level)

        if not (
            isinstance(dataset, tuple)
            and list(map(type, dataset)) == [np.ndarray, np.ndarray]
        ):
            raise TypeError("Dataset type should be of shape (np.ndarray, np.ndarray).")

        self.dataset = dataset
        if not (
            isinstance(target_model, TensorFlowV2Classifier)
            or isinstance(target_model, PyTorchClassifier)
        ):
            raise TypeError("Target model must be of type Classifier.")

        # count of shadow training sets, must be even
        self.amount_sets = amount_sets
        if self.amount_sets % 2 != 0 or self.amount_sets < 2:
            raise ValueError(
                "Number of shadow classifiers must be even and greater than 1."
            )
        self.classes = classes
        if len(self.classes) != 2:
            raise ValueError("Currently attack only works with two classes.")
        for class_number in self.classes:
            if class_number not in dataset[1]:
                raise ValueError(f"Class {class_number} does not exist in dataset.")

        self.size_set = size_set
        for i in classes:
            length_class = len((np.where(dataset[1] == i))[0])
            if length_class < size_set:
                size_set_old = size_set
                size_set = length_class
                warning_message = (
                    "Warning: Number of samples for class {} is {}. "
                    "This is smaller than the given size set ({}). "
                    "{} is now the new size set."
                ).format(i, length_class, size_set_old, size_set)
                self.logger.warning(warning_message)
        self.ratios_for_attack = ratios_for_attack

        if num_epochs_meta_classifier < 1:
            raise ValueError(
                "The number of epochs for training the meta classifier must be at least one."
            )
        self.num_epochs_meta_classifier = num_epochs_meta_classifier

        if len(ratios_for_attack) < 1:
            raise ValueError(
                "Ratios for different properties in sub-attacks can't have length zero."
            )

        self.input_shape = self.dataset[0][0].shape  # [32, 32, 3] for CIFAR10

        super().__init__(target_model, None, None, None, None)

    def create_shadow_training_set(
        self,
        num_elements_per_class: Dict[int, int],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create the shadow training sets with given ratio.
        The function works for the specific binary case that the ratio is a fixed distribution
        specified in the input.
        :param num_elements_per_class: number of elements per class
        :return: shadow training sets for given ratio
        """

        training_sets = []

        # Creation of shadow training sets with the size dictionaries
        # amount_sets divided by 2 because amount_sets describes the total amount of shadow training sets.
        # In this function however only all shadow training sets of one type (follow property OR negation of property) are created, hence amount_sets / 2.
        self.logger.info("Creating shadow training sets")

        for _ in range(int(self.amount_sets / 2)):
            shadow_training_sets = data_utils.new_dataset_from_size_dict(
                self.dataset, num_elements_per_class
            )
            training_sets.append(shadow_training_sets)

        return training_sets

    def train_shadow_classifiers(
        self,
        shadow_training_sets: List[Tuple[np.ndarray, np.ndarray]],
        num_elements_per_classes: Dict[int, int],
    ):
        """
        Train shadow classifiers with each shadow training set (follows property or negation of property).
        :param shadow_training_sets: datasets fulfilling the a specific ratio to train shadow_classifiers
        :param num_elements_per_classes: specific class distribution
        :return: list of shadow classifiers,
                 accuracies for the classifiers
        :rtype: Tuple[  List[:class:.art.estimators.estimator.BaseEstimator]
        """

        shadow_classifiers = []

        num_classes = len(num_elements_per_classes)
        self.logger.info(f"Training {len(shadow_training_sets)} shadow classifier(s):")
        with logging_redirect_tqdm():
            for shadow_training_set in tqdm(
                shadow_training_sets,
                disable=(self.logger.level > logging.INFO),
                desc=f"Training {len(shadow_training_sets)} shadow classifier(s)",
            ):
                model = copy_and_reset_model(self.target_model)
                trainer(
                    shadow_training_set,
                    num_elements_per_classes,
                    model,
                    log_level=self.logger.level,
                    desc="shadow classifier",
                )

                # change pytorch classifier to art classifier
                art_model = Classifier._to_art_classifier(
                    model,
                    "sparse_categorical_crossentropy",
                    num_classes,
                    self.input_shape,
                )
                shadow_classifiers.append(art_model)

        return shadow_classifiers

    def create_shadow_classifier_from_training_set(
        self, num_elements_per_classes: Dict[int, int]
    ) -> list:
        # create training sets
        shadow_training_sets = self.create_shadow_training_set(num_elements_per_classes)

        # create classifiers with trained models based on given data set
        shadow_classifiers = self.train_shadow_classifiers(
            shadow_training_sets,
            num_elements_per_classes,
        )
        return shadow_classifiers

    @staticmethod
    def feature_extraction(model):
        """
        Extract the features of a given model.
        :param model: a model from which the features should be extracted
        :type model: :class:`.art.estimators.estimator.BaseEstimator`
            # BaseEstimator is very general and could be specified to art.classifier
        :return: feature extraction
        :rtype: np.ndarray
        """

        # Filter out all trainable parameters (from every layer)
        # This works differently for PyTorch and TensorFlow. Raise TypeError if model is
        # neither of both.
        if isinstance(model.model, torch.nn.Module):
            model_parameters = list(
                filter(lambda p: p.requires_grad, model.model.parameters())
            )
            # Store the remaining parameters in a concatenated 1D numPy-array
            model_parameters = np.concatenate(
                [el.cpu().detach().numpy().flatten() for el in model_parameters]
            ).flatten()
            return model_parameters

        elif isinstance(model.model, tf.keras.Model):
            model_parameters = np.concatenate(
                [el.numpy().flatten() for el in model.model.trainable_variables]
            ).flatten()
            return model_parameters
        else:
            raise TypeError(
                "Expected model to be an instance of {} or {}, received {} instead.".format(
                    str(torch.nn.Module), str(tf.keras.Model), str(type(model.model))
                )
            )

    def create_meta_training_set(
        self, classifier_list_with_property, classifier_list_without_property
    ):
        """
        Create meta training set out of shadow classifiers.
        :param classifier_list_with_property:
            list of all shadow classifiers that were trained on a dataset which fulfills the property
        :type classifier_list_with_property:
            iterable object of :class:`.art.estimators.estimator.BaseEstimator`
        :param classifier_list_without_property:
            list of all shadow classifiers that were trained on a dataset which does NOT fulfill the
            property
        :type classifier_list_without_property:
            iterable object of :class:`.art.estimators.estimator.BaseEstimator`
        :return: tuple (Meta-training set, label set)
        :rtype: tuple (np.ndarray, np.ndarray)
        """
        # Apply self.feature_extraction on each shadow classifier and concatenate all features
        # into one array
        feature_list_with_property = np.array(
            [
                self.feature_extraction(classifier)
                for classifier in classifier_list_with_property
            ]
        )
        feature_list_without_property = np.array(
            [
                self.feature_extraction(classifier)
                for classifier in classifier_list_without_property
            ]
        )
        meta_features = np.concatenate(
            [feature_list_with_property, feature_list_without_property]
        )
        # Create corresponding labels
        meta_labels = np.concatenate(
            [
                np.ones(len(feature_list_with_property), dtype=int),
                np.zeros(len(feature_list_without_property), dtype=int),
            ]
        )

        return meta_features, meta_labels

    def train_meta_classifier(
        self, meta_training_X: np.ndarray, meta_training_y: np.ndarray
    ) -> TensorFlowV2Classifier:
        """
        Train meta-classifier with the meta-training set.
        :param meta_training_X: Set of feature representation of each shadow classifier.
        :param meta_training_y: Set of labels for each shadow classifier,
                                according to whether property is fullfilled (1) or not (0)
        :return: Art Meta classifier
        """
        # reshaping train data to fit models input
        meta_training_X = meta_training_X.reshape(
            (meta_training_X.shape[0], meta_training_X[0].shape[0], 1)
        )
        meta_training_y = meta_training_y.reshape((meta_training_y.shape[0], 1))
        meta_input_shape = meta_training_X[0].shape

        # currently there are just 2 classes
        nb_classes = len(self.classes)

        inputs = tf.keras.Input(shape=meta_input_shape)

        # create model according to model from https://arxiv.org/pdf/2002.05688.pdf
        self.logger.info("Initialize meta-classifier ... ")
        cnmc = ConvNetMetaClassifier(inputs=inputs, num_classes=nb_classes)

        cnmc.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        # keras functional API provides a verbose variable ranging from {0, 1, 2}.
        # logging uses levels in our case corresponding to numeric values from {30, 20, 10}.
        # We can therefore convert our self.logger.level to the appropriate verbose value in the following manner:
        verbose = 3 - int(self.logger.level / 10)

        self.logger.info(
            f"Training meta-classifier for {self.num_epochs_meta_classifier} epochs ... "
        )
        cnmc.model.fit(
            x=meta_training_X,
            y=meta_training_y,
            epochs=self.num_epochs_meta_classifier,
            batch_size=128,
            verbose=verbose
            # If enough shadow classifiers are available, one could split the training set
            # and create an additional validation set as input:
            # validation_data = (validation_X, validation_y),
        )

        # model has .evaluate(test_X,test_y) function
        # convert model to ART classifier
        art_meta_classifier = Classifier._to_art_classifier(
            cnmc.model,
            loss="sparse_categorical_crossentropy",
            nb_classes=nb_classes,
            input_shape=meta_input_shape,
        )

        return art_meta_classifier

    @staticmethod
    def perform_prediction(
        meta_classifier, feature_extraction_target_model
    ) -> np.ndarray:
        """
        "Actual" attack: Meta classifier gets feature extraction of target model as input, outputs
        property prediction.
        :param meta_classifier: A classifier
        :type meta_classifier: "CLASSIFIER_TYPE" (to be found in .art.estimators)
        :param feature_extraction_target_model: extracted features of target model
        :type feature_extraction_target_model: np.ndarray
        :return: Prediction given as probability distribution vector whether property or negation
            of property is fulfilled for target data set
        :rtype: np.ndarray with shape (1, 2)
        """

        feature_extraction_target_model = feature_extraction_target_model.reshape(
            (feature_extraction_target_model.shape[0], 1)
        )

        assert meta_classifier.input_shape == tuple(
            feature_extraction_target_model.shape
        )

        predictions = meta_classifier.predict(x=[feature_extraction_target_model])
        return predictions

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
            key = "class {}: {}, class {}: {}".format(
                self.classes[0], round(1 - ratio, 5), self.classes[1], round(ratio, 5)
            )
            output[key] = predictions_ratios[ratio][0][0]

        if len(self.ratios_for_attack) >= 2:
            max_message = (
                "The most probable property is class {}: {}, "
                "class {}: {} with a probability of {}.".format(
                    self.classes[0],
                    round(1 - max_property[0], 5),
                    self.classes[1],
                    round(max_property[0], 5),
                    predictions_ratios[max_property[0]][0][0],
                )
            )
        else:
            if list(predictions_ratios.values())[0][0][0] > 0.5:
                max_message = "The given distribution is more likely than a balanced distribution. " "The given distribution is class {}: {}, class {}: {}".format(
                    self.classes[0],
                    round(1 - self.ratios_for_attack[0], 5),
                    self.classes[1],
                    round(self.ratios_for_attack[0], 5),
                )
            else:
                max_message = "A balanced distribution is more likely than the given distribution. " "The given distribution is class {}: {}, class {}: {}".format(
                    self.classes[0],
                    round(1 - self.ratios_for_attack[0], 5),
                    self.classes[1],
                    round(self.ratios_for_attack[0], 5),
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

        # property of given ratio, only two classes allowed right now
        property_num_elements_per_classes = {
            self.classes[0]: int((1 - ratio) * self.size_set),
            self.classes[1]: int(ratio * self.size_set),
        }

        # create shadow classifiers with trained models with unbalanced data set
        shadow_classifiers_property = self.create_shadow_classifier_from_training_set(
            property_num_elements_per_classes
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
            ),
        )

        # balanced ratio
        num_elements = int(round(self.size_set / len(self.classes)))
        neg_property_num_elements_per_class = {i: num_elements for i in self.classes}

        self.logger.info(
            "Creating set of {} balanced shadow classifier(s) ... ".format(
                int(self.amount_sets / 2)
            ),
        )
        # create balanced shadow classifiers negation property
        shadow_classifiers_neg_property = (
            self.create_shadow_classifier_from_training_set(
                neg_property_num_elements_per_class
            )
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
                feature_extraction_target_model,
                shadow_classifiers_neg_property,
                ratio,
            )
        self.logger.info("PIA completed!")
        return self.output_attack(predictions)
