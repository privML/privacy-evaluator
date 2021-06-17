from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier
import privacy_evaluator.utils.data_utils as data_utils
from privacy_evaluator.utils.model_utils import copy_and_reset_model
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.tf.conv_net_meta_classifier import ConvNetMetaClassifier
from privacy_evaluator.utils.model_utils import copy_and_reset_model

import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
from art.estimators.classification import TensorFlowV2Classifier
import string

# count of shadow training sets, must be even
AMOUNT_SETS = 2
# ratio and size for unbalanced data sets
SIZE_SET = 1000
# ratios for different properties in sub-attacks
RATIOS_FOR_ATTACK = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

class PropertyInferenceAttack(Attack):
    def __init__(
        self, target_model: Classifier, dataset: Tuple[np.ndarray, np.ndarray], amount_sets: int = AMOUNT_SETS, size_set: int = SIZE_SET, ratios_for_attack: [int] = RATIOS_FOR_ATTACK
    ):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        :param dataset: dataset for training of shadow classifiers, test_data from dataset
        :param amount_sets: count of shadow training sets, must be even
        :param size_set: ratio and size for unbalanced data sets
        :param ratios_for_attack: ratios for different properties in sub-attacks
        with concatenation [test_features, test_labels]
        """
        self.dataset = dataset
        self.amount_sets = amount_sets
        self.size_set = size_set
        self.ratios_for_attack = ratios_for_attack
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

        for shadow_training_set in shadow_training_sets:
            shadow_training_X, shadow_training_y = shadow_training_set
            train_X, test_X, train_y, test_y = train_test_split(
                shadow_training_X, shadow_training_y, test_size=0.3
            )
            train_set = (train_X, train_y)
            test_set = (test_X, test_y)

            model = copy_and_reset_model(self.target_model)
            trainer(train_set, num_elements_per_classes, model)

            # change pytorch classifier to art classifier
            art_model = Classifier._to_art_classifier(
                model, num_classes, self.input_shape
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

    @staticmethod
    def train_meta_classifier(
        meta_training_X: np.ndarray, meta_training_y: np.ndarray
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
        nb_classes = 2

        inputs = tf.keras.Input(shape=meta_input_shape)

        # create model according to model from https://arxiv.org/pdf/2002.05688.pdf
        cnmc = ConvNetMetaClassifier(inputs=inputs, num_classes=nb_classes)

        cnmc.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )

        cnmc.model.fit(
            x=meta_training_X,
            y=meta_training_y,
            epochs=2,
            batch_size=128,
            # If enough shadow classifiers are available, one could split the training set
            # and create an additional validation set as input:
            # validation_data = (validation_X, validation_y),
        )

        # model has .evaluate(test_X,test_y) function
        # convert model to ART classifier
        art_meta_classifier = Classifier._to_art_classifier(
            cnmc.model, nb_classes=nb_classes, input_shape=meta_input_shape
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

    @staticmethod
    def output_attack(predictions_ratios: Dict[float, np.ndarray]) -> string:
        """
        Determination of prediction with highest probability.
        :param predictions_ratios: Prediction values from meta-classifier for different subattacks (different properties)
        :return: Output message for the attack
        """

        # get key & value of ratio with highest property probability
        max_property = max(predictions_ratios.items(), key=lambda item: item[1][0][0])
        # get average of neg property probabilities of 0.05, 0.95 (most unbalanced datasets --> highest probability for correctness of neg probability)
        average_unbalanced_cases_neg_property = (
            predictions_ratios[0.95][0][1] + predictions_ratios[0.05][0][1]
        ) / 2

        if max_property[1][0][0] > average_unbalanced_cases_neg_property:
            return "The property inference attack predicts that the target model is unbalanced with a ratio of {}.".format(
                max_property[0]
            )
        elif max_property[1][0][0] < average_unbalanced_cases_neg_property:
            return "The property inference attack predicts that the target model is balanced."
        else:
            raise ValueError(
                "Wrong input. Property inference attack cannot predict balanced and unbalanced."
            )

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

        # property of given ratio, only on class 0 and 1 at the moment
        property_num_elements_per_classes = {
            0: int((1 - ratio) * self.size_set),
            1: int(ratio * self.size_set),
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

    def attack(self):
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: prediction about property of target data set
            [[1, 0]]-> property; [[0, 1]]-> negation property
        :rtype: np.ndarray with shape (1, 2)
        """

        # extract features of target model
        feature_extraction_target_model = self.feature_extraction(self.target_model)




        # balanced ratio
        num_elements = int(round(self.size_set / 2))
        neg_property_num_elements_per_class = {0: num_elements, 1: num_elements}

        # create balanced shadow classifiers negation property
        shadow_classifiers_neg_property = (
            self.create_shadow_classifier_from_training_set(
                neg_property_num_elements_per_class
            )
        )

        predictions = {}
        # iterate over unbalanced ratios
        # (e.g. 0.55 means: class 0: 0.45 of all samples, class 1: 0.55 of all samples)

        for ratio in self.ratios_for_attack:
            # goes through ratios 0.55 - 0.95
            predictions[round(ratio, 5)] = self.prediction_on_specific_property(
                feature_extraction_target_model,
                shadow_classifiers_neg_property,
                ratio
            )
        return self.output_attack(predictions)
