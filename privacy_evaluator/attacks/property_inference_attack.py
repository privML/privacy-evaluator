from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.models.train_cifar10_torch import data, train

import math
import numpy as np
import torch
import tensorflow as tf
from sklearn.svm import SVC
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from typing import Tuple, Any, Dict, List


class PropertyInferenceAttack(Attack):
    def __init__(self, target_model: Classifier):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        """

        super().__init__(target_model, None, None, None, None)

    def create_shadow_training_set(
        self,
        dataset: torch.utils.data.Dataset,
        amount_sets: int,
        size_set: int,
        property_num_elements_per_classes: Dict[int, int],
    ) -> Tuple[
        List[torch.utils.data.Dataset],
        List[torch.utils.data.Dataset],
        Dict[int, int],
        Dict[int, int],
    ]:
        """
        Create the shadow training sets, half fulfill the property, half fulfill the negation of the property.
        The function works for the specific binary case that the property is a fixed distribution specified in the input
        and the negation of the property is a balanced distribution.
        :param dataset: Dataset out of which shadow training sets should be created
        :param amount_sets: how many shadow training sets should be created
        :param size_set: size of one shadow training set for one shadow classifier
        :param property_num_elements_per_classes: number of elements per class, this is the property
        :return: shadow training sets for property,
                 shadow training sets for negation,
                 dictionary holding the unbalanced class distribution (=property),
                 dictionary holding the balanced class distribution (=negation of property)
        """

        amount_property = int(round(amount_sets / 2))

        property_training_sets = []
        neg_property_training_sets = []

        # PROPERTY
        # according to property_num_elements_per_classes we select the classes and take random elements out of the dataset
        # and create the shadow training sets with these elements"""
        for i in range(amount_property):
            shadow_training_set = []
            for class_id, num_elements in property_num_elements_per_classes.items():
                subset = data.subset(dataset, class_id, num_elements)
                shadow_training_set.append(subset)
            shadow_training_set = torch.utils.data.ConcatDataset(shadow_training_set)
            property_training_sets.append(shadow_training_set)

        # NEG_PROPERTY (BALANCED)
        # create balanced shadow training sets with the classes specified in property_num_elements_per_classes
        num_elements = int(round(size_set / len(property_num_elements_per_classes)))
        for i in range(amount_property):
            shadow_training_set = []
            for class_id, _ in property_num_elements_per_classes.items():
                subset = data.subset(dataset, class_id, num_elements)
                shadow_training_set.append(subset)
            shadow_training_set = torch.utils.data.ConcatDataset(shadow_training_set)
            neg_property_training_sets.append(shadow_training_set)

        # create neg_property_num_elements_per_classes, later needed in train_shadow_classifier
        neg_property_num_elements_per_classes = {
            class_id: num_elements
            for class_id in property_num_elements_per_classes.keys()
        }

        return (
            property_training_sets,
            neg_property_training_sets,
            property_num_elements_per_classes,
            neg_property_num_elements_per_classes,
        )

    def train_shadow_classifiers(
        self,
        property_training_sets: List[torch.utils.data.Dataset],
        neg_property_training_sets: List[torch.utils.data.Dataset],
        property_num_elements_per_classes: Dict[int, int],
        neg_property_num_elements_per_classes: Dict[int, int],
        input_shape: Tuple[int, ...],
    ):
        """
        Train shadow classifiers with each shadow training set (follows property or negation of property).
        :param shadow_training_sets_property: datasets fulfilling the property to train 50 % of shadow_classifiers
        :param shadow_training_sets_neg_property: datasets not fulfilling the property to train 50 % of shadow_classifiers
        :param property_num_elements_per_classes: unbalanced class distribution (= property)
        :param neg_property_num_elements_per_classes: balanced class distribution (= negation of property)
        :param input_shape: Input shape of a data point for the classifier. Needed in _to_art_classifier.
        :return: list of shadow classifiers for the property,
                 list of shadow classifiers for the negation of the property,
                 accuracies for the property shadow classifiers,
                 accuracies for the negation of the property classifiers
        :rtype: Tuple[  List[:class:`.art.estimators.estimator.BaseEstimator`],
                        List[:class:`.art.estimators.estimator.BaseEstimator`],
                        List[float],
                        List[float]]
        """

        shadow_classifiers_property = []
        shadow_classifiers_neg_property = []
        accuracy_prop = []
        accuracy_neg = []

        num_classes = len(property_num_elements_per_classes)

        for shadow_training_set in property_training_sets:
            len_train_set = math.ceil(len(shadow_training_set) * 0.7)
            len_test_set = math.floor(len(shadow_training_set) * 0.3)

            train_set, test_set = torch.utils.data.random_split(
                shadow_training_set, [len_train_set, len_test_set]
            )
            accuracy, model_property = train.trainer_out_model(
                train_set, test_set, property_num_elements_per_classes, "FCNeuralNet"
            )

            # change pytorch classifier to art classifier
            art_model_property = Classifier._to_art_classifier(
                model_property, num_classes, input_shape
            )

            shadow_classifiers_property.append(art_model_property)
            accuracy_prop.append(accuracy)

        for shadow_training_set in neg_property_training_sets:
            len_train_set = math.ceil(len(shadow_training_set) * 0.7)
            len_test_set = math.floor(len(shadow_training_set) * 0.3)

            train_set, test_set = torch.utils.data.random_split(
                shadow_training_set, [len_train_set, len_test_set]
            )
            accuracy, model_neg_property = train.trainer_out_model(
                train_set,
                test_set,
                neg_property_num_elements_per_classes,
                "FCNeuralNet",
            )

            # change pytorch classifier to art classifier
            art_model_neg_property = Classifier._to_art_classifier(
                model_neg_property, num_classes, input_shape
            )

            shadow_classifiers_neg_property.append(art_model_neg_property)
            accuracy_neg.append(accuracy)

        return (
            shadow_classifiers_property,
            shadow_classifiers_neg_property,
            accuracy_prop,
            accuracy_neg,
        )

    def feature_extraction(self, model):
        """
        Extract the features of a given model.
        :param model: a model from which the features should be extracted
        :type model: :class:`.art.estimators.estimator.BaseEstimator` # BaseEstimator is very general and could be specified to art.classifier
        :return: feature extraction
        :rtype: np.ndarray
        """

        # Filter out all trainable parameters (from every layer)
        # This works differently for PyTorch and TensorFlow. Raise TypeError if model is neither of both.
        if isinstance(model.model, torch.nn.Module):
            model_parameters = list(
                filter(lambda p: p.requires_grad, model.model.parameters())
            )
            # Store the remaining parameters in a concatenated 1D numPy-array
            model_parameters = np.concatenate(
                [el.detach().numpy().flatten() for el in model_parameters]
            ).flatten()
            return model_parameters

        elif isinstance(model.model, tf.keras.Model):
            model_parameters = np.concatenate(
                [el.numpy().flatten() for el in model.model.trainable_variables]
            ).flatten()
            return model_parameters
        else:
            raise TypeError(
                f"Expected model to be an instance of {str(torch.nn.Module)} or {str(tf.keras.Model)}, received {str(type(model.model))} instead."
            )

    def create_meta_training_set(
        self, classifier_list_with_property, classifier_list_without_property
    ):
        """
        Create meta training set out of shadow classifiers.
        :param classifier_list_with_property: list of all shadow classifiers that were trained on a dataset which fulfills the property
        :type classifier_list_with_property: iterable object of :class:`.art.estimators.estimator.BaseEstimator`
        :param classifier_list_without_property: list of all shadow classifiers that were trained on a dataset which does NOT fulfill the property
        :type classifier_list_without_property: iterable object of :class:`.art.estimators.estimator.BaseEstimator`
        :return: tupel (Meta-training set, label set)
        :rtype: tupel (np.ndarray, np.ndarray)
        """
        # Apply self.feature_extraction on each shadow classifier and concatenate all features into one array
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
        # meta_labels = np.concatenate([np.ones(len(feature_list_with_property)), np.zeros(len(feature_list_without_property))])
        # For scikit-learn SVM classifier we need one hot encoded labels, therefore:
        meta_labels = np.concatenate(
            [
                np.array([[1, 0]] * len(feature_list_with_property)),
                np.array([[0, 1]] * len(feature_list_without_property)),
            ]
        )
        return meta_features, meta_labels

    def train_meta_classifier(self, meta_training_X, meta_training_y):
        """
        Train meta-classifier with the meta-training set.
        :param meta_training_X: Set of feature representation of each shadow classifier.
        :type meta_training_X: np.ndarray
        :param meta_training_y: Set of (one-hot-encoded) labels for each shadow classifier,
                                according to whether property is fullfilled ([1, 0]) or not ([0, 1]).
        :type meta_training_y: np.ndarray
        :return: Meta classifier
        :rtype: "CLASSIFIER_TYPE" (to be found in `.art.utils`) # classifier.predict is an one-hot-encoded label vector:
                                                    [1, 0] means target model has the property, [0, 1] means it does not.
        """
        # Create a scikit SVM model, which will be trained on meta_training
        model = SVC(C=1.0, kernel="rbf")

        # Turn into ART classifier
        classifier = SklearnClassifier(model=model)

        # Train the ART classifier as meta_classifier and return
        classifier.fit(meta_training_X, meta_training_y)
        return classifier

    def perform_prediction(
        self, meta_classifier, feature_extraction_target_model
    ) -> np.ndarray:
        """
        "Actual" attack: Meta classifier gets feature extraction of target model as input, outputs property prediction.
        :param meta_classifier: A classifier
        :type meta_classifier: "CLASSIFIER_TYPE" (to be found in .art.estimators)
        :param feature_extraction_target_model: extracted features of target model
        :type feature_extraction_target_model: np.ndarray
        :return: Prediction given as probability distribution vector whether property or negation of property is
        fulfilled for target data set
        :rtype: np.ndarray with shape (1, 2)
        """
        assert meta_classifier.input_shape == tuple(
            feature_extraction_target_model.shape
        )

        predictions = meta_classifier.predict(x=[feature_extraction_target_model])
        return predictions

    def attack(self):
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: prediction about property of target data set [[1, 0]]-> property; [[0, 1]]-> negation property
        :rtype: np.ndarray with shape (1, 2)
        """
        # load data (CIFAR10)
        train_dataset, test_dataset = data.dataset_downloader()
        input_shape = [32, 32, 3]

        # count of shadow training sets
        amount_sets = 6

        # set ratio and size for unbalanced data sets
        size_set = 1500
        property_num_elements_per_classes = {0: 500, 1: 1000}

        # create shadow training sets. Half unbalanced (property_num_elements_per_classes), half balanced
        (
            property_training_sets,
            neg_property_training_sets,
            property_num_elements_per_classes,
            neg_property_num_elements_per_classes,
        ) = self.create_shadow_training_set(
            test_dataset, amount_sets, size_set, property_num_elements_per_classes
        )

        # create shadow classifiers with trained models, half on unbalanced data set, half with balanced data set
        (
            shadow_classifiers_property,
            shadow_classifiers_neg_property,
            accuracy_prop,
            accuracy_neg,
        ) = self.train_shadow_classifiers(
            property_training_sets,
            neg_property_training_sets,
            property_num_elements_per_classes,
            neg_property_num_elements_per_classes,
            input_shape,
        )

        # create meta training set
        meta_features, meta_labels = self.create_meta_training_set(
            shadow_classifiers_property, shadow_classifiers_neg_property
        )

        # create meta classifier
        meta_classifier = self.train_meta_classifier(meta_features, meta_labels)

        # extract features of target model
        feature_extraction_target_model = self.feature_extraction(self.target_model)

        # get prediction
        prediction = self.perform_prediction(
            meta_classifier, feature_extraction_target_model
        )
        return prediction
