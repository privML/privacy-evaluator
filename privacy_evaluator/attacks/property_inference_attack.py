from sklearn.base import clone
from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.classifiers.classifier import Classifier
import privacy_evaluator.utils.data_utils as data_utils
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.torch.fc_neural_net import FCNeuralNet

import math
import numpy as np
import torch
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import SklearnClassifier
from typing import Tuple, Any, Dict, List


class PropertyInferenceAttack(Attack):
    def __init__(self, target_model: Classifier, dataset: Tuple[np.ndarray,np.ndarray]):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        :param dataset: dataset for training of shadow classifiers, test_data from dataset with concatenation [test_features, test_labels]
        """
        self.dataset=dataset
        # count of shadow training sets, must be eval
        self.amount_sets = 6
        self.input_shape =  self.dataset[0].shape #[32, 32, 3]
        super().__init__(target_model, None, None, None, None)

    def create_shadow_training_set(
        self,
        num_elements_per_class: Dict[int, int],
    ) ->List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create the shadow training sets with given ratio.
        The function works for the specific binary case that the ratio is a fixed distribution specified in the input.
        :param num_elements_per_classes: number of elements per class
        :return: shadow training sets for given ratio
        """

        training_sets = []
        #neg_property_training_sets = []

        """negation_num_elements_per_class = {
            class_id: int(
                round(
                    sum(property_num_elements_per_class.values())
                    / len(property_num_elements_per_class)
                )
            )
            for class_id in property_num_elements_per_class.keys()
        }"""

        # Creation of shadow training sets with the size dictionaries
        for i in range(int(self.amount_sets / 2)):
            shadow_training_sets = data_utils.new_dataset_from_size_dict(
                self.dataset, num_elements_per_class
            )
            training_sets.append(shadow_training_sets)

        return training_sets

    def train_shadow_classifiers(
        self,
        shadow_training_sets: List[Tuple[np.ndarray, np.ndarray]],
        num_elements_per_classes: Dict[int, int]
    ):
        """
        Train shadow classifiers with each shadow training set (follows property or negation of property).
        :param shadow_training_sets: datasets fulfilling the a specific ratio to train shadow_classifiers
        :param num_elements_per_classes: specific class distribution
        :return: list of shadow classifiers,
                 accuracies for the classifiers
        :rtype: Tuple[  List[:class:`.art.estimators.estimator.BaseEstimator`],
                        List[float]]
        """

        shadow_classifiers = []
        accuracies =[]

        num_classes = len(num_elements_per_classes)

        for shadow_training_set in shadow_training_sets:
            shadow_training_X, shadow_training_y = shadow_training_set
            train_X, test_X, train_y, test_y = train_test_split(
                shadow_training_X, shadow_training_y, test_size=0.3
            )
            train_set = (train_X, train_y)
            test_set = (test_X, test_y)

            model = FCNeuralNet()
            accuracy = trainer(
                train_set, test_set, num_elements_per_classes, model
            )

            # change pytorch classifier to art classifier
            art_model = Classifier._to_art_classifier(
                model, num_classes, self.input_shape
            )

            shadow_classifiers.append(art_model)
            accuracies.append(accuracy)

        return (
            shadow_classifiers,
            accuracies
        )

    def create_shadow_classifier_from_training_set(self, num_elements_per_classes:  Dict[int, int]
                                )-> Tuple[list,list]:
        
        #create training sets
        
        training_sets = self.create_shadow_training_set(
            num_elements_per_classes
        )

        #create classifiers with trained models based on given data set
        (
            shadow_classifiers,
            accuracy
        ) = self.train_shadow_classifiers(
            training_sets,
            num_elements_per_classes,
        )
        return (shadow_classifiers,accuracy)


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

    def prediction_on_specific_property(self, 
            feature_extraction_target_model: np.ndarray,
            shadow_classifiers_neg_property: list,
            ratio: float,
            size_set: int
            ):

        #property of given ratio, only on class 0 and 1 at the moment
        property_num_elements_per_classes = {0: int((1-ratio) * size_set), 1: int(ratio * size_set)}

        # create shadow classifiers with trained models with unbalanced data set
        shadow_classifiers_property,accuracy_property = self.create_shadow_classifier_from_training_set(property_num_elements_per_classes)

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
        :return: prediction about property of target data set [[1, 0]]-> property; [[0, 1]]-> negation property
        :rtype: np.ndarray with shape (1, 2)
        """

        # extract features of target model
        feature_extraction_target_model = self.feature_extraction(self.target_model)

        # set ratio and size for unbalanced data sets
        size_set = 1000 #TODO get size of on class of dataset

        #balanced ratio
        num_elements = int(round(size_set / 2))
        neg_property_num_elements_per_classes={0: num_elements, 1: num_elements}

        #create balanced shadow classifiers (negation property)
        shadow_classifiers_neg_property, accuracy_neg_property = self.create_shadow_classifier_from_training_set(neg_property_num_elements_per_classes)
      
        predictions = dict()
        #iterate over ratios from 0.55 to 0.95 (means: class 0: 0.45 of all samples, class 1: 0.55 of all samples)
        #TODO add more
        for ratio in np.arange(0.55, 1, 0.05):#range(0.55, 0.95, 0.05):

            predictions[round(ratio, 5)] = self.prediction_on_specific_property(feature_extraction_target_model, shadow_classifiers_neg_property,ratio, size_set)

            predictions[round((1-ratio),5)] = self.prediction_on_specific_property(feature_extraction_target_model, shadow_classifiers_neg_property,(1-ratio), size_set)
            
        return predictions
