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
from typing import Tuple, Dict, List, Union
from art.estimators.classification import TensorFlowV2Classifier,PyTorchClassifier
from collections import OrderedDict


class PropertyInferenceAttack(Attack):
    def __init__(
        self, target_model: Classifier, dataset: Tuple[np.ndarray, np.ndarray]
    ):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        :param dataset: dataset for training of shadow classifiers, test_data from dataset
        with concatenation [test_features, test_labels]
        """

        if not (isinstance(dataset, tuple) and list(map(type, dataset)) == [np.ndarray, np.ndarray]):
            raise TypeError("Dataset type should be of shape (np.ndarray, np.ndarray).")
            
        self.dataset = dataset

        if not(isinstance(target_model, TensorFlowV2Classifier) or isinstance(target_model,PyTorchClassifier)):
            raise TypeError("Target model must be of type Classifier.")

        # count of shadow training sets, must be even
        self.amount_sets = 2
        if self.amount_sets % 2 != 0 or self.amount_sets < 2:
            raise ValueError("Number of shadow classifiers must be even and greater than 1.")

        self.input_shape = self.dataset[0][0].shape  # [32, 32, 3] for CIFAR10
        self.classes = [0,1]
        if len(self.classes) != 2:
            raise ValueError("Currently attack only works with two classes.")
        for class_number in self.classes:
            if class_number not in dataset[1]:
                raise ValueError(f"Class {class_number} does not exist in dataset.")

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


    def train_meta_classifier(self,
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
        nb_classes = len(self.classes)

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


    def output_attack(self, predictions_ratios) -> Tuple[str,Dict[str, float]]:
        """
        Determination of prediction with highest probability.
        :param predictions_ratios: Prediction values from meta-classifier for different subattacks (different properties) 
        :type predictions_ratios: OrderedDict[float, np.ndarray]
        :return: Output message for the attack
        """

        # get key & value of ratio with highest property probability
        max_property = max(predictions_ratios.items(), key=lambda item: item[1][0][0])

        output = dict()
        #rounding because calculation creates values like 0.499999999 when we expected 0.5
        for ratio in predictions_ratios:
            output[f"class {self.classes[0]}: {round(1-ratio,5)}, class {self.classes[1]}: {round(ratio,5)}"] = predictions_ratios[ratio][0][0]

        max_message = f"The most probable property is class {self.classes[0]}: {round(1-max_property[0],5)}, class {self.classes[1]}: {round(max_property[0],5)} with a probability of {predictions_ratios[max_property[0]][0][0]}."

        return (max_message,output)

    def prediction_on_specific_property(
        self,
        feature_extraction_target_model: np.ndarray,
        shadow_classifiers_neg_property: list,
        ratio: float,
        size_set: int,
    ) -> np.ndarray:
        """
        Perform prediction for a subattack (specific property)
        :param feature_extraction_target_model: extracted features of target model
        :param shadow_classifiers_neg_property: balanced shadow classifiers negation property
        :param ratio: distribution for the property
        :param size_set: size of one class from data set
        :return: Prediction of meta-classifier for property and negation property
        """

        # property of given ratio, only two classes allowed right now
        property_num_elements_per_classes = {
            self.classes[0]: int((1 - ratio) * size_set),
            self.classes[1]: int(ratio * size_set),
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

    def attack(self)-> Tuple[str,Dict[str, float]]:
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: message with most probable property, dictionary with all properties
        """

        # extract features of target model
        feature_extraction_target_model = self.feature_extraction(self.target_model)

        # set ratio and size for unbalanced data sets
        size_set = 1000

        # balanced ratio
        num_elements = int(round(size_set / len(self.classes)))
        neg_property_num_elements_per_class = {i:num_elements for i in self.classes}


        # create balanced shadow classifiers negation property
        shadow_classifiers_neg_property = (
            self.create_shadow_classifier_from_training_set(
                neg_property_num_elements_per_class
            )
        )

        ratios = np.concatenate([np.arange(0.55, 1, 0.05), np.arange(0.45, 0, -0.05)])
        ratios.sort()
        predictions = OrderedDict.fromkeys(ratios, 0)
        # iterate over unbalanced ratios in 0.05 steps (0.05-0.45, 0.55-0.95)
        # (e.g. 0.55 means: class 0: 0.45 of all samples, class 1: 0.55 of all samples)
        for ratio in ratios:
            predictions[ratio] = self.prediction_on_specific_property(
                feature_extraction_target_model,
                shadow_classifiers_neg_property,
                ratio,
                size_set,
            )

        return self.output_attack(predictions)
