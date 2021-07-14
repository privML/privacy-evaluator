from unicodedata import decimal
from ...attacks.attack import Attack
from ...classifiers.classifier import Classifier
from ...utils import data_utils
from ...utils.trainer import trainer
from ...models.tf.conv_net_meta_classifier import ConvNetMetaClassifier
from ...utils.model_utils import copy_and_reset_model
from ...output.user_output_property_inference_attack import (
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


class PropertyInferenceAttack(Attack):
    def __init__(
        self,
        target_model: Classifier,
        dataset: Tuple[np.ndarray, np.ndarray],
        amount_sets: int,
        size_shadow_training_set: int,
        ratios_for_attack: List[int],
        num_epochs_meta_classifier: int,
        verbose: int,
    ):
        """
        Initialize the Property Inference Attack Class.
        :param target_model: the target model to be attacked
        :param dataset: dataset for training of shadow classifiers, test_data from dataset
        :param amount_sets: count of shadow training sets, must be even
        :param size_shadow_training_set: ratio and size for unbalanced data sets
        :param ratios_for_attack: ratios for different properties in sub-attacks
        with concatenation [test_features, test_labels]
        :param num_epochs_meta_classifier: number of epochs for training the meta classifier
        :param verbose: 0: no information; 1: backbone (most important) information; 2: utterly detailed information will be printed
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

        self.size_shadow_training_set = size_shadow_training_set
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
        self,
        meta_training_X: np.ndarray,
        meta_training_y: np.ndarray,
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

        nb_classes = len(np.unique(meta_training_y))

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
