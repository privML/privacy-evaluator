from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
import numpy as np
from typing import Tuple, Any



class PropertyInferenceAttackSkeleton(PropertyInferenceAttack):
    def __init__(
            self,
            model,
            property_shadow_training_sets,
            negation_property_shadow_training_sets,
    ):
        """
        Initialize the Property Inference Attack Class.
        :param model: the target model to be attacked
        :type model: :class:`.art.estimators.estimator.BaseEstimator`
        :param property_shadow_training_sets: the shadow training sets that fulfill property
        :type property_shadow_training_sets: np.ndarray # TODO
        :param negation_property_shadow_training_sets: the shadow training sets that fulfill negation of property
        :type negation_property_shadow_training_sets: np.ndarray # TODO
        """

        super().__init__(
            model, property_shadow_training_sets, negation_property_shadow_training_sets
        )
        # TODO: create shadow_training_set
        shadow_training_set = None
        self.shadow_training_set = shadow_training_set

    def train_shadow_classifiers(self, shadow_training_set):
        """
        Train shadow classifiers with each shadow training set (follows property or negation of property).
        :param shadow_training_set: datasets used for shadow_classifiers
        :type shadow_training_set: np.ndarray # TODO
        :return: shadow classifiers
        :rtype: "CLASSIFIER_TYPE" (to be found in `.art.utils`)
        """
        raise NotImplementedError

    def feature_extraction(self, model):
        """
        Extract the features of a given model.
        :param model: a model from which the features should be extracted
        :type model: :class:`.art.estimators.estimator.BaseEstimator`
        :return: feature extraction
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def create_meta_training_set(self, feature_extraction_list):
        """
        Create meta training set out of the feature extraction of the shadow classifiers.
        :param feature_extraction_list: list of all feature extractions of all shadow classifiers
        :type feature_extraction_list: np.ndarray
        :return: Meta-training set
        :rtype: np.ndarray
        """
        raise NotImplementedError

    def train_meta_classifier(self, meta_training_set):
        """
        Train meta-classifier with the meta-training set.
        :param meta_training_set: Set of feature representation of each shadow classifier,
        labeled according to whether property or negotiation of property is fulfilled.
        :type meta_training_set: np.ndarray
        :return: Meta classifier
        :rtype: "CLASSIFIER_TYPE" (to be found in `.art.utils`) # TODO only binary classifiers - special classifier?
        """
        raise NotImplementedError

    def perform_prediction(self, meta_classifier, feature_extraction_target_model):
        """
        "Actual" attack: Meta classifier gets feature extraction of target model as input, outputs property prediction.
        :param meta_classifier: A classifier
        :type meta_classifier: "CLASSIFIER_TYPE" (to be found in `.art.utils`)
        # TODO only binary classifiers-special classifier?
        :param feature_extraction_target_model: extracted features of target model
        :type feature_extraction_target_model: np.ndarray
        :return: Prediction whether property or negation of property is fulfilled for target data set
        :rtype: # TODO
        """
        raise NotImplementedError

    def perform_attack(self, params):
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: prediction about property of target data set
        :rtype: # TODO
        """
        shadow_classifier = self.train_shadow_classifiers(self.shadow_training_set)
        # TODO: create feature extraction for all shadow classifiers
        feature_extraction_list = None
        meta_training_set = self.create_meta_training_set(feature_extraction_list)
        meta_classifier = self.train_meta_classifier(meta_training_set)
        # TODO: create feature extraction for target model, using x
        feature_extraction_target_model = None
        prediction = self.perform_prediction(
            meta_classifier, feature_extraction_target_model
        )
        return prediction
