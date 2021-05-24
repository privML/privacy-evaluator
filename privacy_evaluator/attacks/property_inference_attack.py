import abc
import numpy as np
from privacy_evaluator.attacks.attack_interface import Attack_Interface


class PropertyInferenceAttack(Attack_Interface):
    """
    Abstract class for property inference classes.
    """

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

        super().__init__(model, None, None)
        self.property_shadow_training_sets = property_shadow_training_sets
        self.negation_shadow_training_sets = negation_property_shadow_training_sets

    @abc.abstractmethod
    def perform_attack(self, params):
        """
        Performs the property inference attack on target model.
        :param params: Example data to run through target model for feature extraction
        :type params: np.ndarray
        :return: prediction about property of target data set as a result of the property inference attack
        :rtype: #TODO
        """
        raise NotImplementedError
