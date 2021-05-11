import abc
from privacy_evaluator.attacks.attack_interface import Attack_Interface


class PropertyInferenceAttack(Attack_Interface):
    """
    Abstract class for property inference classes.
    """

    def __init__(self, model, property_shadow_training_sets, negation_property_shadow_training_set):
        """
        Initialize the Property Inference Attack Class.
        :param model: the target model to be attacked
        :param property_shadow_training_sets: the shadow training sets that fulfill property
        :param negation_property_shadow_training_set: the shadow training sets that fulfill negation of property
        """

        super().__init__(model, None, None)
        self.property_shadow_training_sets = property_shadow_training_sets
        self.negation_shadow_training_sets = negation_property_shadow_training_set

    @abc.abstractmethod
    def perform_attack(self, params):
        """
        Performs the property inference attack on target model.
        :param params: Example data to run through target model for feature extraction
        :return: prediction about property of target data set as a result of the property inference attack
        """
        raise NotImplementedError
