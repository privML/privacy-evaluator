from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack


class PropertyInferenceAttackSkeleton(PropertyInferenceAttack):
    def __init__(self, model):
        """
        Initialize the Property Inference Attack Class.
        :param model: the target model to be attacked
        """
        # TODO: create shadow training sets with method below
        property_shadow_training_sets = None
        negation_property_shadow_training_set = None

        super().__init__(
            model, property_shadow_training_sets, negation_property_shadow_training_set
        )

        shadow_training_set = None
        self.shadow_training_set = shadow_training_set

    def create_shadow_training_set(
        self, dataset, property_p, negation_property_p, follows_property
    ):
        """
        Create shadow training set that either follows the property or the negation of the property.
        :param dataset: path to dataset, dataset similar to target dataset
        :param property_p: property p that is analysed in attack
        :param negation_property_p: negation of property p that is analysed in attack
        :param follows_property: whether property is followed or not (negation of property is followed in this case)
        :return: A shadow training set
        """
        raise NotImplementedError

    def train_shadow_classifier(self, shadow_training_set):
        """
        Train shadow classifier with a shadow training set (follows property or negation of property).
        :param shadow_training_set: dataset used for shadow_classifier
        :return: A shadow classifier
        """
        raise NotImplementedError

    def feature_extraction(self, model):
        """
        Extract the features of a given model.
        :param model: a model from which the features should be extracted
        :return: feature extraction
        """
        raise NotImplementedError

    def create_meta_training_set(self, feature_extraction_list):
        """
        Create meta training set out of the feature extraction of the shadow classifiers.
        :param feature_extraction_list: list of all feature extractions of all shadow classifiers
        :return: Meta-training set
        """
        raise NotImplementedError

    def train_meta_classifier(self, meta_training_set):
        """
        Train meta-classifier with the meta-training set.
        :param meta_training_set: Set of feature representation of each shadow classifier,
        labeled according to whether property or negotiation of property is fulfilled.
        :return: Meta classifier
        """
        raise NotImplementedError

    def perform_prediction(self, meta_classifier, feature_extraction_target_model):
        """
        "Actual" attack: Meta classifier gets feature extraction of target model as input, outputs property prediction.
        :param meta_classifier: A classifier
        :param feature_extraction_target_model:
        :return: Prediction whether property or negation of property is fulfilled for target data set
        """
        raise NotImplementedError

    def perform_attack(self, params):
        """
        Perform Property Inference attack.
        :param params: Example data to run through target model for feature extraction
        :return: prediction about property of target data set
        """
        shadow_classifier = self.train_shadow_classifier(self.shadow_training_set)
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
