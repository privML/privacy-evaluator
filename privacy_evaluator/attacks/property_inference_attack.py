from privacy_evaluator.attacks.attack_interface import Attack_Interface

class Property_Inference_Attack(Attack_Interface):
    def __init__(self, model, training_data, test_data):
        """
        :param model: the model to be attacked
        :param training_data:
        :param test_data:
        """

        super().__init__(model, training_data, test_data)

    def create_shadow_training_set(self, dataset, property, follows_property):
        """
        :param dataset: path to dataset, dataset similar to target dataset
        :param property: property of dataset that is analysed
        :param follows_property: whether Property is followed or not (binary)
        :return:
        """
        raise NotImplementedError
        #return shadow_training_set

    def train_shadow_classifier(self, shadow_training_set):
        """
        :param shadow_training_set: dataset used for shadow_classifier
        :return:
        """
        raise NotImplementedError
        #return shadow_classifier

    def feature_extraction(self, model):
        """
        #TODO: create classifier first?
        :param model:
        :return:
        """
        raise NotImplementedError
        #return feature_extraction

    def create_meta_training_set(self, feature_extraction_list):
        """

        :param feature_extraction_list: list of all feature extractions of all shadow classifiers
        :return:
        """
        raise NotImplementedError
        #return meta_training_set

    def train_meta_classifier(self, meta_training_set):
        """
        :param meta_training_set:
        :return:
        """
        raise NotImplementedError
        #return meta_classifier

    def perform_prediction(self, meta_classifier, feature_extraction_target_model):
        """
        :param meta_classifier:
        :param feature_extraction_target_model:
        :return:
        """
        raise NotImplementedError
        #return prediction