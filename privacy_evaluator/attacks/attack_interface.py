import abc


class Attack_Interface():

    """Initilizes the Attack Class.
    :param model: the model to be attacked.
    :type model: ART-Classifier
    :param training_data: training data of the model.
    :type test_accuracy: TODO: What Type?
    :param test_data: test data of the model.
    :type test_data: TODO: What Type?
    """

    def __init__(self, model, training_data, test_data):
        self.model = model
        self.training_data = training_data
        self.test_data = test_data

    """ Performs the actual attack on the model
    :param prams: The prarameters of the Attack
    :type params: dict
    return: result of the attack
    rtype: TODO: what Type?
    """

    def perform_attack(params):
        raise NotImplementedError


