from privacy_evaluator.attacks.attack_interface import Attack


class SampleAttack(Attack):
    def __init__(self, model, training_data, test_data):
        super().__init__(model, training_data, test_data)

    def perform_attack(self, params):
        pass


