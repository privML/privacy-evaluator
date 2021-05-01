from privacy_evaluator.attacks.attack_interface import Attack_Interface


class Sample_Attack(Attack_Interface):
    def __init__(self, model, training_data, test_data):
        super().__init__(model, training_data, test_data)

    def perform_attack(params):
        pass


