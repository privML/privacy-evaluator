from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from tensorflow.keras.datasets import cifar10


amount_sets = 6
property_num_elements_per_classes = {0: 30, 1: 70}
input_shape = [32, 32, 3]


if __name__ == "main":
    print("started")
    train_dataset, test_dataset = cifar10.load_data()
    print("loaded")

    attack = PropertyInferenceAttack(None)
    (
        property_datasets,
        neg_property_datasets,
        property_num_elements_per_classes,
        neg_property_num_elements_per_classes,
    ) = attack.create_shadow_training_set(
        test_dataset,
        amount_sets,
        property_num_elements_per_classes,
    )
    print("created shadow training sets")
    (
        shadow_classifiers_property,
        shadow_classifiers_neg_property,
    ) = attack.train_shadow_classifiers(
        property_datasets,
        neg_property_datasets,
        property_num_elements_per_classes,
        neg_property_num_elements_per_classes,
        input_shape,
    )
