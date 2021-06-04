from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.utils.data_utils import dataset_downloader


amount_sets = 6
property_num_elements_per_classes = {0: 300, 1: 700}
input_shape = [32, 32, 3]

if True:
    print("started")
    train_dataset, test_dataset = dataset_downloader("CIFAR10")
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
        accuracy_prop,
        accuracy_neg,
    ) = attack.train_shadow_classifiers(
        property_datasets,
        neg_property_datasets,
        property_num_elements_per_classes,
        neg_property_num_elements_per_classes,
        input_shape,
    )
    print("trained shadow training sets")
    print("Accuracy Property:")
    print(accuracy_prop)
    print("Accuracy Negation:")
    print(accuracy_neg)
