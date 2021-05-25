import torch
import torchvision

from property_inference_attack_skeleton import PropertyInferenceAttackSkeleton
import data

amount_sets = 6
size_set = 1500
class_ids = [0,1]
property_num_elements_per_classes = {0: 500, 1: 1000}
input_shape = [32,32,3]


if __name__ == "__main__":
    print("started")
    train_dataset, test_dataset = data.dataset_downloader()
    print("loaded")
    attack = PropertyInferenceAttackSkeleton(None)
    property_training_sets, neg_property_training_sets, property_num_elements_per_classes, neg_property_num_elements_per_classes = attack.create_shadow_training_set(test_dataset, amount_sets, size_set, property_num_elements_per_classes)
    print("created shadow training sets")
    shadow_classifiers_property, shadow_classifiers_neg_property, accuracy_prop, accuracy_neg = attack.train_shadow_classifiers(property_training_sets, neg_property_training_sets,property_num_elements_per_classes, neg_property_num_elements_per_classes, input_shape)
    print("trained shadow classifiers")
    print("Accuracies for classifiers trained on data fulfilling the property:")
    print(accuracy_prop)
    print("Accuracies for classifiers trained on data not fulfilling the property:")
    print(accuracy_neg)
