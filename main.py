import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.model_utils import create_and_train_torch_ConvNet_model

from matplotlib import pyplot as plt

train_dataset, test_dataset = dataset_downloader("MNIST")
input_shape = test_dataset[0][0].shape
print(f"Input shape of images: {input_shape}")

NUM_ELEMENTS_PER_CLASSES = {0: 1000, 1: 500}
train_set = new_dataset_from_size_dict(train_dataset, NUM_ELEMENTS_PER_CLASSES)
print(f"Amount of images per class: {NUM_ELEMENTS_PER_CLASSES}")
print(f"Amount of images in total: {train_set[1].shape}")
print(f"Size of each image: {train_set[0][0].shape}")

num_classes = len(NUM_ELEMENTS_PER_CLASSES)
model = create_and_train_torch_ConvNet_model(
    train_set, num_channels=(input_shape[-1], 16, 32, 64), num_epochs=8
)

# Convert to ART classifier
target_model = Classifier._to_art_classifier(
    model, "sparse_categorical_crossentropy", num_classes, input_shape
)

# Number of shadow classifiers (increase for better accuracy of the meta classifier, decrease when not enough computing power is available.)
number_of_shadow_classifiers = 2  # needs to be even

# Size of data set to train the shadow classifiers
size_set = 6

# Ratios to perform the attack for (the real ratios of our example target model is 0.66: {0: 1000, 1: 500}: 33% of data points are from class 0, 66% from class 1.)
ratios_for_attack = [0.66, 0.05]

# Number of epochs for training the meta classifier
num_epochs_meta_classifier = 10

# Classes that the attack should be performed on
classes = [0, 1]  # needs to be two values (binary attack)

attack = PropertyInferenceAttack(
    target_model,
    train_set,
    verbose=1,
    size_set=size_set,
    ratios_for_attack=ratios_for_attack,
    classes=classes,
    amount_sets=number_of_shadow_classifiers,
    num_epochs_meta_classifier=num_epochs_meta_classifier,
)
output = attack.attack()
