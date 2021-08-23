from privacy_evaluator.attacks.property_inference_attack.property_inference_class_distribution_attack import (
    PropertyInferenceClassDistributionAttack,
)
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.output.user_output_property_inference_attack import (
    UserOutputPropertyInferenceAttack,
)
from privacy_evaluator.utils.model_utils import create_and_train_torch_ConvNet_model

from typing import Dict, List
import logging

# ratio for target model
NUM_ELEMENTS_PER_CLASSES = {0: 1000, 1: 1000}
# dataset for the attack
DATASET = "MNIST"
# number of channels for CNN
NUM_CHANNELS = (1, 16, 32, 64)
# number of epochs for trainer
NUM_EPOCHS = 2
# count of shadow training sets, must be even
AMOUNT_SETS = 2
# ratio and size for unbalanced data sets
SIZE_SHADOW_TRAINING_SET = 100
# ratios for different properties in sub-attacks
RATIOS_FOR_ATTACK = [0.9, 0.3]
# classes the attack should be performed on
CLASSES = [4, 5]
# 0: no information; 1: backbone (most important) information; 2: utterly detailed
VERBOSE = 1
# number of epochs for training the meta classifier
NUM_EPOCHS_META_CLASSIFIER = 2
# ratio of negation of property
NEGATIVE_RATIO = 0.5


def test_property_inference_class_distribution_attack(
    num_elements_per_classes: Dict[int, int] = NUM_ELEMENTS_PER_CLASSES,
    dataset: str = DATASET,
    num_channels: int = NUM_CHANNELS,
    num_epochs: int = NUM_EPOCHS,
    amount_sets: int = AMOUNT_SETS,
    size_shadow_training_set: int = SIZE_SHADOW_TRAINING_SET,
    ratios_for_attack: List[float] = RATIOS_FOR_ATTACK,
    negative_ratio: int = NEGATIVE_RATIO,
    classes: List[int] = CLASSES,
    verbose: int = VERBOSE,
    num_epochs_meta_classifier: int = NUM_EPOCHS_META_CLASSIFIER,
):
    logger = logging.getLogger(__name__)
    if verbose == 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logger.setLevel(level)

    train_dataset, test_dataset = dataset_downloader(dataset)
    input_shape = train_dataset[0][0].shape

    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)

    logger.info("Start training target model ...\n")

    # num_channels and input_shape are optional in cnn.py
    model = create_and_train_torch_ConvNet_model(train_set, num_channels, num_epochs)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(
        model, "sparse_categorical_crossentropy", num_classes, input_shape
    )
    logger.info("Start attack ...")

    attack = PropertyInferenceClassDistributionAttack(
        target_model,
        train_dataset,
        amount_sets=amount_sets,
        size_shadow_training_set=size_shadow_training_set,
        ratios_for_attack=ratios_for_attack,
        negative_ratio=negative_ratio,
        classes=classes,
        verbose=verbose,
        num_epochs_meta_classifier=num_epochs_meta_classifier,
    )
    assert (
        attack.input_shape == input_shape
    ), f"Wrong input shape. Input shape should be {input_shape}."
    assert (
        attack.amount_sets >= 2 and attack.amount_sets % 2 == 0
    ), "Number of shadow classifiers must be even and greater than 1."
    output = attack.attack()
    # we expect the ratios to be ordered
    ratios_for_attack.sort()

    assert isinstance(
        output, UserOutputPropertyInferenceAttack
    ), "Wrong output type of attack."
    assert (
        attack.ratios_for_attack == ratios_for_attack
    ), "Ratios for properties are not equal to input."
    assert (
        attack.amount_sets == amount_sets
    ), "Number of shadow classifiers are not equal to input."
    assert (
        attack.size_shadow_training_set == size_shadow_training_set
    ), "Number of samples is not equal to input."
    assert attack.classes == classes, "Classes are not equal to input classes."
    assert len(output.output) == len(
        ratios_for_attack
    ), "Output is not compatible to input."
