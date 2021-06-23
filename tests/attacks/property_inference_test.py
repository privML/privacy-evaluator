from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.torch.cnn import ConvNet
from typing import Dict
from torch import nn


NUM_ELEMENTS_PER_CLASSES = {0: 1000, 1: 1000}
DATASET = "MNIST"


def test_property_inference_attack(
    num_elements_per_classes: Dict[int, int] = NUM_ELEMENTS_PER_CLASSES,
    dataset: str = DATASET,
):
    train_dataset, test_dataset = dataset_downloader(dataset)
    input_shape = test_dataset[0][0].shape

    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)
    # num_channels and input_shape are optional in cnn.py
    model = ConvNet(
        num_classes, input_shape, num_channels=(input_shape[-1], 16, 32, 64)
    )

    print("Start training target model ...\n")
    trainer(train_set, num_elements_per_classes, model, num_epochs=2)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(
        model, "sparse_categorical_crossentropy", num_classes, input_shape
    )
    print("Start attack ...")
    # test parameters for PIA:
    amount_sets = 2
    size_set = 100
    ratios_for_attack = [0.9, 0.3]
    classes = [4, 5]

    attack = PropertyInferenceAttack(
        target_model,
        train_dataset,
        verbose=1,
        size_set=size_set,
        ratios_for_attack=ratios_for_attack,
        classes=classes,
        amount_sets=amount_sets,
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

    assert isinstance(output, tuple) and list(map(type, output)) == [
        str,
        dict,
    ], "Wrong output type of attack."
    assert (
        attack.ratios_for_attack == ratios_for_attack
    ), "Ratios for properties are not equal to input."
    assert (
        attack.amount_sets == amount_sets
    ), "Number of shadow classifiers are not equal to input."
    assert attack.size_set == size_set, "Number of samples is not equal to input."
    assert attack.classes == classes, "Classes are not equal to input classes."
    assert len(output[1]) == len(classes), "Output is not compatible to input."
