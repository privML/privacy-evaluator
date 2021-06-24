from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.torch.cnn import ConvNet
from typing import Dict, List
from torch import nn

#ratio for target model
NUM_ELEMENTS_PER_CLASSES = {0: 1000, 1: 1000}
#dataset for the attack
DATASET = "MNIST"
#number of channels for CNN
NUM_CHANNELS = (1, 16, 32, 64)
#number of epochs for trainer
NUM_EPOCHS = 2
# count of shadow training sets, must be even
AMOUNT_SETS = 2
# ratio and size for unbalanced data sets
SIZE_SET = 100
# ratios for different properties in sub-attacks
RATIOS_FOR_ATTACK = [0.9, 0.3]
# classes the attack should be performed on
CLASSES = [4,5]
#0: no information; 1: backbone (most important) information; 2: utterly detailed
VERBOSE = 1



def test_property_inference_attack(
    num_elements_per_classes: Dict[int, int] = NUM_ELEMENTS_PER_CLASSES,
    dataset: str = DATASET,
    num_channels: int = NUM_CHANNELS,
    num_epochs: int = NUM_EPOCHS,
    amount_sets: int = AMOUNT_SETS,
    size_set: int = SIZE_SET,
    ratios_for_attack: List[float] = RATIOS_FOR_ATTACK,
    classes: List[int] = CLASSES,
    verbose: int = VERBOSE
):
    train_dataset, test_dataset = dataset_downloader(dataset)
    input_shape = test_dataset[0][0].shape

    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)
    # num_channels and input_shape are optional in cnn.py
    model = ConvNet(
        num_classes, input_shape, num_channels= num_channels
    )

    print("Start training target model ...\n")
    trainer(train_set, num_elements_per_classes, model, num_epochs= num_epochs)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(
        model, "sparse_categorical_crossentropy", num_classes, input_shape
    )
    print("Start attack ...")

    attack = PropertyInferenceAttack(
        target_model,
        train_dataset,
        verbose= verbose,
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
