from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
)
from privacy_evaluator.output.user_output_property_inference_attack import (
    UserOutputPropertyInferenceAttack,
)
from privacy_evaluator.utils.model_utils import create_and_train_torch_ConvNet_model
from privacy_evaluator.utils.data_utils import create_new_dataset_with_adaptation

from typing import Dict, List
import numpy as np

# dataset for the attack
DATASET = "MNIST"
# number of channels for CNN
NUM_CHANNELS = (1, 16, 32, 64)
# number of epochs for trainer
NUM_EPOCHS = 2
# count of shadow training sets, must be even
AMOUNT_SETS = 2
# ratio and size for unbalanced data sets
SIZE_SET = 100
# ratios for different properties in sub-attacks
RATIOS_FOR_ATTACK = [0.9, 0.3]
# 0: no information; 1: backbone (most important) information; 2: utterly detailed
VERBOSE = 1
#The type of adaptation. ('mask', 'random_noise', 'brightness')
ADAPTATION = "mask"
#A ratio for the number of adapted samples
RATIO = 0.2
#Involved when adaptation is "mask", the side length of masking boxes.
BOX_LEN = 4

def test_property_inference_data_augmentation_attack(
    dataset: str = DATASET,
    num_channels: int = NUM_CHANNELS,
    num_epochs: int = NUM_EPOCHS,
    amount_sets: int = AMOUNT_SETS,
    size_set: int = SIZE_SET,
    ratios_for_attack: List[float] = RATIOS_FOR_ATTACK,
    verbose: int = VERBOSE,
    box_len: int = BOX_LEN,
    adaptation: str = ADAPTATION,
    ratio: float = RATIO
):
    train_dataset, test_dataset = dataset_downloader(dataset)
    input_shape = train_dataset[0][0].shape

    num_classes = len(np.unique(train_dataset[1]))

    train_set = create_new_dataset_with_adaptation(train_dataset, ratio, adaptation, box_len = box_len)
    
    print("Start training target model ...\n")

    # num_channels and input_shape are optional in cnn.py
    model = create_and_train_torch_ConvNet_model(train_set, num_channels, num_epochs)
    
    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(
        model, "sparse_categorical_crossentropy", num_classes, input_shape
    )
    
    #TODO start attack

