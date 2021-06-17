from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
#from privacy_evaluator.models.tf.cnn import ConvNet
from privacy_evaluator.models.torch.cnn import ConvNet
from typing import Dict

NUM_ELEMENTS_PER_CLASSES = {0: 1000, 1: 1000}
DATASET = "MNIST"


def test_property_inference_attack(num_elements_per_classes: Dict[int, int] = NUM_ELEMENTS_PER_CLASSES, dataset: str = DATASET):
    train_dataset, test_dataset = dataset_downloader(dataset)
    input_shape = test_dataset[0][0].shape

    num_classes = len(num_elements_per_classes)

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)
    # num_channels and input_shape are optional in cnn.py
    model = ConvNet(num_classes, input_shape, num_channels=(input_shape[-1], 16, 32, 64))

    trainer(train_set, num_elements_per_classes, model)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)

    attack = PropertyInferenceAttack(target_model, train_dataset)
    attack.attack()


