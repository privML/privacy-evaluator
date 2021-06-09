import pytest

from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import (
    dataset_downloader,
    new_dataset_from_size_dict,
)
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.torch.fc_neural_net import FCNeuralNet


def test_property_inference_attack():
    train_dataset, test_dataset = dataset_downloader("CIFAR10")
    input_shape = test_dataset[0][0].shape

    num_elements_per_class = {0: 1000, 1: 1000}
    num_classes = len(num_elements_per_class)

    train_set = new_dataset_from_size_dict(
        train_dataset, num_elements_per_class
    )
    test_set = new_dataset_from_size_dict(
        test_dataset, num_elements_per_class
    )

    model = FCNeuralNet()
    trainer(
        test_set, num_elements_per_class, model
    )

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)

    attack = PropertyInferenceAttack(target_model, train_dataset)
    attack.attack()
