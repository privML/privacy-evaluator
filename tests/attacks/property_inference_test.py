import pytest
import numpy as np

from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.models.train_cifar10_torch.data import dataset_downloader, new_dataset_from_size_dict
from privacy_evaluator.models.train_cifar10_torch.train import trainer_out_model


def test_property_inference_attack():
    train_dataset, test_dataset = dataset_downloader()
    input_shape = [32, 32, 3]
    num_classes = 2
    num_elements_per_classes = {0: 5000, 1: 5000}

    train_set, test_set = new_dataset_from_size_dict(
            train_dataset, test_dataset, num_elements_per_classes
        )

    accuracy, model = trainer_out_model(
        train_set, test_set, num_elements_per_classes, "FCNeuralNet"
    )

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(
        model, num_classes, input_shape
    )

    test_dataset=next(iter(test_dataset))[0].numpy()
    
    attack = PropertyInferenceAttack(target_model, test_dataset)
    attack.attack()
