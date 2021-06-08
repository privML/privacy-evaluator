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
    train_dataset, test_dataset = dataset_downloader()
    input_shape = test_dataset[0].shape
    num_classes = 2
    num_elements_per_classes = {0: 1000, 1: 1000}

    train_set = new_dataset_from_size_dict(train_dataset, num_elements_per_classes)
    test_set = new_dataset_from_size_dict(test_dataset, num_elements_per_classes)

    model = FCNeuralNet()
    trainer(train_set, test_set, num_elements_per_classes, model)

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)

    attack = PropertyInferenceAttack(target_model, test_dataset)
    attack.attack()
    """res = attack.attack()
    f = open('test.txt', 'a')
    f.write(str(res))
    f.close()"""
