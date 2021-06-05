import pytest

from privacy_evaluator.attacks.property_inference_attack import PropertyInferenceAttack
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.utils.data_utils import dataset_downloader, new_dataset_from_size_dict
from privacy_evaluator.utils.trainer import trainer
from privacy_evaluator.models.torch.fc_neural_net import FCNeuralNet


def test_property_inference_attack():
    train_dataset, test_dataset = dataset_downloader("MNIST")
    input_shape = test_dataset[0][0].shape

    num_elements_per_class = {0: 500, 1: 500} #TODO change back to higher number
    num_classes = len(num_elements_per_class)

    train_set = new_dataset_from_size_dict(
        train_dataset, num_elements_per_class
    )
    test_set = new_dataset_from_size_dict(
        test_dataset, num_elements_per_class
    )

    model = FCNeuralNet(input_shape)
    trainer(
        test_set, num_elements_per_class, model
    )

    # change pytorch classifier to art classifier
    target_model = Classifier._to_art_classifier(model, num_classes, input_shape)

    attack = PropertyInferenceAttack(target_model, train_dataset)
    print(attack.attack())

test_property_inference_attack()




