from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.cifar10 import CIFAR10
from privacy_evaluator.classifiers.classifier import Classifier
import torch.nn as nn


def test_classifier():
    x_train, _, _, _ = CIFAR10.numpy()
    classifier = Classifier(
        load_dcti(),
        nb_classes=CIFAR10.N_CLASSES,
        input_shape=CIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )
    assert classifier.predict(x_train).shape == (x_train.shape[0], CIFAR10.N_CLASSES)
