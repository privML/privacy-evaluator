from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10
from privacy_evaluator.classifiers.classifier import Classifier
import torch.nn as nn


def test_classifier():
    x_train, _, _, _ = TorchCIFAR10.numpy()
    classifier = Classifier(
        load_dcti(),
        nb_classes=TorchCIFAR10.N_CLASSES,
        input_shape=TorchCIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )
    assert classifier.predict(x_train).shape == (
        x_train.shape[0],
        TorchCIFAR10.N_CLASSES,
    )
