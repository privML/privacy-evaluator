import pytest

from privacy_evaluator.attacks.attack import Attack
from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10
from privacy_evaluator.classifiers.classifier import Classifier

import torch.nn as nn


def test_attack():
    x_train, y_train, x_test, y_test = TorchCIFAR10.numpy(take=10)
    target_model = Classifier(
        load_dcti(),
        nb_classes=TorchCIFAR10.N_CLASSES,
        input_shape=TorchCIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )
    attack = Attack(target_model, x_train, y_train, x_test, y_test)
    with pytest.raises(NotImplementedError):
        attack.attack(x_train, y_train)
