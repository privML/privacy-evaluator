import pytest

from privacy_evaluator.attacks.membership_inference.membership_inference import (
    MembershipInferenceAttack,
)
from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10
from privacy_evaluator.classifiers.classifier import Classifier

import torch.nn as nn


def test_membership_inference_attack():
    target_model = Classifier(
        load_dcti(),
        nb_classes=TorchCIFAR10.N_CLASSES,
        input_shape=TorchCIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )

    with pytest.raises(AttributeError):
        MembershipInferenceAttack(target_model)
