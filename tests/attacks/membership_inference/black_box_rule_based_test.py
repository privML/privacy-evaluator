import torch

from privacy_evaluator.attacks.membership_inference.black_box_rule_based import (
    MembershipInferenceBlackBoxRuleBasedAttack,
)
from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10
from privacy_evaluator.classifiers.classifier import Classifier


def test_membership_inference_black_box_rule_based_attack():
    x_train, y_train, x_test, y_test = TorchCIFAR10.numpy(take=10)

    target_model = Classifier(
        load_dcti(device=torch.device("cpu")),
        nb_classes=TorchCIFAR10.N_CLASSES,
        input_shape=TorchCIFAR10.INPUT_SHAPE,
        loss=torch.nn.CrossEntropyLoss(reduction="none"),
    )

    attack = MembershipInferenceBlackBoxRuleBasedAttack(
        target_model
    )

    assert attack.attack(x_train[100:200], y_train[100:200]).sum() == 92
    assert attack.attack(x_test[100:200], y_test[100:200]).sum() == 88
