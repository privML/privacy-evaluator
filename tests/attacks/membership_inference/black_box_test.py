import torch

from privacy_evaluator.attacks.membership_inference.black_box import (
    MembershipInferenceBlackBoxAttack,
)
from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10
from privacy_evaluator.classifiers.classifier import Classifier


def test_membership_inference_black_box_attack():
    x_train, y_train, x_test, y_test = TorchCIFAR10.numpy(take=10)

    target_model = Classifier(
        load_dcti(),
        nb_classes=TorchCIFAR10.N_CLASSES,
        input_shape=TorchCIFAR10.INPUT_SHAPE,
        loss=torch.nn.CrossEntropyLoss(reduction="none"),
    )

    attack = MembershipInferenceBlackBoxAttack(
        target_model
    )

    attack.fit(x_train[:100], y_train[:100], x_test[:100], y_test[:100])
    assert attack.attack(x_train[100:200], y_train[100:200]).sum() in [58, 59]
    assert attack.attack(x_test[100:200], y_test[100:200]).sum() == 52
