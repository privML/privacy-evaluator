from privacy_evaluator.attacks.membership_inference.black_box import (
    MembershipInferenceBlackBoxAttack,
)
from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.cifar10 import CIFAR10
from privacy_evaluator.classifiers.classifier import Classifier

import torch.nn as nn


def test_membership_inference_black_box_attack():
    x_train, y_train, x_test, y_test = CIFAR10.numpy(model_type="torch")
    target_model = Classifier(
        load_dcti(),
        nb_classes=CIFAR10.N_CLASSES,
        input_shape=CIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )
    attack = MembershipInferenceBlackBoxAttack(
        target_model, x_train[:100], y_train[:100], x_test[:100], y_test[:100]
    )
    attack.fit()
    attack.attack(x_train[:100], y_train[:100])
