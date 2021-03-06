import torch

from privacy_evaluator.attacks.membership_inference.label_only_decision_boundary import (
    MembershipInferenceLabelOnlyDecisionBoundaryAttack,
)
from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.datasets.torch.cifar10 import TorchCIFAR10
from privacy_evaluator.models.torch.dcti.dcti import load_dcti


def test_inference_label_only_decision_boundary_attack():
    x_train, y_train, x_test, y_test = TorchCIFAR10.numpy(take=10)

    target_model = Classifier(
        load_dcti(),
        nb_classes=TorchCIFAR10.N_CLASSES,
        input_shape=TorchCIFAR10.INPUT_SHAPE,
        loss=torch.nn.CrossEntropyLoss(reduction="none"),
    )

    attack = MembershipInferenceLabelOnlyDecisionBoundaryAttack(target_model)

    attack.fit(
        x_train[:1],
        y_train[:1],
        x_test[:1],
        y_test[:1],
        max_iter=1,
        max_eval=1,
        init_eval=1,
    )
    attack.attack(x_train[1:2], y_train[1:2])
    attack.attack(x_test[1:2], y_test[1:2])
