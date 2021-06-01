import numpy as np

from privacy_evaluator.attacks.sample_attack import SampleAttack
from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.cifar10 import CIFAR10
from privacy_evaluator.classifiers.classifier import Classifier

import torch.nn as nn


def test_sample_attack():
    x_train, y_train, x_test, y_test = CIFAR10.numpy()
    target_model = Classifier(
        load_dcti(),
        nb_classes=CIFAR10.N_CLASSES,
        input_shape=CIFAR10.INPUT_SHAPE,
        loss=nn.CrossEntropyLoss(reduction="none"),
    )

    attack = SampleAttack(target_model, x_train, y_train, x_test, y_test)

    actual_inferred_train_data, actual_inferred_test_data = attack.attack()
    expected_inferred_train_data, expected_inferred_test_data = np.ones(10), np.zeros(
        10
    )

    assert len(actual_inferred_train_data) == len(expected_inferred_train_data)
    assert len(actual_inferred_test_data) == len(expected_inferred_test_data)
    assert all(
        [
            a == b
            for a, b in zip(actual_inferred_train_data, expected_inferred_train_data)
        ]
    )
    assert all(
        [a == b for a, b in zip(actual_inferred_test_data, expected_inferred_test_data)]
    )
