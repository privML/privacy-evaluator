import numpy as np

from privacy_evaluator.attacks.sample_attack import SampleAttack
import privacy_evaluator.models.torch.dcti as dcti
from privacy_evaluator.datasets.cifar10 import CIFAR10
from privacy_evaluator.classifiers.classifier import Classifier


def test_sample_attack():
    x_train, y_train, x_test, y_test = CIFAR10.numpy()
    target_model = Classifier(dcti.load_dcti(), nb_classes=CIFAR10.N_CLASSES, input_shape=CIFAR10.INPUT_SHAPE)
    sample_attack = SampleAttack(target_model, x_train, y_train, x_test, y_test)
    assert sample_attack.attack() == np.ones(10), np.zeros(10)
