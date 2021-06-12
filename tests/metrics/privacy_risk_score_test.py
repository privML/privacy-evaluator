import pytest

import numpy as np

import torch
import torch.nn as nn
import tensorflow as tf

from privacy_evaluator.models.torch.dcti.dcti import load_dcti
from privacy_evaluator.datasets.cifar10 import CIFAR10

from privacy_evaluator.metrics import compute_privacy_risk_score
from privacy_evaluator.classifiers.classifier import Classifier

from ..fixtures import download_models
import os
import json


def test_privacy_risk_score(download_models):
    def _rel_path(path):
        return os.path.join(os.path.dirname(__file__), path)

    try:
        assert os.path.exists(_rel_path("../model_data"))
        assert os.path.exists(
            _rel_path(
                "../model_data/Models-for-MIA/01_well-generalized-model(low-privacy-risk)"
            )
        )
        assert os.path.exists(
            _rel_path(
                "../model_data/Models-for-MIA/02_especially_for_privacy_trained_model(no-privacy-risk)"
            )
        )

        assert set(
            os.listdir(
                _rel_path(
                    "../model_data/Models-for-MIA/01_well-generalized-model(low-privacy-risk)/"
                )
            )
        ) == set(
            [
                "architecture.json",
                "dp_001-10.h5",
                "dp_001-20.h5",
                "dp_001-40.h5",
                "dp_001-30.h5",
                "stats.csv",
                "dp_001-50.h5",
            ]
        )
        assert set(
            os.listdir(
                _rel_path(
                    "../model_data/Models-for-MIA/02_especially_for_privacy_trained_model(no-privacy-risk)"
                )
            )
        ) == set(
            [
                "architecture.json",
                "dp_016-10.h5",
                "dp_016-50.h5",
                "dp_016-20.h5",
                "dp_016-30.h5",
                "dp_016-40.h5",
                "stats.csv",
            ]
        )
    except AssertionError:
        raise FileNotFoundError(
            "Model data not complete (might not have been correctly downloaded)"
        )

    # initialize high risk model
    high_risk_model_base_path = _rel_path(
        "../model_data/Models-for-MIA/01_well-generalized-model(low-privacy-risk)/dp_001-50.h5"
    )
    high_risk_model_json_path = _rel_path(
        "../model_data/Models-for-MIA/01_well-generalized-model(low-privacy-risk)/architecture.json"
    )

    with open(high_risk_model_json_path, "r") as read_file:
        json_config = json.load(read_file)
    high_risk_model = tf.keras.models.model_from_json(json_config)
    high_risk_model.load_weights(high_risk_model_base_path)

    high_risk_classifier = Classifier(
        high_risk_model,
        nb_classes=high_risk_model.output_shape[1],
        input_shape=high_risk_model.input_shape,
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )

    # initialize low risk model
    low_risk_model_base_path = _rel_path(
        "../model_data/Models-for-MIA/02_especially_for_privacy_trained_model(no-privacy-risk)/dp_016-40.h5"
    )
    low_risk_model_json_path = _rel_path(
        "../model_data/Models-for-MIA/02_especially_for_privacy_trained_model(no-privacy-risk)/architecture.json"
    )

    with open(low_risk_model_json_path, "r") as read_file:
        json_config = json.load(read_file)
    low_risk_model = tf.keras.models.model_from_json(json_config)
    low_risk_model.load_weights(low_risk_model_base_path)

    # run risk evaluation on high risk model
    (x_train, y_train), (x_test, y_test) = high_risk_model.load_data()
    high_risk_score = compute_privacy_risk_score(
        high_risk_classifier, x_train[:100], y_train[:100], x_test[:100], y_test[:100]
    )
    # run risk evaluation on low risk model
    """
    score = compute_privacy_risk_score(
        classifier, x_train[:100], y_train[:100], x_test[:100], y_test[:100]
    )
    assert bool(score)"""
