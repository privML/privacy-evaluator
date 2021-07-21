import json
import os
import tensorflow as tf

from privacy_evaluator.classifiers.classifier import Classifier
from privacy_evaluator.datasets.tf.mnist import TFMNIST
from privacy_evaluator.attacks.membership_inference import (
    MembershipInferenceAttackOnPointBasis,
)

from ...fixtures import tests_tmp_path, models_path


def test_on_point_basis(models_path):
    try:
        assert os.path.exists(models_path)
        assert os.path.exists(
            os.path.join(models_path, "01_well-generalized-model(low-privacy-risk)")
        )
        assert os.path.exists(
            os.path.join(
                models_path, "02_especially_for_privacy_trained_model(no-privacy-risk)"
            )
        )
        assert set(
            os.listdir(
                os.path.join(
                    models_path, "01_well-generalized-model(low-privacy-risk)/"
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
                os.path.join(
                    models_path,
                    "02_especially_for_privacy_trained_model(no-privacy-risk)",
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
    high_risk_model_base_path = os.path.join(
        models_path, "01_well-generalized-model(low-privacy-risk)/dp_001-50.h5"
    )
    high_risk_model_json_path = os.path.join(
        models_path, "01_well-generalized-model(low-privacy-risk)/architecture.json"
    )

    with open(high_risk_model_json_path, "r") as read_file:
        json_config = json.load(read_file)

    high_risk_model = tf.keras.models.model_from_json(json_config)
    high_risk_model.load_weights(high_risk_model_base_path)

    high_risk_classifier = Classifier(
        high_risk_model,
        nb_classes=TFMNIST.N_CLASSES,
        input_shape=TFMNIST.INPUT_SHAPE,
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )

    # initialize low risk model
    low_risk_model_base_path = os.path.join(
        models_path,
        "02_especially_for_privacy_trained_model(no-privacy-risk)/dp_016-40.h5",
    )
    low_risk_model_json_path = os.path.join(
        models_path,
        "02_especially_for_privacy_trained_model(no-privacy-risk)/architecture.json",
    )

    with open(low_risk_model_json_path, "r") as read_file:
        json_config = json.load(read_file)

    low_risk_model = tf.keras.models.model_from_json(json_config)
    low_risk_model.load_weights(low_risk_model_base_path)

    low_risk_classifier = Classifier(
        low_risk_model,
        nb_classes=TFMNIST.N_CLASSES,
        input_shape=TFMNIST.INPUT_SHAPE,
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )
    x_train, y_train, x_test, y_test = TFMNIST.numpy()
    low_risk_attack = MembershipInferenceAttackOnPointBasis(high_risk_classifier)

    high_risk_attack = MembershipInferenceAttackOnPointBasis(low_risk_classifier)

    # run risk evaluation on high risk model

    high_risk_train_probs, high_risk_test_probs = low_risk_attack.attack(
        x_train[:300], y_train[:300], x_test[:300], y_test[:300]
    )
    # run risk evaluation on low risk model
    low_risk_train_probs, low_risk_test_probs = high_risk_attack.attack(
        x_train[:100], y_train[:100], x_test[:100], y_test[:100]
    )

    # run risk evaluation on high risk model

    high_risk_train_probs, high_risk_test_probs = low_risk_attack.attack()
    # run risk evaluation on low risk model
    low_risk_train_probs, low_risk_test_probs = high_risk_attack.attack()
    # assert that the low privacy model has a lower privacy risk score on the training then on the test data
    assert high_risk_train_probs.sum() / len(
        high_risk_train_probs
    ) > high_risk_test_probs.sum() / len(high_risk_test_probs)
    # assert that the score difference (train - test data) is higher for the low privacy model
    assert high_risk_train_probs.sum() / len(
        high_risk_train_probs
    ) - high_risk_test_probs.sum() / len(
        high_risk_test_probs
    ) > high_risk_train_probs.sum() / len(
        high_risk_train_probs
    ) - low_risk_test_probs.sum() / len(
        low_risk_test_probs
    )
