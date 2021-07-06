from typing import Callable
import numpy as np

from ...metrics.basics import (
    accuracy,
    train_to_test_accuracy_gap,
    train_to_test_accuracy_ratio,
)
from ..attack import Attack
from ...classifiers.classifier import Classifier
from ...validators.attack import validate_parameters

from ...output.user_output_inference_attack import UserOutputInferenceAttack


class MembershipInferenceAttack(Attack):
    """MembershipInferenceAttack base class.

    Interpretation of Outcome:

    Attack Model Accuracy:
    The attack model accuracy specifies how well the membership attack model performs in predicting if a given data
    point was used for training the target model. Since we have a two-class classification problem that the attack model
    solves (member or non-member), the lowest possible accuracy is 50% (random guessing for each sample). The best
    accuracy is at 100% if the model predicts every data point is sees right as member or non-member.

    Train-Test-Gap (difference):
    If your model has a train-test-gap larger than 5%, this could be a sign that your model overfits. Overfitting can be
    beneficial for successful membership inference attacks [1]. Therefore, you might want to reduce it by introducing
    regularization methods in your training, or using specific privacy methods[2,3], such as Differential Privacy [4].

    [1]S. Yeom, I. Giacomelli, M. Fredrikson, and S. Jha. \Privacy Risk in Machine Learning: Analyzing the Connection
    to Overfitting". In: 2018 IEEE 31st Computer Security Foundations Symposium (CSF). July 2018, pp. 268{282.
    doi:10.1109/CSF.2018.00027.
    [2] Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017. Mem-bership Inference Attacks Against
    Machine Learning Models. In2017 IEEE Sym-posium on Security and Privacy (SP). 3–18.
    [3] Milad Nasr, Reza Shokri, and Amir Houmansadr. 2018. Machine Learning withMembership Privacy Using Adversarial
    Regularization. InProceedings of the 2018ACM SIGSAC Conference on Computer and Communications
    Security(Toronto,Canada)(CCS ’18). Association for Computing Machinery, New York, NY, USA,634–64
    [4] Cynthia Dwork. 2006.  Differential Privacy. InAutomata, Languages and Pro-gramming, Michele Bugliesi,
    Bart Preneel, Vladimiro Sassone, and Ingo Wegener(Eds.). Springer Berlin Heidelberg, Berlin, Heidelberg
    """

    _ART_MEMBERSHIP_INFERENCE_ATTACK_MODULE = (
        "art.attacks.inference.membership_inference"
    )

    def __init__(self, target_model: Classifier, init_art_attack: bool = True, **kwargs):
        """Initializes a MembershipInferenceAttack class.

        :param target_model: Target model to be attacked.
        :param init_art_attack: Indicates if belonging ART attack should be initialized.
        """
        super().__init__(target_model)
        if init_art_attack:
            self._art_attack = self._init_art_attack(target_model, **kwargs)
        self._art_attack_model_fitted = False

    def attack(
        self, x: np.ndarray, y: np.ndarray, probabilities: bool = False, **kwargs
    ) -> np.ndarray:
        """Performs the membership inference attack on the target model.

        :param x: Data to be attacked.
        :param y: True, one-hot encoded labels for `x`.
        :param probabilities: If True, the method returns probability vector for each data point instead of predicted class.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        :raises Exception: If attack model is not fitted.
        """

        validate_parameters(
            "attack",
            target_model=self.target_model,
            x=x,
            y=y,
        )
        if self._art_attack_model_fitted is False:
            raise Exception(
                "The attack model needs to be fitted first. Please run `fit()` on the attack."
            )

        return self._art_attack.infer(
            x, y, probabilities=probabilities, **kwargs
        ).reshape(-1)

    def attack_output(
        self, x: np.ndarray, y: np.ndarray, y_attack: np.ndarray
    ) -> UserOutputInferenceAttack:
        """Creates attack output metrics in an extractable format.

        :param x: Data to be attacked.
        :param y: True, one-hot encoded labels for `x`.
        :param y_attack: True, non one-hot encoded labels for the attack model (e.g. the membership status).
        :return: An dict with attack output metrics including the target model train and test accuracy, target model
        train to test accuracy gap and ratio and the attack model accuracy.
        """

        validate_parameters(
            "attack_output", target_model=self.target_model, x=x, y=y, y_attack=y_attack
        )

        train_accuracy = accuracy(self.y_train, self.target_model.predict(self.x_train))
        test_accuracy = accuracy(self.y_test, self.target_model.predict(self.x_test))
        y_attack_prediction = self.attack(x, y)

        return UserOutputInferenceAttack(
            train_accuracy,
            test_accuracy,
            train_to_test_accuracy_gap(train_accuracy, test_accuracy),
            train_to_test_accuracy_ratio(train_accuracy, test_accuracy),
            accuracy(y_attack, y_attack_prediction),
        )

    def fit(self, **kwargs):
        """Fits the attack model.

        :param kwargs: Keyword arguments for the fitting.
        """
        raise NotImplementedError(
            "Method `attack()` needs to be implemented in subclass"
        )

    @classmethod
    def _art_module(cls) -> str:
        """Returns the matching ART module for this class.

        :return: Matching ART module.
        """
        return cls._ART_MEMBERSHIP_INFERENCE_ATTACK_MODULE

    @classmethod
    def _art_class(cls) -> str:
        """Returns the matching ART class for this class.

        :return: Matching ART class.
        :raises AttributeError: If `_ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS` is not defined.

        """
        try:
            return cls._ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS
        except AttributeError:
            raise AttributeError(
                "Attribute `_ART_MEMBERSHIP_INFERENCE_ATTACK_CLASS` needs to be defined in subclass."
            )

    @classmethod
    def _init_art_attack(cls, target_model: Classifier, **kwargs):
        """Initializes an ART attack.

        :param target_model: Target model to be attacked.
        :param kwargs: Keyword arguments of the attack.
        :return: Instance of an ART attack.
        """
        _art_module = __import__(cls._art_module(), fromlist=[cls._art_class()])
        _art_class = getattr(_art_module, cls._art_class())
        return _art_class(target_model.to_art_classifier(), **kwargs)

    @staticmethod
    def _fit_decorator(fit_function: Callable):
        """Decorator for the `fit()` methods of the subclasses.

        Defines a decorator for the `fit()` method which validates the fit parameters and checks weather the attack
        model was already fitted. If not, attack model is fitted and `_art_attack_model_fitted` is set to `True`.

        :param fit_function: Actual `fit()` method that should be decorated.
        :return: Decorator method.
        """

        def __fit_decorator(self, **kwargs):
            validate_parameters("fit", **kwargs)
            if self._art_attack_model_fitted is False:
                fit_function(self, **kwargs)
                self._art_attack_model_fitted = True

        return __fit_decorator
