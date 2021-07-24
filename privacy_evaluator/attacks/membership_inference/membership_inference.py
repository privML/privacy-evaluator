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

import logging


class MembershipInferenceAttack(Attack):
    """MembershipInferenceAttack base class.

    Interpretation of Outcome:

    Attack Model Accuracy:
    The attack model accuracy specifies how well the membership attack model performs in predicting if a given data
    point was used for training the target model. Since we have a two-class classification problem that the attack model
    solves (member or non-member), the lowest possible accuracy is 50%, which is equal to randomly guessing the class of
    a sample.  Theoretically, accuracy can also go below 50%, down to 0%. In this case, an inversion of the prediction
    of the labels can yield better results. E.g. if the accuracy is 40%, an inversion of the predicted labels yields 60%
    accuracy, hence 50% is considered the worst, or lowest possible accuracy. The best accuracy is at 100% if the model
    predicts every data point is sees right as member or non-member.

    Train-Test-Gap (difference):
    If your model has a train-test-gap larger than 5%, this could be a sign that your model overfits. Overfitting can be
    beneficial for successful membership inference attacks [1]. Therefore, you might want to reduce it by introducing
    regularization methods in your training, or using specific privacy methods [2,3], such as Differential Privacy [4].

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

    def __init__(
        self, target_model: Classifier, init_art_attack: bool = True, **kwargs
    ):
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
        :param probabilities: If `True`, the method returns the probability for each
            data sample being a member.
        :param kwargs: Keyword arguments of the attack.
        :return: An array holding the inferred membership status, 1 indicates a member
            and 0 indicates non-member.
            A value between 0 and 1 indicates the probability of being a member
            if `probabilities` is set to `True`.
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

        logger = logging.getLogger(__name__)
        logger.info("running the Attack on the model")
        membership_pred = self._art_attack.infer(
            x, y, probabilities=probabilities, **kwargs
        )

        # Some ART attacks return a 2D vector for each data sample,
        # where the first value is the probability of the data sample *not* being a
        # member of the training set and the second value is the probability of
        # that data sample being a member.
        # Only one of the probabilities is needed to calculate the other,
        # so we return the probability of being a member only.
        if probabilities and membership_pred.shape[1] == 2:
            membership_pred = membership_pred[:, 1]

        return membership_pred.flatten()

    def attack_output(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        y_attack: np.ndarray,
    ) -> UserOutputInferenceAttack:
        """Creates attack output metrics in an extractable format.

        :param x: Data to be attacked.
        :param y: True, one-hot encoded labels for `x`.
        :param x_train: Data which was used to train the target model and will be used for training the attack model.
        :param y_train: True, one-hot encoded labels for `x_train`.
        :param x_test: Data that was not used to train the target model and will be used for training the attack model.
        :param y_test: True, one-hot encoded labels for `x_test`.
        :param y_attack: True, non one-hot encoded labels for the attack model (e.g. the membership status).
        :return: An dict with attack output metrics including the target model train and test accuracy, target model
        train to test accuracy gap and ratio and the attack model accuracy.
        """

        validate_parameters(
            "attack_output", target_model=self.target_model, x=x, y=y, y_attack=y_attack
        )
        logger = logging.getLogger(__name__)
        logger.info("calculating train_accuracy")
        train_accuracy = accuracy(y_train, self.target_model.predict(x_train))
        logger.info("calculating test_accuracy")
        test_accuracy = accuracy(y_test, self.target_model.predict(x_test))
        logger.info("calculating attack_prediction")
        y_attack_prediction = self.attack(x, y)

        return UserOutputInferenceAttack(
            train_accuracy,
            test_accuracy,
            train_to_test_accuracy_gap(train_accuracy, test_accuracy),
            train_to_test_accuracy_ratio(train_accuracy, test_accuracy),
            accuracy(y_attack, y_attack_prediction),
        )

    def fit(self, *args, **kwargs):
        """Fits the attack model.

        :param args: Arguments for the fitting.
        :param kwargs: Keyword arguments for fitting the attack model.
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

        def __fit_decorator(self, *args, **kwargs):
            if self._art_attack_model_fitted is False:
                validate_parameters("fit", self.target_model, *args, **kwargs)
                fit_function(self, *args, **kwargs)
                self._art_attack_model_fitted = True

        return __fit_decorator
