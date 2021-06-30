from tensorflow import keras
from torch import nn
from art.estimators.estimator import BaseEstimator
import numpy as np
import tensorflow as tf
from copy import deepcopy
from typing import Union
import shutil


def copy_and_reset_model(
    model: Union[keras.Model, nn.Module, BaseEstimator]
) -> Union[nn.Module, keras.Model]:
    """
    Get a clone of a model (i.e. with the same architecture) and reset the weights \
        of this copy.

    :param model: A tf(keras) or torch model. Alternatively, an art-classifier encapsulating \
        a keras or torch model.
    :return: The re-initialized copy of the given model. If `model` is a keras model \
        (or art-keras) then we return a keras model; If `model` is a torch model (or art-torch) \
        then we return a torch model.
    """
    assert isinstance(model, tf.Module)
    if isinstance(model, nn.Module):
        return _copy_and_reset_torch_model(model)
    if isinstance(model, keras.Model):
        return _copy_and_reset_tf_model(model)
    if isinstance(model, BaseEstimator):
        return copy_and_reset_model(model.model)
    else:
        raise TypeError(f"Unxpected model type {str(type(model))} received.")


def _copy_and_reset_torch_model(target_model: nn.Module) -> nn.Module:
    """
    Get a clone of a Torch model and reset the weights of the copy.

    :param target_model: A `nn.Module` model, e.g. a subclass or sequential model.
    :return: A copy of `target_model`, but with every layer re-initialized.
    """
    model = deepcopy(target_model)
    for layers in model.children():
        if hasattr(layers, "__iter__"):
            for layer in layers:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        elif hasattr(layers, "reset_parameters"):
            layers.reset_parameters()
    return model


def _copy_and_reset_tf_model(target_model: keras.Model) -> keras.Model:
    """
    Get a clone of a TF model and reset the weights of the copy.

    :param target_model: A `keras.Model` model, e.g. a subclass or sequential model.
    :return: A copy of `target_model`, but with every layer re-initialized.
    """

    # Since deepcopy does not apply to TF model, we have to copy using saved file.
    target_model.save("./target_model")
    model = keras.models.load_model("./target_model")
    shutil.rmtree("./target_model")

    for layer in model.trainable_variables:
        # the default kernel initializer used by Keras is normal distribution
        layer_shape = layer.numpy().shape
        layer.assign(np.random.normal(0, 1, layer_shape))
    return model
