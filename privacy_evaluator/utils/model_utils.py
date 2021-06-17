from tensorflow import keras
from torch import nn
from art.estimators.estimator import BaseEstimator
import numpy as np
from copy import deepcopy
from typing import Union
import shutil

def copy_and_reset_model(
    model: Union[keras.Model, nn.Module, BaseEstimator]
) -> Union[nn.Module, keras.Model]:
    if isinstance(model, nn.Module):
        return _copy_and_reset_torch_model(model)
    if isinstance(model, keras.Model):
        return _copy_and_reset_tf_model(model)
    if isinstance(model, BaseEstimator):
        return copy_and_reset_model(model.model)
    else:
        raise TypeError(f"Unxpected model type {str(type(model))} received.")


def _copy_and_reset_torch_model(target_model):
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
    target_model.save("./target_model")
    model = keras.models.load_model("./target_model")
    shutil.rmtree("./target_model")
    for layer in model.trainable_variables:
        # the default kernel initializer used by Keras is normal distribution
        layer_shape = layer.numpy().shape
        #layer.assign(np.random.normal(0, 1, layer_shape))
        layer.assign(np.zeros(layer.shape))
    return model


