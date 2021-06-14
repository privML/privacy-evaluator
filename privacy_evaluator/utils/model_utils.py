from tensorflow.keras import layers
from privacy_evaluator.models.tf.cnn import ConvNetMNIST, ConvNetCIFAR10
from privacy_evaluator.models.tf.fc_neural_net import FCNeuralNet
from privacy_evaluator.utils.data_utils import dataset_downloader, new_dataset_from_size_dict
from privacy_evaluator.utils.trainer import trainer, tester

import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
from art.estimators.estimator import BaseEstimator
import numpy as np
from copy import deepcopy
from typing import Union


def copy_model_and_reset(
    target_model: Union[nn.Module, keras.Model, BaseEstimator]
) -> Union[nn.Module, keras.Model]:
    """Copy the architechture of a given tf or torch model and reset the weights.

    :param target_model: the model to be copied
    :return: the clone of it with newly initialized weights on the same platform
    """
    if isinstance(target_model, BaseEstimator):
        return copy_model_and_reset(target_model.model)
    
    elif isinstance(target_model, nn.Module):
        return _copy_torch_model_and_reset(target_model)
    elif isinstance(target_model, keras.Model):
        return _copy_tf_model_and_reset(target_model)
    else:
        raise TypeError("Unsupported model type!")


def _copy_torch_model_and_reset(target_model: nn.Module) -> nn.Module:
    model = deepcopy(target_model)
    for layers in model.children():
        if hasattr(layers, "__iter__"):
            for layer in layers:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        elif hasattr(layers, "reset_parameters"):
            layers.reset_parameters()
    return model


def _copy_tf_model_and_reset(target_model: keras.Model) -> keras.Model:
    model = deepcopy(target_model)
    for layer in model.trainable_variables:
        # the default kernel initializer used by Keras is normal distribution
        layer_shape = layer.numpy().shape
        layer.assign(np.random.normal(0, 1, layer_shape))
    return model



if __name__ == "__main__":
    train_set, _ = dataset_downloader("CIFAR10")
    size_dict = {0: 1000, 1: 1000}
    train_set = new_dataset_from_size_dict(train_set, size_dict)
    
    target_model = ConvNetCIFAR10()
    trainer(train_set, size_dict, target_model)
    print(tester(train_set, size_dict, target_model))

    copied_model = copy_model_and_reset(target_model)
    print(tester(train_set, size_dict, copied_model))
