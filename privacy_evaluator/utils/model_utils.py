from tensorflow.keras import layers
from privacy_evaluator.models.tf.cnn import ConvNetMNIST, ConvNetCIFAR10
from privacy_evaluator.models.tf.fc_neural_net import FCNeuralNet
from privacy_evaluator.utils.data_utils import dataset_downloader, new_dataset_from_size_dict
from privacy_evaluator.utils.trainer import trainer, tester

from tensorflow import keras
from torch import nn
from art.estimators.estimator import BaseEstimator
import numpy as np
from copy import deepcopy
from typing import Union


def copy_and_reset_model(model: Union[keras.Model, nn.Module, BaseEstimator])-> Union[nn.Module, keras.Model]:
    if isinstance(model, nn.Module):
        return _copy_and_reset_torch_model(model)
    if isinstance(model, keras.Model):
        return _copy_and_reset_tf_model(model)
    if isinstance(model, BaseEstimator):
        return copy_and_reset_model(model.model)
    else:
        raise TypeError(
            f"Unxpected model type {str(type(model))} received."
        )


def _copy_and_reset_torch_model(target_model):
    model = deepcopy(target_model)
    for layers in model.children():
        if hasattr(layers, "iter"):
            for layer in layers:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        elif hasattr(layers, "reset_parameters"):
            layers.reset_parameters()
    return model


def _copy_and_reset_tf_model(target_model: keras.Model) -> keras.Model:
    target_model.save('./target_model')
    model = keras.models.load_model('./target_model')
    for layer in model.trainable_variables:
        # the default kernel initializer used by Keras is normal distribution
        layer_shape = layer.numpy().shape
        layer.assign(np.random.normal(0, 1, layer_shape))
    return model


if __name__ == "__main__":
    train_set, _ = dataset_downloader("CIFAR10")
    size_dict = {0: 1000, 1: 1000}
    train_set = new_dataset_from_size_dict(train_set, size_dict)
    
    target_model = FCNeuralNet()
    trainer(train_set, size_dict, target_model)
    print(tester(train_set, size_dict, target_model))

    copied_model = copy_and_reset_model(target_model)
    print(tester(train_set, size_dict, copied_model))
    print(tester(train_set, size_dict, target_model))
    trainer(train_set, size_dict, copied_model)
    print(tester(train_set, size_dict, copied_model))
