from tensorflow import keras
from torch import nn
from art.estimators.estimator import BaseEstimator
import numpy as np
from copy import deepcopy
from typing import Union
import shutil
from typing import Tuple

from ..models.torch.cnn import ConvNet
from .trainer import trainer

# number of channels for CNN
NUM_CHANNELS = (1, 16, 32, 64)
# number of epochs for trainer
NUM_EPOCHS = 2


def copy_and_reset_model(
    model: Union[keras.Model, nn.Module, BaseEstimator]
) -> Union[nn.Module, keras.Model]:
    """
    Get a clone of a model (i.e. with the same architecture) and reset the weights \
        of this copy.

    :param model: A TF or Torch model. Alternatively, an art-classifier encapsulating \
        a TF or Torch model.
    :return: The re-initialized copy of the given model. If `model` is a TF model \
        (or art-TF) then we return a TF model; the same applied in Torch case.
    """
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

def create_and_train_torch_ConvNet_model(
    data_set: Tuple[np.ndarray, np.ndarray], 
    num_channels: Tuple[int, ...] = NUM_CHANNELS,
    num_epochs: int = NUM_EPOCHS
) -> nn.Module:
    """
    Creates a torch ConvNet model and trains it on the provided data set.
    :param data_set: Input data set.
    :param num_channels: Number of input channels. 
    :param num_epochs: The number of times each data point in `data_set` is iterated during training.
    """
    num_elements_per_classes = dict(zip(*np.unique(data_set[1], return_counts=True)))
    num_classes = len(num_elements_per_classes)
    input_shape = data_set[0][0].shape

    model = ConvNet(num_classes, input_shape, num_channels=num_channels)
    trainer(data_set, num_elements_per_classes, model, num_epochs=num_epochs)
    return model
