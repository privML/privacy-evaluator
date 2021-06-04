import tensorflow as tf
import torch
import numpy as np
from typing import Union


# compute loss for a batch
def cross_entropy_loss(outputs: tf.Tensor, labels: np.ndarray) -> tf.Tensor:
    """
    Calculate the cross entropy loss between the predicted probability distributions
    and the labels.

    Args:
        outputs: Model output given as tensor of shape `[batch_size, num_classes]`.
        labels: True class given as a numpy array of shape `[batch_size,]`.

    Returns:
        The mean loss for this batch.
    """
    labels = tf.cast(labels, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=outputs)
    return tf.reduce_mean(loss)


def to_numpy(x: Union[torch.Tensor, tf.Tensor]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    elif isinstance(x, tf.Tensor):
        x = x.numpy()
    return x


def accuracy(
    outputs: Union[torch.Tensor, tf.Tensor, np.ndarray],
    labels: Union[torch.Tensor, tf.Tensor, np.ndarray],
) -> float:
    """
    Calculate the accuracy given the predicted probability distribution and label.

    Args:
        outputs: Model output given as tensor of shape `[batch_size, num_classes]`.
        labels: True class given as tensor of shape `[batch_size,]`.

    Returns:
        The accuracy for this batch.
    """
    outputs, labels = to_numpy(outputs), to_numpy(labels)
    assert outputs.shape[0] == labels.shape[0]

    pred = np.argmax(outputs, axis=1)
    correct = (pred == labels).sum()
    total = labels.shape[0]
    accuracy = 1.0 * correct / total
    return accuracy
