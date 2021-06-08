import tensorflow as tf
import numpy as np


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


def accuracy(outputs: tf.Tensor, labels: np.ndarray) -> float:
    """
    Calculate the accuracy given the predicted probability distributions and labels.

    Args:
        outputs: Model output given as tensor of shape `[batch_size, num_classes]`.
        labels: True class given as a numpy array of shape `[batch_size,]`.

    Returns:
        The accuracy for this batch.
    """
    correct = tf.equal(tf.argmax(outputs, 1), tf.cast(labels, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), axis=-1)
    return float(accuracy)
