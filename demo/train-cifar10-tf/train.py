import tensorflow as tf
import numpy as np
from typing import Tuple, Dict
from model import ResNet50, FCNeuralNet
from metric import cross_entropy_loss, accuracy
import os


def trainer(
    train_set: np.ndarray,
    test_set: np.ndarray,
    size_dict: Dict[int, int],
    model: str = "ResNet50",
    batch_size: int = 500,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
    dropout: float = 0,
) -> float:
    """
    Get the best test accuracy during training for `num_epochs` epochs.
    """
    # create dataloader
    train_loader = tf.data.Dataset.from_tensor_slices(train_set)
    test_loader = tf.data.Dataset.from_tensor_slices(test_set)
    train_loader = train_loader.shuffle(
        buffer_size=train_set[1].shape[0], reshuffle_each_iteration=True
    ).batch(batch_size)
    test_loader = test_loader.shuffle(
        buffer_size=test_set[1].shape[0], reshuffle_each_iteration=False
    ).batch(batch_size)

    # set model and optimizer
    num_classes = len(size_dict)
    if model == "ResNet50":
        model = ResNet50(num_classes, dropout)
    elif model == "FCNeuralNet":
        model = FCNeuralNet(num_classes, dropout)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # here class encoding is necessary since we need the dimension
    # of one-hot encoding identical to the number of classes
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}

    # start training
    best_acc = 0
    for _ in range(num_epochs):
        for images, labels in train_loader:
            labels = np.vectorize(lambda id: class_encoding[id])(labels)
            with tf.GradientTape() as g:
                # forward pass
                preds = model(images, training=True)
                loss = cross_entropy_loss(preds, labels)
                l2_loss = weight_decay * tf.add_n(
                    [tf.nn.l2_loss(v) for v in model.trainable_variables]
                )
                loss += l2_loss

            # backward pass
            grad = g.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))

        # test after each epoch
        accuracies = []
        for images, labels in test_loader:
            labels = np.vectorize(lambda id: class_encoding[id])(labels)
            preds = model(images, training=False)
            batch_acc = accuracy(preds, labels)
            accuracies.append(batch_acc)

        epoch_acc = sum(accuracies) / len(accuracies)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            model_name = "tf_fc_class_0_{}_class_1_{}".format(
                size_dict[0], size_dict[1]
            )
            model.save(os.path.join("../../", model_name))
    return float(round(best_acc, 4))
