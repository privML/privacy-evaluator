import tensorflow as tf
from tensorflow import keras
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Dict, Union
from privacy_evaluator.utils.metric import cross_entropy_loss, accuracy


def trainer(
    train_set: Union[Tuple[np.ndarray, np.ndarray], torch.utils.data.Dataset],
    size_dict: Dict[int, int],
    model: Union[nn.Module, keras.Model],
    batch_size: int = 250,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
):
    if isinstance(model, keras.Model):
        return _trainer_tf(
            train_set,
            size_dict,
            model,
            batch_size,
            num_epochs,
            learning_rate,
            weight_decay,
        )
    elif isinstance(model, nn.Module):
        return _trainer_torch(
            train_set,
            size_dict,
            model,
            batch_size,
            num_epochs,
            learning_rate,
            weight_decay,
        )


def tester(
    test_set: Union[Tuple[np.ndarray, np.ndarray], torch.utils.data.Dataset],
    size_dict: Dict[int, int],
    model: Union[nn.Module, keras.Model],
    batch_size: int = 500,
) -> float:
    "Return the accuracy of the model on the test set"
    if isinstance(model, keras.Model):
        return _tester_tf(test_set, size_dict, model, batch_size)
    elif isinstance(model, nn.Module):
        return _tester_torch(test_set, size_dict, model, batch_size)


def _trainer_tf(
    train_set: Tuple[np.ndarray, np.ndarray],
    size_dict: Dict[int, int],
    model: keras.Model,
    batch_size: int = 500,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
):
    """
    Train the given model on the given dataset.
    """

    # set device
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    # create dataloader
    train_loader = tf.data.Dataset.from_tensor_slices(train_set)
    train_loader = train_loader.shuffle(
        buffer_size=train_set[1].shape[0], reshuffle_each_iteration=True
    ).batch(batch_size)

    # set optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # here class encoding is necessary since we need the dimension
    # of one-hot encoding identical to the number of classes
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}

    # start training
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


def _trainer_torch(
    train_set: Union[Tuple[np.ndarray, np.ndarray], torch.utils.data.Dataset],
    size_dict: Dict[int, int],
    model: nn.Module,
    batch_size: int = 500,
    num_epochs: int = 20,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
):
    """
    Train the given model on the given dataset.
    """
    # convert np.array datasets into torch dataset
    if isinstance(train_set, tuple):
        train_x, train_y = train_set
        train_x, train_y = map(torch.tensor, (train_x, train_y))
        train_set = torch.utils.data.TensorDataset(train_x, train_y)

    # create data-loader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )

    # load model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # set model, loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # here class encoding is necessary since we need the dimension
    # of one-hot encoding identical to the number of classes
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}

    # start training
    for _ in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            labels = labels.apply_(lambda id: class_encoding[id])
            images = images / 255.0
            labels = labels.to(torch.long)
            images, labels = images.to(device), labels.to(device)

            # forward pass
            pred = model(images)
            loss = criterion(pred, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _tester_tf(
    test_set: Union[Tuple[np.ndarray, np.ndarray], torch.utils.data.Dataset],
    size_dict: Dict[int, int],
    model: keras.Model,
    batch_size: int = 500,
) -> float:
    # create test dataloader
    test_loader = tf.data.Dataset.from_tensor_slices(test_set)
    test_loader = test_loader.shuffle(
        buffer_size=test_set[1].shape[0], reshuffle_each_iteration=False
    ).batch(batch_size)

    batch_acc = []
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}
    for images, labels in test_loader:
        labels = np.vectorize(lambda id: class_encoding[id])(labels)
        preds = model(images, training=False)
        batch_acc.append(accuracy(preds, labels))
    mean_acc = sum(batch_acc) / len(batch_acc)
    return float(round(mean_acc, 4))


def _tester_torch(
    test_set: Union[Tuple[np.ndarray, np.ndarray], torch.utils.data.Dataset],
    size_dict: Dict[int, int],
    model: nn.Module,
    batch_size: int = 500,
) -> float:
    # convert np.array datasets into torch dataset
    if isinstance(test_set, tuple):
        test_x, test_y = test_set
        test_x, test_y = map(torch.tensor, (test_x, test_y))
        test_set = torch.utils.data.TensorDataset(test_x, test_y)

    # create test dataloader
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # load model to Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}

    batch_acc = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.apply_(lambda id: class_encoding[id])
            images, labels = images.to(device), labels.to(device)

            # forward pass
            pred = model(images)
            batch_acc.append(accuracy(pred, labels))

    mean_acc = sum(batch_acc) / len(batch_acc)
    return float(round(mean_acc, 4))
