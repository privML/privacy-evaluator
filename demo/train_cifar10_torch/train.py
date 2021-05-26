import torch
from torch import nn
from torch.utils.data import DataLoader
from demo.train_cifar10_torch.metric import accuracy
from demo.train_cifar10_torch.model import ResNet18, ResNet50, FCNeuralNet
import os
from typing import Dict, Tuple


def trainer(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    size_dict: Dict[int, int],
    model: str = "ResNet50",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 500,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
    dropout: float = 0,
) -> float:
    """
    Get the best test accuracy after training for `num_epochs` epochs.
    """
    # create data-loaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # set model, loss function and optimizer
    num_classes = len(size_dict)
    if model == "ResNet50":
        model = ResNet50(num_classes, dropout).to(device)
    elif model == "ResNet18":
        model = ResNet18(num_classes, dropout).to(device)
    elif model == "FCNeuralNet":
        model = FCNeuralNet(num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # here class encoding is necessary since we need the dimension
    # of one-hot encoding identical to the number of classes
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}

    # start training
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            labels = labels.apply_(lambda id: class_encoding[id])
            images, labels = images.to(device), labels.to(device)

            # forward pass
            pred = model(images)
            loss = criterion(pred, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test after each epoch
        model.eval()
        accuracies = []
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.apply_(lambda id: class_encoding[id])
                images, labels = images.to(device), labels.to(device)

                # forward pass
                pred = model(images)
                batch_acc = accuracy(pred, labels)
                accuracies.append(batch_acc)

        epoch_acc = sum(accuracies) / len(accuracies)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            model_name = "torch_fc_class_0_{}_class_1_{}.pth".format(
                size_dict[0], size_dict[1]
            )
            torch.save(model, os.path.join("../../", model_name))
    return float(round(best_acc, 4))


def trainer_out_model(
    train_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    size_dict: Dict[int, int],
    model: str = "ResNet50",
    device: torch.device = torch.device("cpu"),
    batch_size: int = 500,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    weight_decay: float = 0,
    dropout: float = 0,
) -> Tuple[float, torch.nn.Module] :
    """
    Get the best test accuracy after training for `num_epochs` epochs.
    """
    # create data-loaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # set model, loss function and optimizer
    num_classes = len(size_dict)
    if model == "ResNet50":
        model = ResNet50(num_classes, dropout).to(device)
    elif model == "ResNet18":
        model = ResNet18(num_classes, dropout).to(device)
    elif model == "FCNeuralNet":
        model = FCNeuralNet(num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # here class encoding is necessary since we need the dimension
    # of one-hot encoding identical to the number of classes
    class_encoding = {class_id: i for i, (class_id, _) in enumerate(size_dict.items())}

    # start training
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            labels = labels.apply_(lambda id: class_encoding[id])
            images, labels = images.to(device), labels.to(device)

            # forward pass
            pred = model(images)
            loss = criterion(pred, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test after each epoch
        model.eval()
        accuracies = []
        with torch.no_grad():
            for images, labels in test_loader:
                labels = labels.apply_(lambda id: class_encoding[id])
                images, labels = images.to(device), labels.to(device)

                # forward pass
                pred = model(images)
                batch_acc = accuracy(pred, labels)
                accuracies.append(batch_acc)

        epoch_acc = sum(accuracies) / len(accuracies)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            model_name = "torch_fc_class_0_{}_class_1_{}.pth".format(
                size_dict[0], size_dict[1]
            )
            torch.save(model, os.path.join("../../", model_name))
    return float(round(best_acc, 4)), model