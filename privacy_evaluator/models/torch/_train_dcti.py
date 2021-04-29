import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from dcti import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, loader, optimizer, criterion, epoch):
    ls = []
    net.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)

        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        ls.append(loss.item())

    return torch.tensor(ls).mean()


def test(net, loader):
    net.eval()
    batch_size = loader.batch_size
    prediction = torch.zeros(len(loader.dataset))
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(device)
            output = net(data).argmax(dim=1).cpu()
            prediction[i * batch_size : i * batch_size + len(output) :] = output

    return prediction


def main():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=4)

    net = DCTI().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    y_true = torch.from_numpy(np.fromiter((x[1] for x in test_set), int))
    for epoch in range(1, 101):
        loss = train(net, train_loader, optimizer, criterion, epoch)
        y_pred = test(net, test_loader)
        accuracy = ((y_true == y_pred).sum() / len(y_true)).item()

        print(f"Train epoch: {epoch:>3}\t Loss: {loss:.4f}\t Accuracy: {accuracy:.2f}")

    torch.save(net.state_dict(), "./dcti/model.pth")


if __name__ == "__main__":
    main()
