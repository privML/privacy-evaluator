import torch
import torch.nn as nn
import torch.optim as optim

from privacy_evaluator.models.torch.dcti.dcti import DCTI
from privacy_evaluator.datasets.cifar10 import CIFAR10
from privacy_evaluator.metrics.basics import accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, loader, optimizer, criterion):
    ls = []
    net.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)

        output = net(data)
        loss = criterion(output, torch.argmax(target, dim=1))
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
    train_loader, test_loader = CIFAR10.pytorch_loader()
    _, _, _, y_test = CIFAR10.numpy("torch")
    net = DCTI().to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 101):
        loss = train(net, train_loader, optimizer, criterion)
        y_prediction = test(net, test_loader).detach().cpu().numpy()
        print(
            f"Train epoch: {epoch:>3}\t Loss: {loss:.4f}\t Accuracy: {accuracy(y_test, y_prediction):.2f}"
        )

    torch.save(net.state_dict(), "./model/model.pth")


if __name__ == "__main__":
    main()
