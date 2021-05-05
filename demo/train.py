import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import accuracy
from model import ResNet

def trainer(train_set, test_set, model='ResNet', 
            batch_size=100, num_epochs=10, num_classes=2, 
            learning_rate=0.001, weight_decay=0.002, dropout=0, 
            device='cuda', class_encoding=None):
    # create data-loaders
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              drop_last=True,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size, 
                             shuffle=False,
                             pin_memory=True)
    
    # set model, loss function and optimizer
    model = ResNet(num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    # start training
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            # here class encoding is necessary since we need the dimension
            # of one-hot encoding identical to the number of classes
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

        acc = sum(accuracies) / len(accuracies)
        best_acc = max(best_acc, acc)

    best_acc = best_acc.item()
    return float(round(best_acc, 4))
