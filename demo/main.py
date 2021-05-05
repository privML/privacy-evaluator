import torch
import torchvision
from torchvision import datasets, transforms
import os
from utils import dataset_downloader, new_dataset
from train import trainer

# hyper-parameters
num_epochs = 10
batch_size = 500
learning_rate = 0.001
weight_decay = 0.002
dropout = 0.3

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# choose dataset
train_dataset, test_dataset = dataset_downloader('CIFAR10')

# put your designed sample distribution here
# each line corresponds to an experiment
size_dicts = [
    {0: 5000, 1: 5000}, 
    {2: 5000, 3: 5000}
]

for size_dict in size_dicts:
    train_set, test_set = new_dataset(train_dataset, test_dataset, size_dict)
    num_classes = len(size_dict)
    class_encoding = {class_id: i 
        for i, (class_id, _) in enumerate(size_dict.items())}

    test_acc = trainer(
        train_set, test_set, 'ResNet', 
        batch_size, num_epochs, num_classes, 
        learning_rate, weight_decay, dropout, 
        device, class_encoding)
    print(size_dict, test_acc)