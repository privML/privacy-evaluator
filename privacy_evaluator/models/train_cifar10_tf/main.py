import tensorflow as tf
import numpy as np
from data import dataset_downloader, new_dataset_from_size_dict
from train import trainer


model_name = "FCNeuralNet"  # "ResNet50"

# hyper-parameters
num_epochs = 15
batch_size = 500
learning_rate = 0.002
weight_decay = 0
dropout = 0

# put your designed sample distribution here
# each line corresponds to an experiment
size_dicts = [
    {0: 5000, 1: 5000},
    {0: 5000, 1: 4000},
    {0: 5000, 1: 3000},
    {0: 5000, 1: 2000},
    {0: 5000, 1: 1000},
    {0: 5000, 1: 500},
    {0: 3000, 1: 1000, 2: 500},
]

if __name__ == "__main__":
    # set device
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")

    train_dataset, test_dataset = dataset_downloader("CIFAR10")
    for size_dict in size_dicts:
        train_set, test_set = new_dataset_from_size_dict(
            train_dataset, test_dataset, size_dict
        )
        test_acc = trainer(
            train_set=train_set,
            test_set=test_set,
            size_dict=size_dict,
            model=model_name,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout=dropout,
        )
        print(size_dict, test_acc)