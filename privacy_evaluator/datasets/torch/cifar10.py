import torchvision
import torchvision.transforms as transforms

from .torch import TorchDataset


class TorchCIFAR10(TorchDataset):
    """PyTorch CIFAR10 dataset class."""

    TORCH_MODULE = torchvision.datasets.CIFAR10
    DATASET_SIZE = {"train": 50000, "test": 10000}
    INPUT_SHAPE = (3, 32, 32)
    N_CLASSES = 10
    NORMALIZE_VALUES = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    TRANSFORMERS = {
        "default": {
            "train": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        },
        "training": {
            "train": transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            ),
        },
    }
