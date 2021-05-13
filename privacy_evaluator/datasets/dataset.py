import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple


class Dataset:

    DATA_ROOT = './data'

    @classmethod
    def pytorch_loader(cls, train_batch_size: int = 128, test_batch_size: int = 128, one_hot_encode: bool = False) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError("Method 'pytorch_loader()' needs to be implemented in subclass.")

    @classmethod
    def numpy(cls, one_hot_encode: bool = False) -> np.ndarray:
        raise NotImplementedError("Method 'numpy()' needs to be implemented in subclass.")

    @classmethod
    def tensorflow_loader(cls, train_batch_size: int = 128, test_batch_size: int = 128, one_hot_encode: bool = False):
        raise NotImplementedError("Method 'tensorflow_loader()' needs to be implemented in subclass.")

    @classmethod
    def _one_hot_encode_pytorch(cls, y: torch.Tensor, n_classes: int) -> torch.Tensor:
        y_one_hot_encoded = torch.zeros(y.shape[0], n_classes)
        y_one_hot_encoded[torch.arange(y.shape[0]), y] = 1
        return y_one_hot_encoded

    @classmethod
    def _one_hot_encode(cls, y: np.ndarray, n_classes: int) -> np.ndarray:
        y_one_hot_encoded = np.zeros((y.shape[0], n_classes))
        y_one_hot_encoded[np.arange(y.shape[0]), y] = 1
        return y_one_hot_encoded
