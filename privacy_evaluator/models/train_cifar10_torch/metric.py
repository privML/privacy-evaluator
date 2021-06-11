import torch


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy given the predicted probability distribution and label.

    Args:
        outputs: Model output given as tensor of shape `[batch_size, num_classes]`.
        labels: True class given as tensor of shape `[batch_size,]`.

    Returns:
        The accuracy for this batch.
    """
    assert outputs.size(0) == labels.size(0)
    _, pred = torch.max(outputs.data, 1)
    correct = (pred == labels).sum()
    total = labels.size(0)
    acc = 1.0 * correct / total
    return float(acc.item())
