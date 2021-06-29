from .data_utils import dataset_downloader

import numpy as np
from copy import deepcopy


BOX_LEN = 3  # default side length of the box we use for mask-adaptation


def images_adaptation(
    images: np.ndarray, adaptation: str = "mask", **kwargs
) -> np.ndarray:
    """
    Apply a specific adaptation on each image in `images`.

    :params images: The original images of shape [N, H, W, D].
    :params adaptation: The type of adaptation, so far "mask" is supported.
    :params **kwargs: optional parameters for the specified adaptation.

    Examples:
        `images_adaptation(images, 'mask', box_len=5)`: Apply mask-adaptation with box \
            of side length 5.
    """
    supported_adaptations = [
        "mask",
    ]
    assert len(images.shape) == 4
    assert adaptation in supported_adaptations

    if adaptation == "mask":
        return (
            _mask_images(images)
            if "box_len" not in kwargs
            else _mask_images(images, kwargs["box_len"])
        )


def _mask_images(images: np.ndarray, box_len: int = BOX_LEN) -> np.ndarray:
    """
    Mask each image in `images` with a white-colored box of side length `box_len`

    :params images: The original images of shape [N, H, W, D].
    :params box_len: The side length of the masking box, to be `min(H, W)` top
    """
    masked_images = deepcopy(images)
    for image in masked_images:
        _mask_image(image, box_len)
    return masked_images


def _mask_image(image: np.ndarray, box_len: int = BOX_LEN):
    """
    Mask one `image` with a white-colored box of side length `box_len`

    :params image: The original image of shape [H, W, D].
    :params box_len: The side length of the masking box, to be `min(H, W)` top
    """
    # depth = 1 for gray-scale images (e.g. MNIST) and 3 for RGB images
    height, width, _ = image.shape
    assert box_len <= height and box_len <= width

    start_x = np.random.randint(0, width - box_len + 1)
    start_y = np.random.randint(0, height - box_len + 1)

    image[start_x : start_x + box_len, start_y : start_y + box_len] = 255
