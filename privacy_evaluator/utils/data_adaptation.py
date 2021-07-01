from .data_utils import dataset_downloader

import numpy as np
from copy import deepcopy


BOX_LEN = 3  # default side length of the box we use for mask-adaptation


def images_adaptation(images: np.ndarray, adaptation: str, **kwargs) -> np.ndarray:
    """
    Apply a specific adaptation on each image in `images`.

    :params images: The original images of shape [N, H, W, D].
    :params adaptation: The type of adaptation, so far "mask" is supported.
    :params **kwargs: Optional parameters for the specified adaptation.
    :return: The adapted images.

    Optional params:
    :params box_len: Involved when `adaptation` is "mask", the side length of masking boxes.
    :params mean: Involved when `adaptation` is "random_noise", the mean of the added noise.
    :params mean: Involved when `adaptation` is "random_noise", the standard deviation of the added noise.

    Examples:
        `images_adaptation(images, 'mask', box_len=5)`: Apply mask-adaptation with box \
            of side length 5.
    """
    supported_adaptations = ["mask", "random_noise"]
    assert len(images.shape) == 4
    assert adaptation in supported_adaptations

    if adaptation == "mask":
        return _mask_images(images, **kwargs)
    elif adaptation == "random_noise":
        return _random_noise_images(images, **kwargs)


def _mask_images(images: np.ndarray, box_len: int = BOX_LEN, **kwargs) -> np.ndarray:
    """
    Mask each image in `images` with a white-colored box of side length `box_len`

    :params images: The original images of shape [N, H, W, D].
    :params box_len: The side length of the masking box, to be `min(H, W)` top.
    :params kwargs: Optional params to make the function run also wenn unexpected \
        params are passed from `images_adaptation()`
    :return: The masked images.
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


def _random_noise_images(
    images: np.ndarray, mean: float = 0.0, std: float = 50.0, **kwargs
):
    """
    Create dataset where noise from normal distribution with mean and standard deviation is added to images.

    :params image: The original image of shape [H, W, D].
    :params mean: mean for the distribution from which noise is computed
    :params std: standard deviation for the distribution from which noise is computed
    :params kwargs: optional params to make the function run also wenn unexpected \
        params are passed from `images_adaptation()`
    """

    # Create noise with shape of image
    noise = np.random.normal(mean, std, images.shape)

    # Add noise to images and convert both to int16 to allow negative noise values
    images = np.int16(images) + np.int16(noise)

    # Clip image values to range [0,255]
    images = np.uint8(np.clip(images, 0, 255))

    return images
