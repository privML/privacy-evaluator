import numpy as np
from copy import deepcopy
import logging


BOX_LEN = 3  # default side length of the box we use for mask-adaptation
BRIGHTNESS = 50  # default value to adjust brightness pictures
MEAN = 0.0  # default mean for the distribution from which noise is computed
STD = (
    50.0  # default standard deviation for the distribution from which noise is computed
)


def images_adaptation(images: np.ndarray, adaptation: str, **kwargs) -> np.ndarray:
    """
    Apply a specific adaptation on each image in `images`.

    :params images: The original images of shape [N, H, W, D].
    :params adaptation: The type of adaptation.
    :params **kwargs: Optional parameters for the specified adaptation.
    :return: The adapted images.

    Optional params:
    :params box_len: Involved when `adaptation` is "mask", the side length of masking boxes.
    :params brightness: Involved when `adaptation` is "brightness", the amount the brightness should be raised or lowered
    :params mean: Involved when `adaptation` is "random_noise", the mean of the added noise.
    :params mean: Involved when `adaptation` is "random_noise", the standard deviation of the added noise.

    Examples:
        `images_adaptation(images, 'mask', box_len=5)`: Apply mask-adaptation with box \
            of side length 5.
    """
    supported_adaptations = ["mask", "random_noise", "brightness"]
    assert len(images.shape) == 4
    assert adaptation in supported_adaptations

    if adaptation == "mask":
        return (
            _mask_images(images)
            if "box_len" not in kwargs
            else _mask_images(images, kwargs["box_len"])
        )
    elif adaptation == "brightness":
        return (
            _brighten_images(images)
            if "brightness" not in kwargs
            else _brighten_images(images, kwargs["brightness"])
        )
    elif adaptation == "random_noise":
        return _random_noise_images(images, **kwargs)


def _mask_images(images: np.ndarray, box_len: int = BOX_LEN, **kwargs) -> np.ndarray:
    """
    Mask each image in `images` with a white-colored box of side length `box_len`

    :params images: The original images of shape [N, H, W, D].
    :params box_len: The side length of the masking box, to be `min(H, W)` top.
    :params kwargs: Optional params to make the function run also when unexpected \
        params are passed from `images_adaptation()`
    :return: The masked images.
    """
    # give warning if there are still unexpected parameters
    if kwargs:
        logging.warning(
            "Unexpected parameter(s) for mask adaptation: {}.".format(
                list(kwargs.keys())
            )
        )

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


def _brighten_images(images: np.ndarray, brightness: int = BRIGHTNESS) -> np.ndarray:
    """
    Adjust the brightness of all input images

    :params images: The original images of shape [H, W, D].
    :params brightness: The amount the brighness should be raised or lowered
    :return: Images with adjusted brightness
    """
    brighten_images = deepcopy(images)
    for image in brighten_images:
        _brighten_image(image, brightness)
    return brighten_images


def _brighten_image(image: np.ndarray, brightness: int):
    """
    Adjust the brightness of one image

    :params image: The original image of shape [H, W, D].
    :params brightness: The amount the brightness should be raised or lowered
    """
    height, width, _ = image.shape
    for x in range(height):
        for y in range(width):
            if image[x, y] < (0 - brightness):
                image[x, y] = 0
            elif image[x, y] > (255 - brightness):
                image[x, y] = 255
            else:
                image[x, y] = image[x, y] + brightness


def _random_noise_images(
    images: np.ndarray, mean: float = MEAN, std: float = STD, **kwargs
):
    """
    Create dataset where noise from normal distribution with mean and standard deviation is added to images.

    :params images: The original image of shape [H, W, D].
    :params mean: mean for the distribution from which noise is computed
    :params std: standard deviation for the distribution from which noise is computed
    :params kwargs: optional params to make the function run also when unexpected \
        params are passed from `images_adaptation()`
    """
    # give warning if there are still unexpected parameters
    if kwargs:
        logging.warning(
            "Unexpected parameter(s) for random noise adaptation: {}.".format(
                list(kwargs.keys())
            )
        )

    # Create noise with shape of image
    noise = np.random.normal(mean, std, images.shape)

    # Add noise to images and convert both to int16 to allow negative noise values
    images = np.int16(images) + np.int16(noise)

    # Clip image values to range [0,255]
    images = np.uint8(np.clip(images, 0, 255))

    return images
