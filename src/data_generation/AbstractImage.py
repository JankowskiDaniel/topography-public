from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt

from src.data_generation._models import ImageDetails


def _check_image(
    size: Tuple[int, int],
    epsilon: float,
    ring_center: Tuple[int, int],
    brightness: Tuple[int, int],
    noise_file_index,
) -> None:
    width, height = size
    width_ring_center, height_ring_center = ring_center
    min_brightness, max_brightness = brightness
    _ = ImageDetails(  # type: ignore
        width=width,
        height=height,
        epsilon=epsilon,
        ring_center_width=width_ring_center,
        ring_center_height=height_ring_center,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        used_noise=noise_file_index,
    )


def generate_pure_image(
    size: Tuple[int, int],
    epsilon: float,
    ring_center: Tuple[int, int],
    brightness: Tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Generate pure image

    :param size: Size of the image (width, height)
    :type size: Tuple[int, int]
    :param epsilon: Epsilon value
    :type epsilon: float
    :param ring_center: Position of the central ring
    :type ring_center: Tuple[int, int]
    :param brightness: Range of brightness
    :type brightness: Tuple[int, int]
    :return: 2D array which represents pure image
    :rtype: np.array
    """
    _check_image(size, epsilon, ring_center, brightness, noise_file_index=5)
    width, height = size
    min_brightness, max_brightness = brightness

    mean_brightness = (min_brightness + max_brightness) / 2
    diff_brightness = max_brightness - mean_brightness

    diff_betweeen_rings_denominator = 6.07
    diff_between_rings = width * width / diff_betweeen_rings_denominator

    width_ring_center, height_ring_center = ring_center

    y, x = np.indices([height, width])
    img = mean_brightness + (
        diff_brightness
        * np.cos(
            2
            * np.pi
            * (
                1.0
                - epsilon
                + (
                    (
                        np.power((x - width_ring_center) * 2, 2)
                        + np.power((y - height_ring_center) * 2, 2)
                    )
                    / diff_between_rings
                )
            )
        )
    )
    img = img.astype(np.uint8)
    return img


# https://refactoring.guru/design-patterns/decorator/python/example
class AbstractImage(ABC):
    """
    The base Component interface defines generates that can be altered by
    decorators.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (250, 250),
        epsilon: float = 0.1,
        ring_center: Tuple[int, int] = (130, 130),
        brightness: Tuple[int, int] = (100, 210),
    ):
        print(size, ring_center)
        self.img = generate_pure_image(size, epsilon, ring_center, brightness)
        pass

    @abstractmethod
    def generate(self) -> npt.NDArray[np.uint8]:
        raise NotImplementedError
