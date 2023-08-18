from typing import Tuple

import numpy as np
import numpy.typing as npt
from AbstractDecorator import AbstractDecorator
from AbstractImage import AbstractImage


def add_blackbox(
    img: np.ndarray,
    width: Tuple[int, int] = (50, 150),
    height: Tuple[int, int] = (30, 100),
):
    """
    ---
    Attributes:
    * img (numpy.ndarray): input image
    ---
    Returns:
    * modified image (numpy.ndarray)
    """
    # if np.random.rand() < 0.5:
    #     return img
    h, w = img.shape
    blackbox_w = np.random.randint(width[0], width[1] + 1)
    blackbox_h = np.random.randint(height[0], height[1] + 1)

    blackbox_x = np.random.randint(0, w)
    blackbox_y = np.random.randint(0, h)

    blackbox_w = min(blackbox_w, w - blackbox_x)
    blackbox_h = min(blackbox_h, h - blackbox_y)

    img[
        blackbox_y : blackbox_y + blackbox_h,
        blackbox_x : blackbox_x + blackbox_w,
    ] = 0

    return img


class BlackBox(AbstractDecorator):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def __init__(
        self,
        component: AbstractImage,
        width: Tuple[int, int] = (5, 5),
        height: Tuple[int, int] = (320, 240),
    ) -> None:
        super().__init__(component)
        self.width = width
        self.height = height

    def generate(self) -> npt.NDArray[np.uint8]:
        img = self.component.generate()
        return add_blackbox(img, self.width, self.height)
