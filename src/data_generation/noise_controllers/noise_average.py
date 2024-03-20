import os
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import pkg_resources

from src.data_generation.image.image_interface import AbstractGenerator
from src.data_generation.noise_controllers.decorator import NoiseController


def count_available_noises(noise_path: str) -> int:
    return len(
        [
            name
            for name in os.listdir(noise_path)
            if (os.path.isfile(os.path.join(noise_path, name))
            and not name == ".gitkeep")
        ]
    )


def add_noise(
    pure_img: npt.NDArray[np.uint8],
    noise_path: Optional[str] = None,
    noise_file_index: int = 0,
) -> npt.NDArray[np.uint8]:
    """Generate the image. In case of generating single image
    (using this function) you don't have to pass noise_path
    argument only if you use this code as a package.
    If you didn't install it via pip, you have to pass
    the argument noise_path. It is assumed that noise images
    in you directory are named with integers started from
    0 to N, e.g. you have 5 noise image in you directory,
    so files are named: 0.png, 1.png, ..., 4.png.
    Noise_file_index argument points to the noise image, by default,
    it is set to 0. In case of generating single image this
    implementation might now look reasonable, however in case of
    generating whole dataset you don't have to care about anything.
    The code is just written this way that it's easier to use it
    for generating whole dataset in case of single images.

    :param pure_img: generated image without any noise
    :param noise_path: path to noise dataset, defaults to None
    :type noise_path: str, optional
    :param noise_file_index: Index (filename) of noise image used
    to generate image, optional
    :type noise_file_index: int, optional
    :return: 2D array which represents an image
    :rtype: np.array
    """

    if noise_path:
        noise_image_filename = f"{noise_path}/{noise_file_index}.png"
    else:
        noise_image_filename = pkg_resources.resource_filename(
            __name__, f"/samples/noise/{noise_file_index}.png"
        )

    # TODO For this moment I do not have access to generated noise images,
    # so I created one half black half white
    noise_image = cv2.imread(noise_image_filename)

    if noise_image.shape[:2] != pure_img.shape:
        noise_image = cv2.resize(
            noise_image, pure_img.shape, interpolation=cv2.INTER_AREA
        )

    noise_image = noise_image[:, :, 0]
    noise_mean = np.mean(noise_image)
    difference = -(noise_image - noise_mean)

    noised_image = pure_img - difference
    noised_image = np.clip(noised_image, 0, 255)
    return noised_image.astype(np.uint8)


class AverageController(NoiseController):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def __init__(
        self, noise_path: str = ""
    ) -> None:
        self.noise_path = noise_path

    def _set_additional_parameters(self, num_images: int) -> None:
        self.num_available_noises = count_available_noises(
            noise_path=self.noise_path
        )
        self.choosen_noises = np.random.randint(
            0, self.num_available_noises, num_images
        )
        self.noise_index = 0

    def generate(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        noised_image = add_noise(
            img,
            noise_path=self.noise_path,
            noise_file_index=self.choosen_noises[self.noise_index],
        )
        self.noise_index += 1
        return noised_image
