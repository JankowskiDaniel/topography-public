import os
import cv2
import numpy as np
import numpy.typing as npt
import pkg_resources
from src.data_generation.noise_controllers.decorator import NoiseController


def count_available_noises(path_average_noise: str) -> int:
    return len(
        [
            name
            for name in os.listdir(path_average_noise)
            if (
                os.path.isfile(os.path.join(path_average_noise, name))
                and not name == ".gitkeep"
            )
        ]
    )


def add_noise(
    pure_img: npt.NDArray[np.uint8],
    path_average_noise: str | None = None,
    noise_file_index: int = 0,
) -> npt.NDArray[np.uint8]:
    """Generate the image. In case of generating single image
    (using this function) you don't have to pass path_average_noise
    argument only if you use this code as a package.
    If you didn't install it via pip, you have to pass
    the argument path_average_noise. It is assumed that noise images
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
    :param path_average_noise: path to noise dataset, defaults to None
    :type path_average_noise: str, optional
    :param noise_file_index: Index (filename) of noise image used
    to generate image, optional
    :type noise_file_index: int, optional
    :return: 2D array which represents an image
    :rtype: np.array
    """

    if path_average_noise:
        noise_image_filename = f"{path_average_noise}/{noise_file_index}.png"
    else:
        noise_image_filename = pkg_resources.resource_filename(
            __name__, f"/samples/average_noise/{noise_file_index}.png"
        )

    # TODO For this moment I do not have access to generated noise images,
    # so I created one half black half white
    noise_image = cv2.imread(noise_image_filename)  # type: ignore

    if noise_image.shape[:2] != pure_img.shape:
        noise_image = cv2.resize(  # type: ignore
            noise_image, pure_img.shape, interpolation=cv2.INTER_AREA
        )

    noise_image = noise_image[:, :, 0]
    noise_mean = np.mean(noise_image)  # type: ignore
    difference = -(noise_image - noise_mean)  # type: ignore

    noised_image = pure_img - difference
    noised_image = np.clip(noised_image, 0, 255)
    return noised_image.astype(np.uint8)


class AverageController(NoiseController):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def __init__(self, path_average_noise: str = "") -> None:
        self.path_average_noise = path_average_noise

    def _set_additional_parameters(self, num_images: int) -> None:
        self.num_available_noises = count_available_noises(
            path_average_noise=self.path_average_noise
        )
        self.choosen_noises = np.random.randint(
            0, self.num_available_noises, num_images
        )
        self.noise_index = 0

    def generate(
        self, img: npt.NDArray[np.uint8], epsilon: float
    ) -> npt.NDArray[np.uint8]:
        noised_image = add_noise(
            img,
            path_average_noise=self.path_average_noise,
            noise_file_index=self.choosen_noises[self.noise_index],
        )
        self.noise_index += 1
        return noised_image
