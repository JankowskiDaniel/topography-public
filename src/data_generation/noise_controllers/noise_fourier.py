import os
from typing import Optional
import glob
import cv2
import pandas as pd
import numpy as np
import numpy.typing as npt
import pkg_resources

from src.data_generation.image.image_interface import AbstractGenerator
from src.data_generation.noise_controllers.decorator import NoiseController


def count_available_noises(path_fourier_noise: str) -> int:
    return len(
        [
            name
            for name in os.listdir(path_fourier_noise)
            if (os.path.isfile(os.path.join(path_fourier_noise, name))
            and not name == ".gitkeep")
        ]
    )

def get_available_noises(path_fourier_noise: str) -> int:

    paths = sorted(glob.glob(os.path.join(path_fourier_noise, "noise*")))
    return paths


def add_noise_amplitude(
    pure_img: npt.NDArray[np.uint8],
    noise_file_path: str = "",
) -> npt.NDArray[np.uint8]:
    """Generate the image. In case of generating single image
    (using this function) you don't have to pass path_fourier_noise
    argument only if you use this code as a package.
    If you didn't install it via pip, you have to pass
    the argument path_fourier_noise. It is assumed that noise images
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
    :param path_fourier_noise: path to noise dataset, defaults to None
    :type path_fourier_noise: str, optional
    :param noise_file_index: Index (filename) of noise image used
    to generate image, optional
    :type noise_file_index: int, optional
    :return: 2D array which represents an image
    :rtype: np.array
    """

    # TODO For this moment I do not have access to generated noise images,
    # so I created one half black half white
    noise_image = cv2.imread(noise_file_path)

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

def add_noise_frequency(
    pure_img: npt.NDArray[np.uint8],
    pass_value: int = 4,
    noise_file_path: str = "",
) -> npt.NDArray[np.uint8]:
    """Generate the image. In case of generating single image
    (using this function) you don't have to pass path_fourier_noise
    argument only if you use this code as a package.
    If you didn't install it via pip, you have to pass
    the argument path_fourier_noise. It is assumed that noise images
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
    :param path_fourier_noise: path to noise dataset, defaults to None
    :type path_fourier_noise: str, optional
    :param noise_file_index: Index (filename) of noise image used
    to generate image, optional
    :type noise_file_index: int, optional
    :return: 2D array which represents an image
    :rtype: np.array
    """

    noise = pd.read_csv(noise_file_path, header=None).to_numpy().astype(complex)
    
    fft_real_image = np.fft.fftshift(np.fft.fft2(pure_img))
    row, col = pure_img.shape
    center_row, center_col = row // 2, col // 2

    fft_real_image[center_row - pass_value:center_row + pass_value, 
                    center_col - pass_value:center_col + pass_value] = \
        (fft_real_image[center_row - pass_value:center_row + pass_value, 
                    center_col - pass_value:center_col + pass_value] + noise) / 2
    
    img = abs(np.fft.ifft2(fft_real_image)).clip(0,255)

    return img


class FourierController(NoiseController):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def __init__(
        self, 
        domain: str = "",
        path_fourier_noise_freq: str = "",
        path_fourier_noise_ampl: str = "",
        pass_value: int = 4,
    ) -> None:
        self.domain = domain
        self.path_fourier_noise_freq = path_fourier_noise_freq
        self.path_fourier_noise_ampl = path_fourier_noise_ampl
        self.pass_value = pass_value

    def _set_additional_parameters(self, num_images: int) -> None:
        if self.domain == "ampl":
            path_noise = self.path_fourier_noise_ampl
        else:
            path_noise = self.path_fourier_noise_freq

        self.available_noises = get_available_noises(
            path_fourier_noise=path_noise
        )
        self.choosen_noises = np.random.choice(
            self.available_noises, num_images,
            replace=True
        )
        self.noise_index = 0

    def generate(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if self.domain == "ampl":
            noised_image = add_noise_amplitude(
                img,
                noise_file_path=self.choosen_noises[self.noise_index],
            )
        else:
            noised_image = add_noise_frequency(
                img,
                pass_value=self.pass_value,
                noise_file_path=self.choosen_noises[self.noise_index],
            )
        self.noise_index += 1
        return noised_image
