import os
import random
import glob
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import pkg_resources
from tqdm import tqdm
from src.data_generation.datasets.generate_utils import (
    save2directory,
    save2zip,
)

from src.data_generation.datasets.generate_utils import _count_available_raw_images


def _check_args(num_images: int):
    if num_images <= 0:
        raise ValueError("Number of generated images must be grater than 0.")


def _generate_noise_image_amplitude_domain(
    size: Tuple[int, int] = (640, 480),
    used_raw_image: str ='',
    pass_value: int = 10,
) -> np.array:
    """Generate single noise image. The function is given indices (filenames) of raw images used for extraction. In case of generating single noise image (by this function)
    there is no need to parse argument with raw images filenames, the list of raw images will be automatically produced

    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param used_raw_image: Index of raw image used for extracting noise, defaults to None
    :type used_raw_image: np.array, optional
    :param seed: Set to some integer value to generate the same noise every function run, defaults to None
    :type seed: int, optional
    :param pass_value:
    :type pass_value: int, optional
    :return: Noise image
    :rtype: np.array
    """

    img = cv2.imread(used_raw_image)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, 0]

    ## TODO: teraz tutaj trzeba zrobić fouriera
    rgb_fft = np.fft.fftshift(np.fft.fft2(img))

    row, col = img.shape
    center_row, center_col = row // 2, col // 2
    mask = np.zeros((row, col), np.uint8)
    mask[center_row - pass_value:center_row + pass_value, 
         center_col - pass_value:center_col + pass_value] = 1
    rgb_fft = rgb_fft * mask
    img = abs(np.fft.ifft2(rgb_fft)).clip(0,255)

    raw_img = used_raw_image.split("/")[-1].split(".")[0]
    return raw_img, img.astype(np.uint8)

def _generate_noise_image_frequency_domain(
    size: Tuple[int, int] = (640, 480),
    used_raw_image: str ='',
    pass_value: int = 10,
) -> np.array:
    """Generate single noise image. The function is given indices (filenames) of raw images used for extraction. In case of generating single noise image (by this function)
    there is no need to parse argument with raw images filenames, the list of raw images will be automatically produced

    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param used_raw_image: Index of raw image used for extracting noise, defaults to None
    :type used_raw_image: np.array, optional
    :param seed: Set to some integer value to generate the same noise every function run, defaults to None
    :type seed: int, optional
    :param pass_value:
    :type pass_value: int, optional
    :return: Noise image
    :rtype: np.array
    """

    img = cv2.imread(used_raw_image)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, 0]

    ## TODO: teraz tutaj trzeba zrobić fouriera
    rgb_fft = np.fft.fftshift(np.fft.fft2(img))

    row, col = img.shape
    center_row, center_col = row // 2, col // 2
    rgb_fft = rgb_fft[center_row - pass_value:center_row + pass_value, 
         center_col - pass_value:center_col + pass_value]
    # img = abs(np.fft.ifft2(rgb_fft)).clip(0,255)

    raw_img = used_raw_image.split("/")[-1].split(".")[0]
    return raw_img, rgb_fft


def generate_fourier_noise_dataset(
    path: str,
    size: Tuple[int, int] = (640, 480),
    num_images: int = 50,
    pass_value: int = 10,
    raw_epsilons_path: str = None,
    zipfile: bool = False,
    zip_filename: str = None,
    seed: int = None,
    domain: str = "amplitude"
) -> None:
    """Create noise dataset. The way how seed work is as follows:
        1. We count how many raw images are available. If path_to_raw was not passed (available only when you installed package via pip)
            then this number is equal to 300 (package has 300 raw images installed, something like example datasets in scikit-learn).
            In case when path to raw was passed, then we count how many files are in that directory, therefore there should be only raw images and any other files.
            Moreover, raw images in that directory should be named like in the original dataset, always from 0-N with zeros at the beginning of the filename, so that
            in result filename contains always 5 chars. This is becuase for know the code assumes that for example if directory contains 200 images, then we have there
            200 images named from 00000.png, 00001.png, ..., 00199.png. IT WON'T WORK IN CASE OF ANY OTHER FILES ORGANIZATION.
        2. Numpy generates random list of images indices which will be used to extract noise, if you use seed, this list will be same all the time.

    :param path: Path where images should be stored
    :type path: str
    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param num_images: Number of images that will be created, defaults to 50
    :type num_images: int, optional
    :param pass_value:
    :type pass_value: int, optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param zipfile: Set to True if you want to save noise in zipfile, defaults to False
    :type zipfile: bool, optional
    :param zip_filename: Name of the zipfile, should be passed only if zipfile=True, defaults to None
    :type zip_filename: str, optional
    :param seed: Set seed to obtain the same result, defaults to None
    :type seed: int, optional
    """
    _check_args(num_images)

    if seed is not None:
        np.random.seed(seed)

    eps_est = pd.read_csv(os.path.join(raw_epsilons_path, "raw_epsilons.csv"))
    selected_raw_images = []
    for eps in np.arange(0,1,1/num_images):
        eps = round(eps, 3)
        paths = eps_est[eps_est.epsilon == eps].path
        chosen_path = np.random.choice(paths, 1)
        selected_raw_images.append(chosen_path[0])
    selected_raw_images = np.array(selected_raw_images)

    if domain == "amplitude":
        for img in tqdm(range(num_images)):
            raw_filename, noise_image = _generate_noise_image_amplitude_domain(
                size=size, 
                used_raw_image=selected_raw_images[img], 
                pass_value=pass_value
            )
            if zipfile:
                save2zip(
                    noise_image,
                    img_filename=f"noise_{img}_{raw_filename}.png",
                    filename=zip_filename,
                    path=path,
                )
            else:
                save2directory(noise_image, img_filename=f"noise_{img}_{raw_filename}.png", path=path)
    
    else:
        for img in tqdm(range(num_images)):
            raw_filename, noise = _generate_noise_image_frequency_domain(
                size=size, 
                used_raw_image=selected_raw_images[img], 
                pass_value=pass_value
            )
            freq = pd.DataFrame(noise)
            freq.to_csv(os.path.join(path, f"noise_{img}_{raw_filename}.csv"), header=None, index=None)
            


if __name__ == "__main__":
    generate_fourier_noise_dataset(
        path="data/fourier_noise/steel",
        size=(640, 480),
        num_images=2,
        path_to_raw="data/raw/steel",
        seed=42,
        pass_value=4,
        domain="frequency"
    )
