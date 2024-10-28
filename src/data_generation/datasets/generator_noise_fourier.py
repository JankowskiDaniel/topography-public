import os
from typing import Literal
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from src.data_generation.datasets.generate_utils import (
    save2directory,
    save2zip,
)


def _check_args(num_images: int, domain: str) -> None:
    if num_images <= 0:
        raise ValueError("Number of generated images must be grater than 0.")
    if domain not in ["amplitude", "frequency"]:
        raise ValueError("Domain must be either 'amplitude' or 'frequency'.")


def _generate_noise_image_amplitude_domain(
    size: tuple[int, int] = (640, 480),
    used_raw_image: str = "",
    pass_value: int = 10,
) -> tuple[str, npt.NDArray[np.uint8]]:
    """Generate single noise image. The function is given indices (filenames)
    of raw images used for extraction. In case of generating single noise
    image (by this function) there is no need to parse argument with raw
    images filenames, the list of raw images will be automatically produced

    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param used_raw_image: Index of raw image used for extracting noise,
    defaults to None
    :type used_raw_image: np.array, optional
    :param seed: Set to some integer value to generate the same noise every
    function run, defaults to None
    :type seed: int, optional
    :param pass_value:
    :type pass_value: int, optional
    :return: Noise image
    :rtype: np.array
    """

    img = cv2.imread(used_raw_image)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, 0]

    # TODO: teraz tutaj trzeba zrobić fouriera
    rgb_fft = np.fft.fftshift(np.fft.fft2(img))

    row, col = img.shape
    center_row, center_col = row // 2, col // 2
    mask = np.zeros((row, col), np.uint8)
    mask[
        center_row - pass_value: center_row + pass_value,
        center_col - pass_value: center_col + pass_value,
    ] = 1
    rgb_fft = rgb_fft * mask
    img = abs(np.fft.ifft2(rgb_fft)).clip(0, 255)

    raw_img = used_raw_image.split("/")[-1].split(".")[0]
    return raw_img, img.astype(np.uint8)


def _generate_noise_image_frequency_domain(
    size: tuple[int, int] = (640, 480),
    used_raw_image: str = "",
    pass_value: int = 10,
) -> tuple[str, npt.NDArray[np.uint8]]:
    """Generate single noise image. The function is given indices (filenames)
    of raw images used for extraction. In case of generating single noise
    image (by this function) there is no need to parse argument with raw
    images filenames, the list of raw images will be automatically produced.

    :param size: Size of generated noise image. Defaults to (640, 480).
    :type size: Tuple[int, int], optional
    :param path_to_raw: Path to raw images, defaults to None
    :type path_to_raw: str, optional
    :param used_raw_image: Index of raw image used for extracting noise,
    defaults to None
    :type used_raw_image: np.array, optional
    :param seed: Set to some integer value to generate the same noise every
    function run, defaults to None
    :type seed: int, optional
    :param pass_value:
    :type pass_value: int, optional
    :return: Noise image
    :rtype: np.array
    """

    img = cv2.imread(used_raw_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # Perform Fourier transform and shift zero frequency component to the center
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    
    # Create a mask to keep the low frequencies
    row, col = img.shape
    mask = np.zeros((row, col), dtype=np.uint8)
    center_row, center_col = row // 2, col // 2
    mask[center_row - pass_value:center_row + pass_value, center_col - pass_value:center_col + pass_value] = 1
    
    # Apply the mask to the frequency domain representation
    img_fft_filtered = img_fft * mask
    
    # Perform the inverse Fourier transform to get the image back in the spatial domain
    img_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft_filtered)))
    
    # Normalize the result to the range [0, 255] and convert to uint8
    img_filtered_normalized = cv2.normalize(img_filtered, None, 0, 255, cv2.NORM_MINMAX)
    img_filtered_normalized = np.uint8(img_filtered_normalized)
    
    # Extract the raw image name without extension
    raw_img = used_raw_image.split("/")[-1].split(".")[0]
    
    return raw_img, img_filtered_normalized


def generate_fourier_noise_dataset(
    path: str,
    raw_epsilons_path: str,
    size: tuple[int, int] = (640, 480),
    num_images: int = 50,
    pass_value: int = 10,
    zipfile: bool = False,
    zip_filename: str = "dataset.zip",
    seed: int | None = None,
    domain: Literal["amplitude", "frequency"] = "amplitude",
) -> None:
    """Generate dataset of noise images in Fourier domain.

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
    :param zipfile: Set to True if you want to save noise in zipfile,
    defaults to False
    :type zipfile: bool, optional
    :param zip_filename: Name of the zipfile, should be passed only if
    zipfile=True, defaults to None
    :type zip_filename: str, optional
    :param seed: Set seed to obtain the same result, defaults to None
    :type seed: int, optional
    """
    _check_args(num_images, domain)

    if seed is not None:
        np.random.seed(seed)

    eps_est = pd.read_csv(os.path.join(raw_epsilons_path, "raw_epsilons.csv"))

    # round epsilons to 3 decimal places
    eps_est["epsilon"] = eps_est["epsilon"].apply(lambda x: round(x, 3))
    threshold = 0.01

    selected_raw_images = []
    for eps in np.arange(0, 1, 1 / num_images):
        eps = round(eps, 3)

        # filter raw images where |epsilon - eps| < 0.01
        eps_est["diff"] = abs(eps_est.epsilon - eps)
        temp = eps_est[eps_est["diff"] < threshold]

        # take a list of filenames
        available_raw_images = temp.filename.to_list()

        chosen_filename = np.random.choice(available_raw_images)
        chosen_path = os.path.join(raw_epsilons_path, chosen_filename)
        selected_raw_images.append(chosen_path)

    gen_funct = {
        "amplitude": _generate_noise_image_amplitude_domain,
        "frequency": _generate_noise_image_frequency_domain,
    }


    for img in tqdm(range(num_images)):
        raw_path, noise_image = gen_funct[domain](  # noqa: E501
            size=size,
            used_raw_image=selected_raw_images[img],
            pass_value=pass_value,
        )
        raw_filename = os.path.basename(raw_path)
        if zipfile:
            save2zip(
                noise_image,
                img_filename=f"noise_{img}_{raw_filename}.png",
                filename=zip_filename,
                path=path,
            )
        else:
            save2directory(
                noise_image,
                img_filename=f"noise_{img}_{raw_filename}.png",
                path=path,
            )


if __name__ == "__main__":
    path_repo = os.path.dirname(os.getcwd())
    path = os.path.join(path_repo, "data", "fourier_noise", "freq")
    path_raw = os.path.join(path_repo, "data", "raw", "steel", "1channel")
    num_noise_images = 2000
    generate_fourier_noise_dataset(
        path=path,
        raw_epsilons_path=path_raw,
        num_images=num_noise_images,
        seed=23,
        pass_value=4,
        domain="frequency"
    )
