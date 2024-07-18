import os
import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from src.data_generation.datasets.generate_utils import (
    save2directory,
    save2zip,
)


def _check_args(num_images: int):
    if num_images <= 0:
        raise ValueError("Number of generated images must be grater than 0.")


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

    img = cv2.imread(used_raw_image)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)[:, :, 0]

    # TODO: teraz tutaj trzeba zrobić fouriera
    rgb_fft: npt.NDArray[np.uint8] = np.fft.fftshift(np.fft.fft2(img))

    row, col = img.shape
    center_row, center_col = row // 2, col // 2
    rgb_fft = rgb_fft[
        center_row - pass_value: center_row + pass_value,
        center_col - pass_value: center_col + pass_value,
    ]
    # img = abs(np.fft.ifft2(rgb_fft)).clip(0,255)

    raw_img = used_raw_image.split("/")[-1].split(".")[0]
    return raw_img, rgb_fft


def generate_fourier_noise_dataset(
    path: str,
    raw_epsilons_path: str,
    size: tuple[int, int] = (640, 480),
    num_images: int = 50,
    pass_value: int = 10,
    zipfile: bool = False,
    zip_filename: str = "dataset.zip",
    seed: int | None = None,
    domain: str = "amplitude",
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
    _check_args(num_images)

    if seed is not None:
        np.random.seed(seed)

    eps_est = pd.read_csv(os.path.join(raw_epsilons_path, "raw_epsilons.csv"))
    selected_raw_images = []
    for eps in np.arange(0, 1, 1 / num_images):
        eps = round(eps, 3)
        paths = eps_est[eps_est.epsilon == eps].path
        chosen_path = np.random.choice(paths, 1)
        selected_raw_images.append(chosen_path[0])
    # selected_raw_images = np.array(selected_raw_images)  # type: ignore

    if domain == "amplitude":
        for img in tqdm(range(num_images)):
            raw_filename, noise_image = _generate_noise_image_amplitude_domain(  # noqa: E501
                size=size,
                used_raw_image=selected_raw_images[img],
                pass_value=pass_value,
            )
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

    else:
        for img in tqdm(range(num_images)):
            raw_filename, noise = _generate_noise_image_frequency_domain(
                size=size,
                used_raw_image=selected_raw_images[img],
                pass_value=pass_value,
            )
            freq = pd.DataFrame(noise)
            freq.to_csv(
                os.path.join(path, f"noise_{img}_{raw_filename}.csv"),
                header=None,
                index=None,
            )


if __name__ == "__main__":
    generate_fourier_noise_dataset(
        path="data/fourier_noise/steel",
        size=(640, 480),
        num_images=2,
        raw_epsilons_path="data/raw/steel",
        seed=42,
        pass_value=4,
        domain="frequency",
    )
