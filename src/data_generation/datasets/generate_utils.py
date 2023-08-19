import os
from typing import Dict, List
from zipfile import ZipFile

import cv2
import numpy.typing as npt
import pandas as pd
from PIL import Image


def save2zip(
    img: npt.ArrayLike, img_filename: str, filename: str, path: str
) -> None:
    """Save image to the .zip file

    :param img: 2D array which represents an image
    :type img: np.array
    :param img_filename: Name of the image file
    :type img_filename: str
    :param filename: Name of the .zip file
    :type filename: str
    :param path: Path to the directory where .zip file is stored
    :type path: str
    """
    zip_path = filename
    if path:
        zip_path = path + filename

    _, encoded_image = cv2.imencode(".png", img)
    with ZipFile(zip_path, "a") as zip:
        zip.writestr(img_filename, encoded_image.tobytes())


def save2directory(img: npt.ArrayLike, img_filename: str, path: str) -> None:
    """Save image to the directory

    :param img: 2D array which represents an image
    :type img: np.array
    :param img_filename: Name of the image file
    :type img_filename: str
    :param path: Path to the output directory
    :type path: str
    """
    image = Image.fromarray(img)
    image.save(path + img_filename)


def _check_args(
    path: str, n_copies: int, epsilon_step: float, zipfile: bool, filename: str
) -> None:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Provided path: {path} is not directory.")
    if n_copies <= 0:
        raise ValueError("Number of created copies must be positive.")
    if epsilon_step < 0.0 or epsilon_step > 1.0:
        raise ValueError("Epsilon step must be in range <0.0 - 1.0>.")
    if zipfile:
        if not filename:
            raise Exception(
                "In case of exporting dataset to zipfile, filename must be provided."  # noqa: 501
            )
        if filename[-4:] != ".zip":
            raise Exception(
                'Provide filename with .zip extension, e.g. "dataset.zip"'
            )


def parameters2csv(
    parameters: List[Dict], path: str, parameters_filename: str
) -> None:
    """Save parameters to .csv file

    :param parameters: Parameters for each image
    :type parameters: List[Dict]
    :param path: Path to the output directory
    :type path: str
    :param parameters_filename: Name of the parameters file
    :type parameters_filename: str
    """
    df = pd.DataFrame.from_dict(parameters)
    df.to_csv(path + parameters_filename, encoding="utf-8", index=False)
