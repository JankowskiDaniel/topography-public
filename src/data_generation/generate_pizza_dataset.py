import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from _models import ImageDetails
from generate_utils import save2directory, save2zip, _check_args, parameters2csv
from tqdm import tqdm
from Image import Image
from Pizza import Pizza


def generate_pizza_dataset(
    path: str,
    n_copies: int,
    epsilon_range: Tuple[float, float] = (0.0, 1.0),
    epsilon_step: float = 0.001,
    size: Tuple[int, int] = (640, 480),
    brightness: Tuple[int, int] = (80, 210),
    center_shift: float = 0.01,
    zipfile: bool = False,
    filename: str = "",
    save_parameters: bool = True,
    parameters_filename: str = "parameters.csv",
    seed: Optional[int] = None,
    nr_of_pizzas: Tuple[int, int] = (3,7),
    strength_of_pizzas: Tuple[int, int] = (10,15),
) -> None:
    """Generate balanced dataset and save to the output
    directory or .zip file. Noise_path argument need to be
    passed if you didn't install package via pip.

    :param path: Path where output images or compressed .zip file
        should be stored
    :type path: str
    :param n_copies: Number of images that has to be created with the
        same epsilon value.
    :type n_copies: int
    :param epsilon_range: Range of epsilons values used to generate images,
        defaults to (0.0, 1.0)
    :type epsilon_range: Tuple[float, float], optional
    :param epsilon_step: Step by epsilon value increases every iteration,
        defaults to 0.001
    :type epsilon_step: float, optional
    :param size: Size of generated images (width, height),
        defaults to (640, 480)
    :type size: Tuple[int, int], optional
    :param brightness: Brightness range of each pixel, defaults to (80,210)
    :type brightness: Tuple[int, int], optional
    :param center_shift: Percentage of random shifting ring center
        in the image, defaults to 0.01.
    :type brightness: Tuple[int, int], optional
    :param zipfile: Set to True if output images should be compressed
        to .zip file, defaults to False
    :type zipfile: bool, optional
    :param filename: Name of output .zip file.
        Need to be provided if zipfile is True, defaults to None
    :type filename: str, optional
    :param save_parameters: Set to False if additional file with each
    image parameters should not be stored, defaults to True
    :type save_parameters: bool, optional
    :param parameters_filename: Name of parameters file,
        defaults to "parameters.csv"
    :type parameters_filename: str, optional
    :param noise_path: Path to the noise dataset, optional
    :type noise_path: str, optional
    :param seed: Set seed to create identical dataset each time.
    :type seed: int, optional
    :param nr_of_pizzas: Number of triangles imposed on the image (drawn from the specified range)
        defaults to (3, 7)
    :type nr_of_pizzas: Tuple[int, int], optional
    :param strength_of_pizzas: Defines how intense the triangles will be (drawn from the specified range)
        defaults to (10, 15)
    :type strength_of_pizzas: Tuple[int, int], optional
    """
    _check_args(path, n_copies, epsilon_step, zipfile, filename)

    if seed:
        random.seed(seed)

    min_epsilon, max_epsilon = epsilon_range
    width, height = size

    max_width_center_shift = width * center_shift
    min_width_center = int(width / 2 - max_width_center_shift)
    max_width_center = int(width / 2 + max_width_center_shift)

    max_height_center_shift = height * center_shift
    min_height_center = int(height / 2 - max_height_center_shift)
    max_height_center = int(height / 2 + max_height_center_shift)

    img_index = 0
    parameters: List[Dict] = []
    epsilons = np.arange(
        start=min_epsilon, stop=max_epsilon, step=epsilon_step
    )

    # create arrays with ring_center position and choosen noises.
    # Those arrays will be always the same if you set the seed.
    if seed is not None:
        np.random.seed(seed)
    width_centers = np.random.randint(
        min_width_center, max_width_center + 1, len(epsilons) * n_copies
    )
    height_centers = np.random.randint(
        min_height_center, max_height_center + 1, len(epsilons) * n_copies
    )

    for _epsilon in tqdm(epsilons):
        _epsilon = float("{:.3f}".format(_epsilon))
        for _ in range(n_copies):
            ring_center = (width_centers[img_index], height_centers[img_index])

            # There is no noise added here.
            img = Image(size, _epsilon, ring_center, brightness)
            img = Pizza(img, nr_of_pizzas, ring_center, 1, strength=strength_of_pizzas)
            img = img.generate()

            img_filename = f"{str(img_index).zfill(5)}.png"

            if zipfile:
                save2zip(img, img_filename, filename, path)
            else:
                save2directory(img, img_filename, path)

            if save_parameters:
                img_details = ImageDetails(
                    filename=img_filename,
                    width=width,
                    height=height,
                    epsilon=_epsilon,
                    ring_center_width=ring_center[0],
                    ring_center_height=ring_center[1],
                    min_brightness=brightness[0],
                    max_brightness=brightness[1],
                    used_noise=-1, # dałem roboczo noise jako -1, (nie mogłem dać none) w ten sposób będzie wiadomo że avg noise nie był nałożony 
                )
                parameters.append(img_details.dict())
            img_index += 1
        
    parameters2csv(parameters, path, parameters_filename)


if __name__ == "__main__":

    generate_pizza_dataset(
        path='data/pizza_dataset/', # trzeba ręcznie utworzyć katalog
        n_copies=1,
        seed=22,
        nr_of_pizzas=(10, 20), # warto modyfikować
        strength_of_pizzas=(3,8), # warto modyfikować
        zipfile=True,
        filename='pizza.zip'
    )
