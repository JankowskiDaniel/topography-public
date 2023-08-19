import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from src.models.image_models import ImageDetails
from src.data_generation.noise_controllers.decorator import AbstractDecorator

from src.data_generation.noise_controllers.builder import build_noise_controller

from src.data_generation.datasets.generate_utils import (
    _check_args,
    parameters2csv,
    save2directory,
    save2zip,
)
from src.data_generation.image.image_generator import Image
from tqdm import tqdm


def generate_dataset(
    noise_type: str,
    path: str,
    n_copies: int,
    epsilon_range: tuple[float, float] = (0.0, 1.0),
    epsilon_step: float = 0.001,
    size: Tuple[int, int] = (640, 480),
    brightness: Tuple[int, int] = (80, 210),
    center_shift: float = 0.01,
    zipfile: bool = False,
    filename: str = "",
    save_parameters: bool = True,
    parameters_filename: str = "parameters.csv",
    seed: Optional[int] = None,
    *args, 
    **kwargs
) -> None:
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
    
    num_images = len(epsilons) * n_copies

    # create arrays with ring_center position and choosen noises.
    # Those arrays will be always the same if you set the seed.
    if seed is not None:
        np.random.seed(seed)
    width_centers = np.random.randint(
        min_width_center, max_width_center + 1, num_images
    )
    height_centers = np.random.randint(
        min_height_center, max_height_center + 1, num_images
    )
    
    
    noise_controller: AbstractDecorator = build_noise_controller(
        noiser=noise_type,
        **kwargs
    )
    noise_controller._set_additional_parameters(num_images=num_images)

    for _epsilon in tqdm(epsilons):
        _epsilon = float("{:.3f}".format(_epsilon))
        for _ in range(n_copies):
            ring_center = (width_centers[img_index], height_centers[img_index])

            # There is no noise added here.
            builder = Image(size, _epsilon, ring_center, brightness)
            img = builder.generate()
            img = noise_controller.generate(img)
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
                    used_noise=-1,
                    # dałem roboczo noise jako -1,
                    # (nie mogłem dać none) w ten sposób
                    # będzie wiadomo że avg noise nie był nałożony
                )
                parameters.append(img_details.dict())
            img_index += 1

    parameters2csv(parameters, path, parameters_filename)