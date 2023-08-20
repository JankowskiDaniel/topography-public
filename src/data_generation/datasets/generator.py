import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from src.data_generation.datasets.generate_utils import (
    _check_args,
    parameters2csv,
    save2directory,
    save2zip,
)
from src.data_generation.image.image_generator import PureImageGenerator
from src.data_generation.noise_controllers.builder import (
    build_noise_controller,
)
from src.data_generation.noise_controllers.decorator import NoiseController
from src.models.image_models import ImageDetails, PureImageParams


def generate_dataset(
    noise_type: Union[list[str], str],
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
    **kwargs,
) -> None:
    _check_args(path, n_copies, epsilon_step, zipfile, filename)
    
    if isinstance(noise_type, str):
        noise_type = [noise_type]

    if seed:
        random.seed(seed)
        np.random.seed(seed)


    
    
    min_epsilon, max_epsilon = epsilon_range
    epsilons = np.arange(
        start=min_epsilon, stop=max_epsilon, step=epsilon_step
    )

    num_images = len(epsilons) * n_copies

    

    pure_generator = PureImageGenerator(
        size=size,
        num_images=num_images,
        brightness=brightness,
        center_shift=center_shift
    )
    controllers: list[NoiseController] = [
        build_noise_controller(noise, **kwargs) for noise in noise_type
    ]
    for controller in controllers:
        controller._set_additional_parameters(
            num_images=num_images
        )

    img_index = 0
    parameters: List[Dict] = []
    for _epsilon in tqdm(epsilons):
        _epsilon = float("{:.3f}".format(_epsilon))
        for _ in range(n_copies):
            
            img = pure_generator.generate(_epsilon,
                                           img_index=img_index)
            
            for controller in controllers:
                img = controller.generate(img)
                
            img_filename = f"{str(img_index).zfill(5)}.png"

            if zipfile:
                save2zip(img, img_filename, filename, path)
            else:
                save2directory(img, img_filename, path)

            if save_parameters:
                pure_parameters: PureImageParams = pure_generator.current_image_stats
                img_details = ImageDetails(
                    filename=img_filename,
                    width=pure_parameters.width,
                    height=pure_parameters.height,
                    epsilon=_epsilon,
                    ring_center_width=pure_parameters.ring_center_width,
                    ring_center_height=pure_parameters.ring_center_height,
                    min_brightness=brightness[0],
                    max_brightness=brightness[1],
                    used_noise=-1,
                    # dałem roboczo noise jako -1,
                    # (nie mogłem dać none) w ten sposób
                    # będzie wiadomo że avg noise nie był nałożony
                )
                parameters.append(img_details.dict())
            # img_index += 1

    parameters2csv(parameters, path, parameters_filename)
