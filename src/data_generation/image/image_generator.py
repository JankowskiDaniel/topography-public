import numpy as np
import numpy.typing as npt

from src.models.image_models import ImageDetails, PureImageParams
from src.data_generation.image.image_interface import AbstractGenerator


def _check_image(
    size: tuple[int, int],
    epsilon: float,
    ring_center: tuple[int, int],
    brightness: tuple[int, int],
    noise_file_index,
) -> None:
    width, height = size
    width_ring_center, height_ring_center = ring_center
    min_brightness, max_brightness = brightness
    _ = ImageDetails(  # type: ignore
        width=width,
        height=height,
        epsilon=epsilon,
        ring_center_width=width_ring_center,
        ring_center_height=height_ring_center,
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        used_noise=noise_file_index,
    )


def generate_pure_image(
    size: tuple[int, int],
    epsilon: float,
    ring_center: tuple[int, int],
    brightness: tuple[int, int],
) -> npt.NDArray[np.uint8]:
    """Generate pure image

    :param size: Size of the image (width, height)
    :type size: tuple[int, int]
    :param epsilon: Epsilon value
    :type epsilon: float
    :param ring_center: Position of the central ring
    :type ring_center: tuple[int, int]
    :param brightness: Range of brightness
    :type brightness: tuple[int, int]
    :return: 2D array which represents pure image
    :rtype: np.array
    """
    _check_image(size, epsilon, ring_center, brightness, noise_file_index=5)
    width, height = size
    min_brightness, max_brightness = brightness

    mean_brightness = (min_brightness + max_brightness) / 2
    diff_brightness = max_brightness - mean_brightness

    diff_betweeen_rings_denominator = 6.07
    diff_between_rings = width * width / diff_betweeen_rings_denominator

    width_ring_center, height_ring_center = ring_center

    y, x = np.indices([height, width])
    img = mean_brightness + (
        diff_brightness
        * np.cos(
            2
            * np.pi
            * (
                1.0
                - epsilon
                + (
                    (
                        np.power((x - width_ring_center) * 2, 2)
                        + np.power((y - height_ring_center) * 2, 2)
                    )
                    / diff_between_rings
                )
            )
        )
    )
    img = img.astype(np.uint8)
    return img


class PureImageGenerator(AbstractGenerator):
    """
    Concrete Components provide default implementations of the generates. There
    might be several variations of these classes.
    """

    def __init__(
        self,
        size: tuple[int, int],
        num_images: int,
        brightness: tuple[int, int] = (80, 210),
        center_shift: float = 0.01,
    ) -> None:
        self.size = size
        self.brightness = brightness

        self.width, self.height = size

        # center shift - random value between 0.15 and 0.25
        center_shift = np.random.uniform(0.15, 0.25)
        max_width_center_shift = self.width * center_shift
        min_width_center = int(self.width / 2 - max_width_center_shift)
        max_width_center = int(self.width / 2 + max_width_center_shift)

        max_height_center_shift = self.height * center_shift
        min_height_center = int(self.height / 2 - max_height_center_shift)
        max_height_center = int(self.height / 2 + max_height_center_shift)

        self.width_centers = np.random.randint(
            min_width_center, max_width_center + 1, num_images
        )
        self.height_centers = np.random.randint(
            min_height_center, max_height_center + 1, num_images
        )

        self.current_image_stats: PureImageParams = PureImageParams()

    def _update_image_stats(
        self, epsilon: float, ring_center: tuple[int, int]
    ) -> None:
        self.current_image_stats.epsilon = epsilon
        self.current_image_stats.ring_center_width = ring_center[0]
        self.current_image_stats.ring_center_height = ring_center[1]

    def generate(
        self, epsilon: float, img_index: int
    ) -> npt.NDArray[np.uint8]:
        ring_center = (
            self.width_centers[img_index],
            self.height_centers[img_index],
        )
        self._update_image_stats(epsilon=epsilon, ring_center=ring_center)
        return generate_pure_image(
            size=self.size,
            epsilon=epsilon,
            ring_center=ring_center,
            brightness=self.brightness,
        )
