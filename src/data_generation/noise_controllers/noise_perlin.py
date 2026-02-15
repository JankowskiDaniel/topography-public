import random
import numpy as np
import numpy.typing as npt
import cv2
from noise import pnoise2
from src.data_generation.noise_controllers.decorator import NoiseController


def generate_perlin_noise(
    width: int,
    height: int,
    scale: float = 100.0,
    octaves: int = 6,
    persistence: float = 0.7,
    lacunarity: float = 2.0,
    offset_x: float | None = None,
    offset_y: float | None = None,
) -> npt.NDArray[np.uint8]:
    """Generate Perlin noise pattern.

    :param width: Width of the noise pattern
    :param height: Height of the noise pattern
    :param scale: Scale of the noise (higher = more zoomed out)
    :param octaves: Number of noise octaves
    :param persistence: Amplitude multiplier per octave
    :param lacunarity: Frequency multiplier per octave
    :param offset_x: Random offset for x (auto-generated if None)
    :param offset_y: Random offset for y (auto-generated if None)
    :return: 2D array representing Perlin noise
    """
    if offset_x is None:
        offset_x = random.uniform(0, 1000)
    if offset_y is None:
        offset_y = random.uniform(0, 1000)

    noise = np.zeros((height, width), dtype=np.float32)

    for i in range(height):
        for j in range(width):
            noise[i][j] = pnoise2(
                (i + offset_x) / scale,
                (j + offset_y) / scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )

    # Normalize to [0, 255] range
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)


def apply_clahe(
    image: npt.NDArray[np.uint8],
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> npt.NDArray[np.uint8]:
    """Apply Contrast Limited Adaptive Histogram Equalization.

    :param image: Input grayscale image
    :param clip_limit: Threshold for contrast limiting
    :param tile_grid_size: Size of grid for histogram equalization
    :return: Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced  # type: ignore[return-value]


class PerlinController(NoiseController):
    """Perlin noise controller with CLAHE enhancement.

    This controller generates Perlin noise and blends it with the input image,
    then applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to
    enhance local contrast.
    """

    def __init__(
        self,
        alpha: float = 0.45,
        scale: float = 100.0,
        octaves: int = 6,
        persistence_range: tuple[float, float] = (0.7, 0.9),
        lacunarity: float = 2.0,
        grid_sizes: list[tuple[int, int]] | None = None,
        clip_limit_range: tuple[float, float] = (1.0, 2.0),
    ) -> None:
        """Initialize Perlin noise controller.

        :param alpha: Blending weight for Perlin noise (0.0 = no noise, 1.0 = pure noise)
        :param scale: Scale parameter for Perlin noise generation
        :param octaves: Number of octaves for Perlin noise
        :param persistence_range: Range for random persistence selection per image
        :param lacunarity: Lacunarity parameter for Perlin noise
        :param grid_sizes: List of possible CLAHE grid sizes (random selection per image)
        :param clip_limit_range: Range for random CLAHE clip limit selection per image
        """
        self.alpha = alpha
        self.scale = scale
        self.octaves = octaves
        self.persistence_range = persistence_range
        self.lacunarity = lacunarity
        self.grid_sizes = grid_sizes or [(2, 2), (4, 4), (8, 8), (16, 16)]
        self.clip_limit_range = clip_limit_range

    def _set_additional_parameters(self, num_images: int) -> None:
        """Set image-specific random parameters.

        Pre-generates random parameters for each image in the batch to ensure
        reproducibility when using seeds.

        :param num_images: Total number of images to generate parameters for
        """
        self.num_images = num_images
        self.noise_index = 0

        # Pre-generate random parameters for each image
        self.persistences = [
            random.uniform(*self.persistence_range) for _ in range(num_images)
        ]
        self.grid_sizes_chosen = [
            random.choice(self.grid_sizes) for _ in range(num_images)
        ]
        self.clip_limits = [
            random.uniform(*self.clip_limit_range) for _ in range(num_images)
        ]

    def generate(
        self, img: npt.NDArray[np.uint8], epsilon: float
    ) -> npt.NDArray[np.uint8]:
        """Apply Perlin noise to image.

        :param img: Input pure image
        :param epsilon: Epsilon value (not used in noise generation but kept for interface)
        :return: Image with Perlin noise and CLAHE enhancement applied
        """
        height, width = img.shape

        # Generate Perlin noise
        perlin_noise = generate_perlin_noise(
            width=width,
            height=height,
            scale=self.scale,
            octaves=self.octaves,
            persistence=self.persistences[self.noise_index],
            lacunarity=self.lacunarity,
        )

        # Blend with original image
        blended = cv2.addWeighted(img, 1 - self.alpha, perlin_noise, self.alpha, 0)

        # Apply CLAHE for contrast enhancement
        enhanced = apply_clahe(
            blended,  # type: ignore[arg-type]
            clip_limit=self.clip_limits[self.noise_index],
            tile_grid_size=self.grid_sizes_chosen[self.noise_index],
        )

        self.noise_index += 1
        return enhanced
