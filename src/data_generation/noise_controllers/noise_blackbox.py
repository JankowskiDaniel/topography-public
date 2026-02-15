import numpy as np
import numpy.typing as npt
from src.data_generation.noise_controllers.decorator import NoiseController


def add_blackbox(
    img: np.ndarray,
    blackbox_w: int = 50,
    blackbox_h: int = 30,
    blackbox_x: int = 0,
    blackbox_y: int = 0,
):
    """
    ---
    Attributes:
    * img (numpy.ndarray): input image
    * blackbox_w (int): width of blackbox
    * blackbox_h (int): height of blackbox
    * blackbox_x (int): horizontal position
    * blackbox_y (int): vertical position
    ---
    Returns:
    * modified image (numpy.ndarray)
    """
    h, w = img.shape

    blackbox_w = min(blackbox_w, w - blackbox_x)
    blackbox_h = min(blackbox_h, h - blackbox_y)

    img[
        blackbox_y : blackbox_y + blackbox_h,
        blackbox_x : blackbox_x + blackbox_w,
    ] = 0

    return img


class BlackboxController(NoiseController):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def __init__(
        self,
        width: tuple[int, int] = (30, 320),
        height: tuple[int, int] = (30, 240),
        x: tuple[int, int] = (0, 640),
        y: tuple[int, int] = (0, 480),
    ) -> None:
        self.width = width
        self.height = height
        self.x = x
        self.y = y

    def _set_additional_parameters(self, num_images: int) -> None:
        self.num_images = num_images
        self.choosen_widths = np.random.randint(
            self.width[0], self.width[1], num_images
        )

        self.choosen_heights = np.random.randint(
            self.height[0], self.height[1], num_images
        )

        self.choosen_xs = np.random.randint(self.x[0], self.x[1], num_images)

        self.choosen_ys = np.random.randint(self.y[0], self.y[1], num_images)

        self.noise_index = 0

    def generate(
        self, img: npt.NDArray[np.uint8], epsilon: float
    ) -> npt.NDArray[np.uint8]:
        noised_image = add_blackbox(
            img,
            blackbox_w=self.choosen_widths[self.noise_index],
            blackbox_h=self.choosen_heights[self.noise_index],
            blackbox_x=self.choosen_xs[self.noise_index],
            blackbox_y=self.choosen_ys[self.noise_index],
        )
        self.noise_index += 1
        return noised_image
