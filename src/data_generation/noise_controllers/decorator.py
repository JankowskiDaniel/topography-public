import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod


class NoiseController(ABC):
    """This is interface for all noise controllers
    implemented in this module.
    """

    @abstractmethod
    def _set_additional_parameters(self, num_images: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        raise NotImplementedError
