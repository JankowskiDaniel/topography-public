from abc import ABC, abstractmethod, abstractproperty
from typing import Tuple

import numpy as np
import numpy.typing as npt

from models.image_models import PureImageParams


class AbstractGenerator(ABC):
    """
    The base Component interface defines generates that can be altered by
    decorators.
    """
    
    @abstractmethod
    def _update_image_stats(self, 
                            epsilon: float,
                            ring_center: tuple[int, int]
                            ) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, 
                 epsilon: float,
                 img_index: int
                 ) -> npt.NDArray[np.uint8]:
        raise NotImplementedError
