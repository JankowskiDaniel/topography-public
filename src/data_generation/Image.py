import numpy as np
import numpy.typing as npt
from AbstractImage import AbstractImage


class Image(AbstractImage):
    """
    Concrete Components provide default implementations of the generates. There
    might be several variations of these classes.
    """

    def generate(self) -> npt.NDArray[np.uint8]:
        return self.img
