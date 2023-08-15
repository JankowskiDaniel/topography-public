from AbstractImage import AbstractImage


class AbstractDecorator(AbstractImage):
    """
    The base Decorator class follows the same interface as the other
    AbstractImages. The primary purpose of this class is to define
    the wrapping interface for all concrete decorators. The default
    implementation of the wrapping code might include a field for
    storing a wrapped AbstractImage and the means to initialize it.
    """

    _AbstractImage: AbstractImage = None

    def __init__(self, component: AbstractImage) -> None:
        self.component = component

    @property
    def AbstractImage(self) -> AbstractImage:
        """
        The Decorator delegates all work to the wrapped AbstractImage.
        """

        return self.component

    def generate(self) -> str:
        return self.component.generate()
