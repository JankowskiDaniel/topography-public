from AbstractImage import AbstractImage

class Image(AbstractImage):
    """
    Concrete Components provide default implementations of the generates. There
    might be several variations of these classes.
    """
    
    def generate(self) -> str:
        return self.img