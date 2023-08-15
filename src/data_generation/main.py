import cv2
from AbstractImage import AbstractImage
from BlackBox import BlackBox
from Bubble import Bubble
from Image import Image
from Noise import Noise
from Pizza import Pizza


def generate_image(component: AbstractImage, name="img") -> None:
    cv2.imshow(name, component.generate())
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    size = [640, 480]
    epsilon = 0.1
    ring_center = [300, 250]
    brightness = [100, 210]

    SPRAY_PARTICLES = 800
    SPRAY_DIAMETER = 8
    fringes_color = None
    range_of_blobs = [30, 40]

    simple = Image(size, epsilon, ring_center, brightness)

    generate_image(simple, "basic")

    # If you comment a section the noise will not be added.
    img = Bubble(
        simple,
        SPRAY_PARTICLES=None,
        SPRAY_DIAMETER=None,
        fringes_color=None,
        range_of_blobs=range_of_blobs,
        sigma=100,
    )
    img = Pizza(img, [3, 7], [130, 130], 1)
    img = Noise(img)  # half black half white image
    img = BlackBox(img)
    generate_image(img, "final")
