import pytest

from src.data_generation.image.decorators.abstract_image import generate_pure_image

def test_image_attributes():
    with pytest.raises(ValueError):
        generate_pure_image(
            size=(640, 480),
            epsilon=1.5,
            ring_center=(320, 240),
            brightness=(80, 210)
        )
        