from typing import Optional

from pydantic import BaseModel, validator


class PureImageParams(BaseModel):
    width: int = 640
    height: int = 480
    epsilon: float = 0.0
    ring_center_width: int = 320
    ring_center_height: int = 240
    min_brightness: int = 80
    max_brightness: int = 210


class AverageNoiseParams(BaseModel):
    noise_used: int


class PizzaNoiseParams(BaseModel):
    pizza_count: int
    center_width: int
    center_height: int
    strength: int


class BlackboxNoiseParams(BaseModel):
    box_width: int
    box_height: int
    box_x: int
    box_y: int


class ImageDetails(BaseModel):
    width: int
    height: int
    epsilon: float
    ring_center_width: int
    ring_center_height: int
    min_brightness: int
    max_brightness: int
    used_noise: int
    filename: Optional[str] = None

    @validator("width")
    def _check_width(cls, width) -> None:
        if width <= 0:
            raise ValueError(f"Width must be positive integer, not {width}")
        return width

    @validator("height")
    def _check_height(cls, height) -> None:
        if height <= 0:
            raise ValueError(f"Height must be positive integer, not {height}")
        return height

    @validator("min_brightness")
    def _check_min_brightness(cls, min_brightness) -> None:
        if min_brightness < 40 or min_brightness > 120:
            raise ValueError("Minimal brightness must be in in range (40; 120)")
        return min_brightness

    @validator("max_brightness")
    def _check_max_brightness(cls, max_brightness) -> None:
        if max_brightness < 170 or max_brightness > 210:
            raise ValueError("Maximal brightness must be in range (170; 210)")
        return max_brightness

    @validator("max_brightness")
    def _check_min_max(cls, max_brightness, values, **kwargs) -> None:
        min_brightness = values["min_brightness"]
        if min_brightness >= max_brightness:
            raise ValueError(
                "Minimal brightness must be smaller than maximal brightness"
            )
        return max_brightness

    @validator("epsilon")
    def _check_epsilon(cls, epsilon) -> None:
        if epsilon < 0.0 or epsilon > 1.0:
            raise ValueError("Epsilon must be in range <0.0; 1.0>")
        return epsilon


if __name__ == "__main__":
    pass
