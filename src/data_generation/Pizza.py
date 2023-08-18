import random
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
from AbstractDecorator import AbstractDecorator
from AbstractImage import AbstractImage


def change_region(img, pts, channels=3, add=True, strenght=10):
    img_copy = img.copy()
    x, y, w, h = cv2.boundingRect(pts)
    pts = pts - np.array([x, y])

    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, pts, (255, 255, 255))
    mask = cv2.merge([mask] * channels)
    inversed_mask = cv2.bitwise_not(mask)

    image_rect = img_copy[y: y + h, x: x + w]

    change = strenght

    if add:
        image_rect_changed = cv2.add(image_rect, change)
    else:
        image_rect_changed = cv2.subtract(image_rect, change)

    image_rect_masked = cv2.bitwise_and(mask, image_rect_changed)
    image_rect_unmasked = cv2.bitwise_and(inversed_mask, image_rect)

    full_rect = cv2.add(image_rect_masked, image_rect_unmasked)

    img_copy[y: y + h, x: x + w] = full_rect
    return img_copy


def from_distance_to_point(distance, w, h):
    if distance // w == 0:
        return [distance, 0], "top"
    elif distance // (w + h - 1) == 0:
        return [w - 1, distance % (w - 1)], "right"
    elif distance // (2 * w + h - 2) == 0:
        return [w - 1 - distance % (w + h - 2), h - 1], "bottom"
    else:
        return [0, h - 1 - distance % (2 * w + h - 3)], "left"


def corner_point_if_feasible(pts):
    p1, p2 = pts
    if p1[1] == p2[1]:
        return []
    elif p1[1] == "bottom" or p1[1] == "top":
        return [p2[0][0], p1[0][1]]
    else:
        return [p1[0][0], p2[0][1]]


def pizza_noise(
    img: np.ndarray,
    nr_of_pizzas: tuple[int, int] = (5, 5),
    center_point: tuple[int, int] = (320, 240),
    strength: tuple[int, int] = (10, 15),
    channels: int = 3,
):
    """
    Randomly brightens or darkens triangular areas of the
    image starting in the center.

    ---

    Attributes:
    * img (numpy.ndarray): input image
    * nr_of_pizzas (tuple[int,int]): a list containing the
    minimum and maximum number of modified areas
    * center_point (tuple[int,int]): center location
    * strenght (int): defines how intense the change will be
    * channels (int): number of image channels

    ---

    Returns:
    * modified image (numpy.ndarray)
    """
    h, w = img.shape

    rand_l = 2 * (h + w - 2)

    random_distances = random.sample(
        range(rand_l), random.randint(*nr_of_pizzas)
    )
    random_distances_pairs = [
        [
            random_distance,
            (random_distance + int(random.uniform(rand_l // 28, rand_l // 7)))
            % rand_l,
        ]
        for random_distance in random_distances
    ]

    random_points_pairs = [
        [from_distance_to_point(x, w, h), from_distance_to_point(y, w, h)]
        for x, y in random_distances_pairs
    ]

    corners = [corner_point_if_feasible(x) for x in random_points_pairs]

    full_shapes = [
        [center_point]
        + [random_points_pairs[i][0][0]]
        + [corners[i]]
        + [random_points_pairs[i][1][0]]
        for i in range(len(random_points_pairs))
    ]

    full_shapes = [
        [point for point in shape if point != []] for shape in full_shapes
    ]

    for shape in full_shapes:
        add = random.randint(0, 1)
        rand_strength = random.randint(*strength)
        img = change_region(
            img,
            np.array(shape),
            add=add,
            strenght=rand_strength,
            channels=channels,
        )

    return img


class Pizza(AbstractDecorator):
    """
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    """

    def __init__(
        self,
        component: AbstractImage,
        nr_of_pizzas: Tuple[int, int] = (5, 5),
        center_point: Tuple[int, int] = (320, 240),
        channels: int = 3,
        strength: tuple[int, int] = (10, 15),
    ) -> None:
        super().__init__(component)
        self.nr_of_pizzas = nr_of_pizzas
        self.center_point = center_point
        self.channels = channels
        self.strength = strength

    def generate(self) -> npt.NDArray[np.uint8]:
        img = self.component.generate()
        return pizza_noise(
            img,
            self.nr_of_pizzas,
            self.center_point,
            self.strength,
            self.channels,
        )
