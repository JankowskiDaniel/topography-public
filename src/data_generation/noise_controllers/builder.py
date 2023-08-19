from src.data_generation.noise_controllers.decorator import AbstractDecorator
from src.data_generation.noise_controllers.noise_average import AverageNoise
from src.data_generation.noise_controllers.noise_blackbox import BlackBox
from src.data_generation.noise_controllers.noise_bubble import Bubble
from src.data_generation.noise_controllers.noise_pizza import Pizza


CONTROLLERS = {
    "average": AverageNoise,
    "blackbox": BlackBox,
    "bubble": Bubble,
    "pizza": Pizza
}


# TODO
# All below methods should validate parameters
# for a particular noise type. Should return None
# but raises exceptions if needed
def _validate_average_noise_params(**params) -> None:
    pass

def _validate_blackbox_noise_params(**params) -> None:
    pass


def _validate_bubble_noise_params(**params) -> None:
    pass


def _validate_pizza_noise_params(**params) -> None:
    pass


VALIDATORS = {
    "average": _validate_average_noise_params,
    "blackbox": _validate_blackbox_noise_params,
    "bubble": _validate_bubble_noise_params,
    "pizza": _validate_pizza_noise_params
}

def build_noise_controller(noiser: str, **params) -> AbstractDecorator:
    VALIDATORS[noiser](params=params)
    controller: AbstractDecorator = CONTROLLERS[noiser]()
    for p in params:
        setattr(controller, p, params[p])
    return controller
