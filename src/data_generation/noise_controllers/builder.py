from src.data_generation.noise_controllers.decorator import NoiseController
from src.data_generation.noise_controllers.noise_average import AverageController
from src.data_generation.noise_controllers.noise_blackbox import BlackboxController
from src.data_generation.noise_controllers.noise_bubble import BubbleController
from src.data_generation.noise_controllers.noise_pizza import PizzaController
from src.data_generation.noise_controllers.noise_fourier import FourierController


CONTROLLERS = {
    "average": AverageController,
    "blackbox": BlackboxController,
    "bubble": BubbleController,
    "pizza": PizzaController,
    "fourier": FourierController
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

def _validate_fourier_noise_params(**params) -> None:
    pass


VALIDATORS = {
    "average": _validate_average_noise_params,
    "blackbox": _validate_blackbox_noise_params,
    "bubble": _validate_bubble_noise_params,
    "pizza": _validate_pizza_noise_params,
    "fourier": _validate_fourier_noise_params,
}


def build_noise_controller(noiser: str, **params) -> NoiseController:
    VALIDATORS[noiser](params=params)
    controller: NoiseController = CONTROLLERS[noiser]()
    for p in params:
        setattr(controller, p, params[p])
    return controller
