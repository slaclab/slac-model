from functools import wraps

from impact import Impact

from . import _screen_common
from ._screen_common import histogram_screen_image

def validate_element(element_type: str):
    def decorator(func):
        @wraps(func)
        def wrapper(simulator: Impact, element_name: str, *args, **kwargs):
            actual_type = simulator.ele[element_name]["type"]
            if actual_type.lower() != element_type.lower():
                raise ValueError(f"Element {element_name} is not of type {element_type}")
            return func(simulator, element_name, *args, **kwargs)
        return wrapper
    return decorator

@validate_element(element_type="quadrupole")
def set_quadrupole_bctrl(simulator: Impact, element_name: str, value: float):
    effective_length = simulator.ele[element_name]["L_effective"]
    simulator.ele[element_name]["b1_gradient"] = -value / (10 * effective_length)

@validate_element(element_type="quadrupole")
def get_quadrupole_bctrl(simulator: Impact, element_name: str) -> float:
    effective_length = simulator.ele[element_name]["L_effective"]
    return -simulator.ele[element_name]["b1_gradient"] * effective_length * 10


get_quadrupole_bact = get_quadrupole_bctrl


@validate_element(element_type="write_beam")
def get_screen_image(simulator: Impact, element_name: str, shape, pixel_size):
    """Histogram the tracked beam at `element_name` into a `shape`-sized image, normalized to unit scale."""
    beam = simulator.particles[element_name]
    return histogram_screen_image(beam, shape, pixel_size, normalize=True)


get_screen_image_array_size = _screen_common.get_screen_image_array_size
get_screen_resolution = _screen_common.get_screen_resolution


def get_screen_x(simulator: Impact, element_name: str) -> float:
    return simulator.particles[element_name].x.mean() * 1e3


def get_screen_y(simulator: Impact, element_name: str) -> float:
    return simulator.particles[element_name].y.mean() * 1e3