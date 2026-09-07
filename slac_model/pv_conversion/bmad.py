from functools import partial, wraps
from typing import Any, Callable, Optional

import numpy as np
from pytao import Tao

from . import _screen_common
from ._screen_common import histogram_screen_image

def validate_element(element_type: str):
    def decorator(func):
        @wraps(func)
        def wrapper(simulator: Tao, element_name: str, *args, **kwargs):
            element_attributes = simulator.ele_gen_attribs(element_name)
            if element_attributes["TYPE"].lower() != element_type.lower():
                raise ValueError(f"Element {element_name} is not of type {element_type}")
            return func(simulator, element_name, *args, **kwargs)
        return wrapper
    return decorator

def set_element_attribute(tao: Tao, element_name: str, attribute_name: str, value):
    """
    Set an attribute of a Bmad element.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the Bmad element whose attribute is to be set.
    attribute_name : str
        The name of the attribute to set.
    value : any
        The value to set the attribute to.

    """
    tao.cmd(f"set ele {element_name} {attribute_name} = {value}")

def get_element_attribute(simulator: Tao, element_name: str, attribute_name: str):
    """
    Get an attribute of a Bmad element.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the Bmad element whose attribute is to be retrieved.
    attribute_name : str
        The name of the attribute to retrieve.

    Returns:
    --------
    The value of the specified attribute.
    """
    return simulator.ele_gen_attribs(element_name)[attribute_name]

@validate_element(element_type="overlay")
def get_overlay_attribute(simulator: Tao, element_name: str, attribute_name: str):
    """
    Get an overlay attribute of a Bmad element.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the Bmad element whose overlay attribute is to be retrieved.
    attribute_name : str
        The name of the overlay attribute to retrieve.

    Returns:
    --------
    The value of the specified overlay attribute.
    """
    return simulator.ele_gen_attribs(element_name).control_vars[attribute_name]

@validate_element(element_type="overlay")
def set_overlay_attribute(simulator: Tao, element_name: str, attribute_name: str, value: Any):
    """
    Set an overlay attribute of a Bmad element.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the Bmad element whose overlay attribute is to be set.
    attribute_name : str
        The name of the overlay attribute to set.
    value : any
        The value to set the overlay attribute to.

    """
    simulator.ele_gen_attribs(element_name).control_vars[attribute_name] = value

def _make_element_attribute_funcs(element_type, field_attr, to_pv, from_pv):
    @validate_element(element_type=element_type)
    def get_bctrl(simulator: Tao, element_name: str):
        return to_pv(simulator.ele_gen_attribs(element_name))

    @validate_element(element_type=element_type)
    def set_bctrl(simulator: Tao, element_name: str, value: float):
        attrs = simulator.ele_gen_attribs(element_name)
        set_element_attribute(simulator, element_name, field_attr, from_pv(attrs, value))

    return get_bctrl, set_bctrl

    
get_quadrupole_bctrl, set_quadrupole_bctrl = _make_element_attribute_funcs(
    element_type="quadrupole",
    field_attr="B1_GRADIENT",
    to_pv=lambda attrs: -attrs["B1_GRADIENT"] * attrs["L"] * 10,
    from_pv=lambda attrs, value: -value / (attrs["L"] * 10),
)
get_quadrupole_bact = get_quadrupole_bctrl

get_solenoid_bctrl, set_solenoid_bctrl = _make_element_attribute_funcs(
    element_type="solenoid",
    field_attr="BS_FIELD",
    to_pv=lambda attrs: -attrs["BS_FIELD"] * 10,
    from_pv=lambda attrs, value: -value / 10,
)
get_solenoid_bact = get_solenoid_bctrl

get_kicker_bctrl, set_kicker_bctrl = _make_element_attribute_funcs(
    element_type="kicker",
    field_attr="BL_KICK",
    to_pv=lambda attrs: -attrs["BL_KICK"] * 10,
    from_pv=lambda attrs, value: -value / 10,
)
get_kicker_bact = get_kicker_bctrl


@validate_element(element_type="sbend")
def get_sbend_bctrl(simulator: Tao, element_name: str) -> float:
    attrs = simulator.ele_gen_attribs(element_name)
    if attrs["G"] == 0:
        return 0.0
    momentum = attrs["P0C"] * (1 + attrs["DG"] / attrs["G"])
    return momentum * 1e-9


@validate_element(element_type="sbend")
def set_sbend_bctrl(simulator: Tao, element_name: str, value: float):
    attrs = simulator.ele_gen_attribs(element_name)
    relative_momentum = (value * 1e9 - attrs["P0C"]) / attrs["P0C"]
    set_element_attribute(simulator, element_name, "DG", relative_momentum * attrs["G"])


get_sbend_bact = get_sbend_bctrl

def get_bpm_loc(simulator: Tao, element_name: str, coordinate: str) -> float:
    """
    Get the beam centroid along the `coordinate` direction in mm.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the BPM element.
    coordinate : str
        The coordinate to retrieve (`x` or `y`).

    Returns:
    --------
    float
        The location of the BPM in the specified coordinate.
    """
    if coordinate.lower() not in ['x', 'y']:
        raise ValueError("Coordinate must be 'x' or 'y'")
    return getattr(simulator.ele(element_name).orbit, coordinate.lower()) * 1e3

get_bpm_x = partial(get_bpm_loc, coordinate="x")
get_bpm_y = partial(get_bpm_loc, coordinate="y")
get_screen_x = get_bpm_x
get_screen_y = get_bpm_y

def _make_overlay_funcs(
    attr_name: str,
    scale: float = 1.0,
    to_val: Optional[Callable[[Any], Any]] = None,
    from_val: Optional[Callable[[Any], Any]] = None,
):
    def get_func(simulator: Tao, element_name: str):
        val = get_overlay_attribute(simulator, element_name, attr_name)
        return to_val(val) if to_val else val * scale

    def set_func(simulator: Tao, element_name: str, value: Any):
        val = from_val(value) if from_val else value / scale
        set_overlay_attribute(simulator, element_name, attr_name, val)

    return get_func, set_func


def _make_scaled_element_funcs(attribute_name: str, scale_factor: float):
    def get_func(simulator: Tao, element_name: str):
        return get_element_attribute(simulator, element_name, attribute_name) * scale_factor

    def set_func(simulator: Tao, element_name: str, value: float):
        set_element_attribute(simulator, element_name, attribute_name, value / scale_factor)

    return get_func, set_func


get_cavity_areq, set_cavity_areq = _make_scaled_element_funcs("VOLTAGE", 1e6)
get_cavity_preq, set_cavity_preq = _make_scaled_element_funcs("PHI0", 1 / 360.0)
get_cavity_areq_readback = get_cavity_areq
get_cavity_preq_readback = get_cavity_preq


def get_cavity_modecfg(simulator: Tao, element_name: str) -> str:
    return "ACCEL_STDBY" if simulator.ele(element_name).head.is_on else "STDBY"


def set_cavity_modecfg(simulator: Tao, element_name: str, value: str):
    if value == "ACCEL_STDBY":
        set_element_attribute(simulator, element_name, "is_on", True)
    elif value == "STDBY":
        set_element_attribute(simulator, element_name, "is_on", False)
    else:
        raise ValueError(f"Invalid value for CavityMODECFGVariable: {value}")

def _klystron_stat_from_pv(value: int) -> bool:
    if value not in (0, 1):
        raise ValueError("Status must be 0 (off) or 1 (on)")
    return value == 0

get_klystron_enld, set_klystron_enld = _make_overlay_funcs("ENLD_MEV", scale=1e-6)
get_klystron_pdes, set_klystron_pdes = _make_overlay_funcs("PHASE_DEG")
get_klystron_pact = get_klystron_pdes


def get_screen_image(simulator: Tao, element_name: str, shape, pixel_size):
    """Histogram the tracked beam at `element_name` into a `shape`-sized image."""
    if simulator.tao_global()["track_type"] != "beam":
        return np.zeros(shape)

    beam = simulator.particles(element_name)
    return histogram_screen_image(beam, shape, pixel_size)


get_screen_image_array_size = _screen_common.get_screen_image_array_size
get_screen_resolution = _screen_common.get_screen_resolution
get_klystron_stat, set_klystron_stat = _make_overlay_funcs(
    "IN_USE",
    to_val=lambda b: 0 if b else 1,
    from_val=_klystron_stat_from_pv,
)