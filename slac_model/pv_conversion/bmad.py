"""Bmad (pytao `Tao`) accessor functions for reading/writing element PV-equivalent values."""

from functools import partial, wraps
from typing import Any, Callable, Optional

import numpy as np
from pytao import Tao

from . import _screen_common
from ._screen_common import histogram_screen_image
from .utils import get_magnetic_rigidity

def validate_element(element_type: str):
    def decorator(func):
        @wraps(func)
        def wrapper(simulator: Tao, element_name: str, *args, **kwargs):
            # the "key" string label lives on the element head
            element_label = simulator.ele(element_name).head.key
            if element_label.lower() != element_type.lower():
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
    attributes = simulator.ele_gen_attribs(element_name)
    key = attribute_name if attribute_name in attributes else attribute_name.upper()
    return attributes[key]

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
    return simulator.ele_control_var(element_name)[attribute_name]

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
    set_element_attribute(simulator, element_name, attribute_name, value)

def _make_element_attribute_funcs(element_type, field_attr, to_pv, from_pv):
    @validate_element(element_type=element_type)
    def get_bctrl(simulator: Tao, element_name: str):
        return to_pv(simulator.ele_gen_attribs(element_name))

    @validate_element(element_type=element_type)
    def set_bctrl(simulator: Tao, element_name: str, value: float):
        attrs = simulator.ele_gen_attribs(element_name)
        set_element_attribute(simulator, element_name, field_attr, from_pv(attrs, value))

    return get_bctrl, set_bctrl

    
_get_quadrupole_bctrl, _set_quadrupole_bctrl = _make_element_attribute_funcs(
    element_type="quadrupole",
    field_attr="B1_GRADIENT",
    to_pv=lambda attrs: -attrs["B1_GRADIENT"] * attrs["L"] * 10,
    from_pv=lambda attrs, value: -value / (attrs["L"] * 10),
)


def _multipole_quadrupole_scale(simulator: Tao, element_name: str) -> float:
    """Convert K1L to the electron-beam quadrupole PV convention in kG."""
    attrs = simulator.ele_gen_attribs(element_name)
    # With field_master, K1L is an integrated gradient in T, not normalized 1/m.
    if attrs["field_master"]:
        return -10.0
    return get_magnetic_rigidity(attrs["P0C"])


def get_quadrupole_bctrl(simulator: Tao, element_name: str) -> float:
    """Read a quadrupole or thin multipole's integrated quadrupole strength."""
    if simulator.ele_head(element_name)["key"].lower() == "multipole":
        strength, = simulator.evaluate(f"ele::{element_name}[k1l]")
        return float(strength) * _multipole_quadrupole_scale(simulator, element_name)
    return _get_quadrupole_bctrl(simulator, element_name)


def set_quadrupole_bctrl(simulator: Tao, element_name: str, value: float):
    """Set integrated quadrupole strength without changing multipole tilt."""
    if simulator.ele_head(element_name)["key"].lower() == "multipole":
        strength = value / _multipole_quadrupole_scale(simulator, element_name)
        set_element_attribute(simulator, element_name, "K1L", strength)
        return
    _set_quadrupole_bctrl(simulator, element_name, value)


get_quadrupole_bact = get_quadrupole_bctrl

get_solenoid_bctrl, set_solenoid_bctrl = _make_element_attribute_funcs(
    element_type="solenoid",
    field_attr="BS_FIELD",
    to_pv=lambda attrs: -attrs["BS_FIELD"] * 10,
    from_pv=lambda attrs, value: -value / 10,
)
get_solenoid_bact = get_solenoid_bctrl

get_hkicker_bctrl, set_hkicker_bctrl = _make_element_attribute_funcs(
    element_type="hkicker",
    field_attr="BL_KICK",
    to_pv=lambda attrs: -attrs["BL_KICK"] * 10,
    from_pv=lambda attrs, value: -value / 10,
)
get_hkicker_bact = get_hkicker_bctrl

get_vkicker_bctrl, set_vkicker_bctrl = _make_element_attribute_funcs(
    element_type="vkicker",
    field_attr="BL_KICK",
    to_pv=lambda attrs: -attrs["BL_KICK"] * 10,
    from_pv=lambda attrs, value: -value / 10,
)
get_vkicker_bact = get_vkicker_bctrl


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

# cavity PV is in MV
get_cavity_areq, set_cavity_areq = _make_scaled_element_funcs("VOLTAGE", 1e-6)
# cavity phase in degrees; PHI0 is stored in Bmad as rad / 2pi (turns)
get_cavity_preq, set_cavity_preq = _make_scaled_element_funcs("PHI0", 360.0)
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

def _make_klystron_beamcode_stat_funcs(on_value: int):
    # Production BEAMCODE{1,2}_STAT is a float (0.0, 1.0, 2.0, 4.0 bypassed, ...).
    # Only the beamcode's on-value maps to IN_USE=1; everything else is off.
    def from_pv(value) -> int:
        return 1 if value == on_value else 0

    def to_pv(in_use) -> float:
        return float(on_value) if in_use == 1 else 0.0

    return _make_overlay_funcs("IN_USE", to_val=to_pv, from_val=from_pv)


get_klystron_enld, set_klystron_enld = _make_overlay_funcs("ENLD_MEV")
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
get_klystron_beamcode1_stat, set_klystron_beamcode1_stat = (
    _make_klystron_beamcode_stat_funcs(on_value=1)
)
get_klystron_beamcode2_stat, set_klystron_beamcode2_stat = (
    _make_klystron_beamcode_stat_funcs(on_value=2)
)


# ============================================================================
# Limit and Control State Variables (Dummy Implementations)
# ============================================================================
# These are read-only properties that represent hardware limits and control state.
# For Bmad (simulation), we return fixed dummy values that represent typical
# operating limits.


def get_bmin(simulator: Tao, element_name: str) -> float:
    """Get minimum field strength for a magnet (dummy value for simulation)."""
    return -100.0


def get_bmax(simulator: Tao, element_name: str) -> float:
    """Get maximum field strength for a magnet (dummy value for simulation)."""
    return 100.0


def get_control_state(simulator: Tao, element_name: str) -> str:
    """Get control state of a magnet (dummy value: always 'Remote')."""
    return "Remote"

def get_bpm_tmit(simulator: Tao, element_name: str) -> float:
    """Get total intensity (TMIT) from BPM.
    
    Returns dummy value for total beam charge in nC.
    """
    return 1.0