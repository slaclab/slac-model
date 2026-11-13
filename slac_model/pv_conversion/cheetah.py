"""Cheetah (`CheetahSimulator`) accessor functions for reading/writing element PV-equivalent values."""

from functools import wraps
from slac_model.pv_conversion.utils import get_magnetic_rigidity
import torch

def _get_element(simulator, element_name: str):
	"""Get element by normalized (uppercase) name from Cheetah segment."""
	# Normalize to uppercase since Cheetah stores lowercase names but we use uppercase
	normalized_name = element_name.upper()
	
	# Search for the element by name (case-insensitive)
	for element in simulator.segment.elements:
		if element.name.upper() == normalized_name:
			return element
	
	raise AttributeError(
		f"Segment has no element named '{element_name}' "
		f"(normalized to '{normalized_name}')"
	)


def validate_element(element_type: str):
	def decorator(func):
		@wraps(func)
		def wrapper(simulator, element_name: str, *args, **kwargs):
			actual_type = type(_get_element(simulator, element_name)).__name__
			if actual_type.lower() != element_type.lower():
				raise ValueError(f"Element {element_name} is not of type {element_type}")
			return func(simulator, element_name, *args, **kwargs)
		return wrapper
	return decorator


def _get_energy(simulator, element_name: str) -> torch.Tensor:
	"""Get the tracked entrance reference energy in eV, averaged over batches.

	Raise ValueError if the element has no tracked energy.
	"""
	normalized_name = element_name.upper()
	
	for tracked_elem_name, energy_tensor in simulator.energies.items():
		if tracked_elem_name.upper() == normalized_name:
			return energy_tensor.mean()
	
	raise ValueError(f"No tracked beam energy for element {element_name}")


def _get_magnet_value(simulator, element_name: str, attribute_name: str) -> torch.Tensor:
	element = _get_element(simulator, element_name)
	return getattr(element, attribute_name) * get_magnetic_rigidity(_get_energy(simulator, element_name))


def _set_magnet_value(simulator, element_name: str, attribute_name: str, value: torch.Tensor):
	element = _get_element(simulator, element_name)
	setattr(element, attribute_name, value / get_magnetic_rigidity(_get_energy(simulator, element_name)))


@validate_element(element_type="quadrupole")
def get_quadrupole_bctrl(simulator, element_name: str) -> torch.Tensor:
	element = _get_element(simulator, element_name)
	return (
		element.k1
		* element.length
		* get_magnetic_rigidity(_get_energy(simulator, element_name))
	)


@validate_element(element_type="quadrupole")
def set_quadrupole_bctrl(simulator, element_name: str, value: torch.Tensor):
	element = _get_element(simulator, element_name)
	element.k1 = value / get_magnetic_rigidity(_get_energy(simulator, element_name)) / element.length


get_quadrupole_bact = get_quadrupole_bctrl

@validate_element(element_type="solenoid")
def get_solenoid_bctrl(simulator, element_name: str) -> torch.Tensor:
	return _get_magnet_value(simulator, element_name, "k")


@validate_element(element_type="solenoid")
def set_solenoid_bctrl(simulator, element_name: str, value: torch.Tensor):
	_set_magnet_value(simulator, element_name, "k", value)


get_solenoid_bact = get_solenoid_bctrl

@validate_element(element_type="HorizontalCorrector")
def get_hkicker_bctrl(simulator, element_name: str) -> torch.Tensor:
	return _get_magnet_value(simulator, element_name, "angle")

@validate_element(element_type="HorizontalCorrector")
def set_hkicker_bctrl(simulator, element_name: str, value: torch.Tensor):
	_set_magnet_value(simulator, element_name, "angle", value)

get_hkicker_bact = get_hkicker_bctrl

@validate_element(element_type="VerticalCorrector")
def get_vkicker_bctrl(simulator, element_name: str) -> torch.Tensor:
	return _get_magnet_value(simulator, element_name, "angle")

@validate_element(element_type="VerticalCorrector")
def set_vkicker_bctrl(simulator, element_name: str, value: torch.Tensor):
	_set_magnet_value(simulator, element_name, "angle", value)

get_vkicker_bact = get_vkicker_bctrl


@validate_element(element_type="bpm")
def get_bpm_x(simulator, element_name: str) -> torch.Tensor:
	reading = _get_element(simulator, element_name).reading
	# BPM reading is [batch, 2]
	return reading[..., 0] * 1e3


@validate_element(element_type="bpm")
def get_bpm_y(simulator, element_name: str) -> torch.Tensor:
	reading = _get_element(simulator, element_name).reading
	# BPM reading is [batch, 2]
	return reading[..., 1] * 1e3

@validate_element(element_type="transversedeflectingcavity")
def get_cavity_areq(simulator, element_name: str) -> torch.Tensor:
	return _get_element(simulator, element_name).voltage / 1e6

@validate_element(element_type="transversedeflectingcavity")
def set_cavity_areq(simulator, element_name: str, value: torch.Tensor):
	element = _get_element(simulator, element_name)
	element.voltage = element.voltage.new_tensor(value * 1e6)

get_cavity_areq_readback = get_cavity_areq

# cavity phase in degrees --> rad / 2pi
@validate_element(element_type="transversedeflectingcavity")
def get_cavity_preq(simulator, element_name: str) -> torch.Tensor:
	return _get_element(simulator, element_name).phase * 360.0

@validate_element(element_type="transversedeflectingcavity")
def set_cavity_preq(simulator, element_name: str, value: torch.Tensor):
	element = _get_element(simulator, element_name)
	element.phase = element.phase.new_tensor(value / 360.0)


get_cavity_preq_readback = get_cavity_preq

@validate_element(element_type="screen")
def get_screen_image(simulator, element_name: str, shape=None, pixel_size=None):
	# shape/pixel_size are accepted for backend-signature parity but unused: Cheetah screens carry their own geometry.
	return _get_element(simulator, element_name).reading.mT * 65535


def get_screen_image_array_size(simulator, element_name: str, shape=None, index: int = 0):
	return _get_element(simulator, element_name).resolution[index]


def get_screen_resolution(simulator, element_name: str, pixel_size=None) -> torch.Tensor:
	return _get_element(simulator, element_name).pixel_size[0] * 1e6


def get_cavity_modecfg(simulator, element_name: str) -> str:
	return "ACCEL_STDBY"


def get_screen_pneumatic(simulator, element_name: str) -> float:
	return 1.0 if bool(_get_element(simulator, element_name).is_active) else 0.0


def set_screen_pneumatic(simulator, element_name: str, value: float):
	_get_element(simulator, element_name).is_active = 1.0 if bool(value) else 0.0


def _get_screen_centroid(simulator, element_name: str, axis: str) -> torch.Tensor:
	beam = _get_element(simulator, element_name).get_read_beam()
	return getattr(beam, axis).mean() * 1e3


def get_screen_x(simulator, element_name: str) -> torch.Tensor:
	return _get_screen_centroid(simulator, element_name, "x")

def get_screen_y(simulator, element_name: str) -> torch.Tensor:
	return _get_screen_centroid(simulator, element_name, "y")


# ============================================================================
# Limit and Control State Variables (Dummy Implementations)
# ============================================================================
# These are read-only properties that represent hardware limits and control state.
# For Cheetah (simulation), we return fixed dummy values that represent typical
# operating limits.


def get_bmin(simulator, element_name: str) -> torch.Tensor:
	"""Get minimum field strength for a magnet (dummy value for simulation)."""
	return torch.tensor(-100.0, dtype=torch.float32)


def get_bmax(simulator, element_name: str) -> torch.Tensor:
	"""Get maximum field strength for a magnet (dummy value for simulation)."""
	return torch.tensor(100.0, dtype=torch.float32)


def get_control_state(simulator, element_name: str) -> str:
	"""Get control state of a magnet (dummy value: always 'Remote')."""
	return "Remote"
