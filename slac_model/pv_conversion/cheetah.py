from functools import wraps


def get_magnetic_rigidity(energy: float) -> float:
	"""Calculate magnetic rigidity in kG-m for beam energy in eV."""
	return 33.356 * energy / 1e9


def _get_element(simulator, element_name: str):
	return getattr(simulator.segment, element_name)


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


def _get_energy(simulator, element_name: str) -> float:
	return simulator.energies[element_name]


def _get_magnet_value(simulator, element_name: str, attribute_name: str) -> float:
	element = _get_element(simulator, element_name)
	return getattr(element, attribute_name) * get_magnetic_rigidity(_get_energy(simulator, element_name))


def _set_magnet_value(simulator, element_name: str, attribute_name: str, value: float):
	element = _get_element(simulator, element_name)
	setattr(element, attribute_name, value / get_magnetic_rigidity(_get_energy(simulator, element_name)))


@validate_element(element_type="quadrupole")
def get_quadrupole_bctrl(simulator, element_name: str) -> float:
	element = _get_element(simulator, element_name)
	return (
		element.k1
		* element.length
		* get_magnetic_rigidity(_get_energy(simulator, element_name))
	)


@validate_element(element_type="quadrupole")
def set_quadrupole_bctrl(simulator, element_name: str, value: float):
	element = _get_element(simulator, element_name)
	element.k1 = value / get_magnetic_rigidity(_get_energy(simulator, element_name)) / element.length


get_quadrupole_bact = get_quadrupole_bctrl

@validate_element(element_type="solenoid")
def get_solenoid_bctrl(simulator, element_name: str) -> float:
	return _get_magnet_value(simulator, element_name, "k")


@validate_element(element_type="solenoid")
def set_solenoid_bctrl(simulator, element_name: str, value: float):
	_set_magnet_value(simulator, element_name, "k", value)


get_solenoid_bact = get_solenoid_bctrl


@validate_element(element_type="dipole")
def get_sbend_bctrl(simulator, element_name: str) -> float:
	element = _get_element(simulator, element_name)
	if not all(hasattr(element, attr) for attr in ("g", "dg", "p0c")):
		raise ValueError(f"Element {element_name!r} does not expose sbend field attributes")
	if element.g == 0:
		return 0.0
	return element.p0c * (1 + element.dg / element.g) * 1e-9


@validate_element(element_type="dipole")
def set_sbend_bctrl(simulator, element_name: str, value: float):
	element = _get_element(simulator, element_name)
	if not all(hasattr(element, attr) for attr in ("g", "p0c")):
		raise ValueError(f"Element {element_name!r} does not expose sbend field attributes")
	element.dg = ((value * 1e9 - element.p0c) / element.p0c) * element.g


get_sbend_bact = get_sbend_bctrl


def get_kicker_bctrl(simulator, element_name: str) -> float:
	return _get_magnet_value(simulator, element_name, "angle")


def set_kicker_bctrl(simulator, element_name: str, value: float):
	_set_magnet_value(simulator, element_name, "angle", value)


get_kicker_bact = get_kicker_bctrl


@validate_element(element_type="bpm")
def get_bpm_x(simulator, element_name: str) -> float:
	return _get_element(simulator, element_name).reading[0]


@validate_element(element_type="bpm")
def get_bpm_y(simulator, element_name: str) -> float:
	return _get_element(simulator, element_name).reading[1]


def get_cavity_areq(simulator, element_name: str) -> float:
	return _get_element(simulator, element_name).voltage / 1e6


def set_cavity_areq(simulator, element_name: str, value: float):
	_get_element(simulator, element_name).voltage = value * 1e6


get_cavity_areq_readback = get_cavity_areq


def get_cavity_preq(simulator, element_name: str) -> float:
	return _get_element(simulator, element_name).phase * 360.0


def set_cavity_preq(simulator, element_name: str, value: float):
	_get_element(simulator, element_name).phase = value / 360.0


get_cavity_preq_readback = get_cavity_preq

@validate_element(element_type="screen")
def get_screen_image(simulator, element_name: str, shape=None, pixel_size=None):
	# shape/pixel_size are accepted for backend-signature parity but unused: Cheetah screens carry their own geometry.
	return _get_element(simulator, element_name).reading.mT * 65535


def get_screen_image_array_size(simulator, element_name: str, shape=None, index: int = 0):
	return _get_element(simulator, element_name).resolution[index]


def get_screen_resolution(simulator, element_name: str, pixel_size=None) -> float:
	return _get_element(simulator, element_name).pixel_size[0] * 1e6


def get_cavity_modecfg(simulator, element_name: str) -> str:
	return "ACCEL_STDBY"


def get_screen_pneumatic(simulator, element_name: str) -> float:
	return 1.0 if bool(_get_element(simulator, element_name).is_active) else 0.0


def set_screen_pneumatic(simulator, element_name: str, value: float):
	_get_element(simulator, element_name).is_active = 1.0 if bool(value) else 0.0


def _get_screen_centroid(simulator, element_name: str, axis: str) -> float:
	beam = _get_element(simulator, element_name).get_read_beam()
	return getattr(beam, axis).mean().item() * 1e3


def get_screen_x(simulator, element_name: str) -> float:
	return _get_screen_centroid(simulator, element_name, "x")


def get_screen_y(simulator, element_name: str) -> float:
	return _get_screen_centroid(simulator, element_name, "y")
