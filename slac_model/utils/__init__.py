from .bmad import get_all_element_types_bmad, slice_lattice_bmad
from .impact import get_all_element_types_impact, slice_lattice_impact
from .cheetah import get_all_element_types_cheetah, slice_lattice_cheetah


def _isinstance_if_importable(simulator, import_path: str, class_name: str) -> bool:
	"""Return whether `simulator` is an instance of `class_name`, tolerating a missing optional dependency."""
	try:
		module = __import__(import_path, fromlist=[class_name])
		cls = getattr(module, class_name)
	except ImportError:
		return False
	return isinstance(simulator, cls)


def get_all_element_types(simulator) -> dict[str, str]:
	"""Dispatch element-type extraction based on the simulator interface."""

	if _isinstance_if_importable(simulator, "pytao", "Tao"):
		return get_all_element_types_bmad(simulator)

	if _isinstance_if_importable(simulator, "impact", "Impact"):
		return get_all_element_types_impact(simulator)

	if _isinstance_if_importable(simulator, "lume_cheetah.simulator", "CheetahSimulator"):
		return get_all_element_types_cheetah(simulator)

	raised_type = type(simulator).__name__
	raised_module = type(simulator).__module__
	raise TypeError(
		f"Unsupported simulator type {raised_module}.{raised_type}; expected "
		"a Bmad Tao, Impact simulator, or Cheetah CheetahSimulator."
	)


def slice_lattice(
	simulator,
	first_element=None,
	last_element=None,
	include_first: bool = True,
	include_last: bool = True,
):
	"""Dispatch lattice-restriction to the backend matching `simulator`'s type."""

	if _isinstance_if_importable(simulator, "pytao", "Tao"):
		return slice_lattice_bmad(
			simulator, first_element, last_element, include_first=include_first, include_last=include_last
		)

	if _isinstance_if_importable(simulator, "impact", "Impact"):
		return slice_lattice_impact(
			simulator, first_element, last_element, include_first=include_first, include_last=include_last
		)

	if _isinstance_if_importable(simulator, "cheetah.accelerator", "Segment"):
		return slice_lattice_cheetah(
			simulator, first_element, last_element, include_first=include_first, include_last=include_last
		)

	raised_type = type(simulator).__name__
	raised_module = type(simulator).__module__
	raise TypeError(
		f"Unsupported simulator type {raised_module}.{raised_type}; expected "
		"a Bmad Tao, Impact simulator, or Cheetah Segment."
	)
