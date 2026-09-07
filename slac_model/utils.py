import re


_BMAD_KLYSTRON_PATTERN = re.compile(r"^K\d{2}_\d[A-Z]#?$")
_BMAD_ELEMENT_TYPE_MAPPING = {
	"VKicker": "VerticalCorrector",
	"HKicker": "HorizontalCorrector",
}
_IMPACT_SUPPORTED_TYPES = {"quadrupole", "write_beam", "solrf"}
_CHEETAH_ELEMENT_TYPE_ALIASES = {
	"TransverseDeflectingCavity": "Crab_Cavity",
}


def get_all_element_types_bmad(tao) -> dict[str, str]:
	"""Return normalized Bmad element names mapped to their element types."""
	elements = [
		name
		for name in dict.fromkeys(tao.lat_list("*", "ele.name"))
		if name not in ("BEGINNING", "END")
	]
	normalized_names = []
	for name in elements:
		base_name = name.split("#", 1)[0]
		if _BMAD_KLYSTRON_PATTERN.match(base_name):
			base_name = base_name[:-1]
		normalized_names.append(base_name)

	normalized_names = list(dict.fromkeys(normalized_names))
	element_types = {}
	for name in normalized_names:
		element_type = tao.ele_head(name)["key"]
		if element_type == "Monitor" and name.startswith("BPM"):
			element_type = "BPM"
		elif element_type == "Monitor" and (
			name.startswith("OTR")
			or name.startswith("PR")
			or name.startswith("YAG")
		):
			element_type = "Screen"
		elif element_type == "Overlay" and name.startswith("K"):
			element_type = "Klystron"
		element_types[name] = _BMAD_ELEMENT_TYPE_MAPPING.get(element_type, element_type)
	return element_types


def get_all_element_types_impact(impact) -> dict[str, str]:
	"""Return supported Impact element names mapped to normalized types."""
	normalized_types = {}
	for name, element in impact.ele.items():
		element_type = element["type"]
		if element_type not in _IMPACT_SUPPORTED_TYPES:
			continue
		if element_type == "quadrupole":
			element_type = "Quadrupole"
		elif element_type == "write_beam":
			element_type = "Screen"
		normalized_types[name] = element_type
	return normalized_types


def get_all_element_types_cheetah(segment) -> dict[str, str]:
	"""Return normalized Cheetah element names mapped to their element types."""
	element_types = {}
	for element in segment.elements:
		element_name = element.name.split("#", 1)[0]
		element_type = type(element).__name__
		element_type = _CHEETAH_ELEMENT_TYPE_ALIASES.get(element_type, element_type)
		element_types.setdefault(element_name, element_type)
	return element_types


def get_all_element_types(simulator) -> dict[str, str]:
	"""Dispatch element-type extraction based on the simulator interface."""

	from pytao import Tao
	from impact import Impact
	from cheetah.accelerator import Segment

	if isinstance(simulator, Tao):
		return get_all_element_types_bmad(simulator)

	if isinstance(simulator, Impact):
		return get_all_element_types_impact(simulator)

	if isinstance(simulator, Segment):
		return get_all_element_types_cheetah(simulator)

	raised_type = type(simulator).__name__
	raised_module = type(simulator).__module__
	raise TypeError(
		f"Unsupported simulator type {raised_module}.{raised_type}; expected "
		"a Bmad Tao, Impact simulator, or Cheetah segment."
	)
