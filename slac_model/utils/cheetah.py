_CHEETAH_ELEMENT_TYPE_ALIASES = {
	"TransverseDeflectingCavity": "Crab_Cavity",
}


def get_all_element_types_cheetah(simulator) -> dict[str, str]:
	"""Return normalized Cheetah element names mapped to their element types."""
	element_types = {}
	for element in simulator.segment.elements:
		element_name = element.name.split("#", 1)[0]
		element_type = type(element).__name__
		element_type = _CHEETAH_ELEMENT_TYPE_ALIASES.get(element_type, element_type)
		element_types.setdefault(element_name, element_type)
	return element_types
