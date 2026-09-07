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


def slice_lattice_cheetah(
	segment,
	first_element: str | None = None,
	last_element: str | None = None,
	include_first: bool = True,
	include_last: bool = True,
):
	"""Return a new Segment restricted to the elements between first_element and last_element."""
	return segment.subcell(
		start=first_element, end=last_element, include_start=include_first, include_end=include_last
	)
