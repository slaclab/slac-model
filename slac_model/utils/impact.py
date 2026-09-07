_IMPACT_SUPPORTED_TYPES = {"quadrupole", "write_beam", "solrf"}


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
