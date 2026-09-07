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


def slice_lattice_impact(
	impact,
	first_element: str | float | None = None,
	last_element: str | float | None = None,
	include_first: bool = True,
	include_last: bool = True,
):
	"""Restrict `impact`'s lattice in place to the elements between first_element and last_element."""

	def resolve_s(location):
		if isinstance(location, str):
			try:
				return impact.ele[location]["s"]
			except KeyError:
				raise ValueError(f"Element {location!r} not found in the impact model.")
		return float(location)

	start_s = resolve_s(first_element) if first_element is not None else None
	stop_s = resolve_s(last_element) if last_element is not None else None

	def keep(s) -> bool:
		if s is None:
			return start_s is None and stop_s is None
		if start_s is not None:
			if s < start_s or (s == start_s and not include_first):
				return False
		if stop_s is not None:
			if s > stop_s or (s == stop_s and not include_last):
				return False
		return True

	if stop_s is not None:
		impact.stop = stop_s

	impact.ele = {k: v for k, v in impact.ele.items() if keep(v.get("s"))}
	impact.input["lattice"] = [elem for elem in impact.lattice if keep(elem.get("s"))]
	return impact
