import re


_BMAD_KLYSTRON_PATTERN = re.compile(r"^K\d{2}_\d[A-Z]#?$")
_BMAD_ELEMENT_TYPE_MAPPING = {
	"VKicker": "VerticalCorrector",
	"HKicker": "HorizontalCorrector",
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


def _get_ordered_element_names(tao) -> list[str]:
	return [name for name in tao.lat_list("*", "ele.name") if name not in ("BEGINNING", "END")]


def slice_lattice_bmad(
	tao,
	first_element: str | None = None,
	last_element: str | None = None,
	include_first: bool = True,
	include_last: bool = True,
) -> tuple[str, str]:
	"""
	Resolve the (first, last) element names bounding the requested slice of `tao`'s
	lattice, for use with Tao's `-slice_lattice` init flag (which only supports
	slicing at construction time, hence returning bounds rather than a sliced `tao`).

	`first_element`/`last_element` default to the lattice's own first/last element.
	When `include_first`/`include_last` is `False`, the corresponding boundary is
	shifted to the next/previous element in the lattice.
	"""
	names = _get_ordered_element_names(tao)

	first_index = names.index(first_element) if first_element is not None else 0
	last_index = names.index(last_element) if last_element is not None else len(names) - 1

	if not include_first:
		first_index += 1
	if not include_last:
		last_index -= 1

	if first_index > last_index:
		raise ValueError(
			"Resulting slice is empty: first_element is after last_element once exclusions are applied."
		)

	return names[first_index], names[last_index]
