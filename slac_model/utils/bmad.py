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
