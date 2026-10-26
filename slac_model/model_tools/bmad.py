from pytao import Tao
import re
KLYSTRON_PATTERN = re.compile(r"^K\d+_\d+[A-Z]$")

def get_bmad_device_names(tao: Tao):
    """Return slac-db device names for the given BMAD Tao instance.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.

    Returns
    -------
    list[str]
        Ordered element names with slac-db device names.
    """
    elements = tao.lat_list("*", "ele.name")

    # Remove sentinel elements and preserve Tao's unique element IDs to avoid
    # ambiguous lookups when multiple elements share the same base name.
    elements = list(
        dict.fromkeys(elem for elem in elements if elem not in ("BEGINNING", "END"))
    )

    normalized_elements = []

    # if an element has "#<number>" suffix then it is a split element, remove the suffix -- duplicates will be removed later while preserving order
    elements = [elem.split("#")[0] for elem in elements]

    # if an element matches the klystron segment pattern similar to K21_1D#1, normalize it to the base klystron name without the segment suffix K21_1
    for elem in elements:
        match = KLYSTRON_PATTERN.match(elem)
        if match:
            normalized_elements.append(
                elem[:-1]
            )  # remove the segment suffix to get the overlay element name

        else:
            normalized_elements.append(elem)

    # remove duplicates while preserving order
    normalized_elements = list(dict.fromkeys(normalized_elements))

    return normalized_elements
