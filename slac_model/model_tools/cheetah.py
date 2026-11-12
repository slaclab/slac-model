"""Utilities for managing Cheetah simulator device names and configuration."""

import re
from lume_cheetah.simulator import CheetahSimulator

# Pattern for klystron segment elements (e.g., K21_1A, K21_1B, K21_1C, K21_1D)
KLYSTRON_PATTERN = re.compile(r"^K\d+_\d+[A-Z]$")


def get_cheetah_device_names(simulator: CheetahSimulator) -> list[str]:
    """Return normalized device names for the given Cheetah simulator instance.

    Normalizes Cheetah element names to match slac-db conventions:
    - Converts to uppercase
    - Skips marker and drift elements (not physical devices)
    - Removes duplicate elements
    - Normalizes klystron segment names to base names (e.g., K21_1A -> K21_1)

    Parameters
    ----------
    simulator : CheetahSimulator
        Active Cheetah simulator instance containing the lattice segment.

    Returns
    -------
    list[str]
        Ordered element names normalized to slac-db conventions.
    """
    normalized_elements = []

    # Iterate through elements and normalize, skipping markers and drifts
    for element in simulator.segment.elements:
        # Skip marker and drift elements (not physical devices)
        # Check by class name to avoid import issues
        elem_type = type(element).__name__
        if elem_type in ("Marker", "Drift"):
            continue
            
        elem = element.name.upper()
        
        # Normalize klystron segment names: remove the segment suffix (A-Z) to get the base name
        # This matches the pattern of K21_1A, K21_1B, etc. -> K21_1
        match = KLYSTRON_PATTERN.match(elem)
        if match:
            # Remove the segment suffix (last character) to get the base klystron name
            normalized_elements.append(elem[:-1])
        else:
            normalized_elements.append(elem)

    # Remove duplicates while preserving order
    normalized_elements = list(dict.fromkeys(normalized_elements))

    return normalized_elements
    