from lume_cheetah import CheetahSimulator


def get_cheetah_device_names(simulator: CheetahSimulator) -> list[str]:
    """Get the device names for the Cheetah simulator instance."""
    element_names = [element.name.upper() for element in simulator.segment.elements]
    return element_names
    