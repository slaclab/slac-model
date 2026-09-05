from lume_impact import Impact

def set_quadrupole_bctrl(simulator: Impact, element_name: str, value: float):
    effective_length = simulator.ele[element_name]["L_effective"]
    simulator.ele[element_name]["b1_gradient"] = -value / (10 * effective_length)

def get_quadrupole_bctrl(simulator: Impact, element_name: str) -> float:
    effective_length = simulator.ele[element_name]["L_effective"]
    return -simulator.ele[element_name]["b1_gradient"] * effective_length * 10


get_quadrupole_bact = get_quadrupole_bctrl