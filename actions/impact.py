from lume_impact import Impact

def set_quadrupole_bctrl(simulator: Impact, element_name: str, value: float):
    effective_length = simulator.ele[element_name]["effective_length"]
    simulator.ele[element_name]["bl_gradient"] = - value / (10 * effective_length)

def get_quadrupole_bctrl(simulator: Impact, element_name: str) -> float:
    effective_length = simulator.ele[element_name]["effective_length"]
    return - simulator.ele[element_name]["bl_gradient"] * effective_length * 10