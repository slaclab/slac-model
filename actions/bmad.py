from pytao import Tao

def set_element_attribute(tao: Tao, element_name: str, attribute_name: str, value):
    """
    Set an attribute of a Bmad element.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the Bmad element whose attribute is to be set.
    attribute_name : str
        The name of the attribute to set.
    value : any
        The value to set the attribute to.

    """
    tao.cmd(f"set ele {element_name} {attribute_name} = {value}")


def validate_element(element_type: str):
    def decorator(func):
        def wrapper(tao: Tao, element_name: str, *args, **kwargs):
            element_attributes = tao.ele_gen_attribs(element_name)
            if element_attributes["TYPE"].lower() != element_type.lower():
                raise ValueError(f"Element {element_name} is not of type {element_type}")
            return func(tao, element_name, *args, **kwargs)
        return wrapper
    return decorator


def _make_bctrl_funcs(element_type, field_attr, to_bctrl, from_bctrl):
    @validate_element(element_type=element_type)
    def get_bctrl(tao: Tao, element_name: str):
        return to_bctrl(tao.ele_gen_attribs(element_name))

    @validate_element(element_type=element_type)
    def set_bctrl(tao: Tao, element_name: str, value: float):
        attrs = tao.ele_gen_attribs(element_name)
        set_element_attribute(tao, element_name, field_attr, from_bctrl(attrs, value))

    return get_bctrl, set_bctrl


get_quadrupole_bctrl, set_quadrupole_bctrl = _make_bctrl_funcs(
    element_type="quadrupole",
    field_attr="B1_GRADIENT",
    to_bctrl=lambda attrs: -attrs["B1_GRADIENT"] * attrs["L"] * 10,
    from_bctrl=lambda attrs, value: -value / (attrs["L"] * 10),
)

get_solenoid_bctrl, set_solenoid_bctrl = _make_bctrl_funcs(
    element_type="solenoid",
    field_attr="BS_FIELD",
    to_bctrl=lambda attrs: -attrs["BS_FIELD"] * 10,
    from_bctrl=lambda attrs, value: -value / 10,
)

get_kicker_bctrl, set_kicker_bctrl = _make_bctrl_funcs(
    element_type="kicker",
    field_attr="BL_KICK",
    to_bctrl=lambda attrs: -attrs["BL_KICK"] * 10,
    from_bctrl=lambda attrs, value: -value / 10,
)

def get_bpm_loc(tao: Tao, element_name: str, coordinate: str) -> float:
    """
    Get the beam centroid along the `coordinate` direction in mm.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the BPM element.
    coordinate : str
        The coordinate to retrieve (`x` or `y`).

    Returns:
    --------
    float
        The location of the BPM in the specified coordinate.
    """
    if coordinate.lower() not in ['x', 'y']:
        raise ValueError("Coordinate must be 'x' or 'y'")
    return getattr(tao.ele(element_name).orbit, coordinate.lower()) * 1e3

get_bpm_x = lambda tao, element_name: get_bpm_loc(tao, element_name, 'x')
get_bpm_y = lambda tao, element_name: get_bpm_loc(tao, element_name, 'y')

def get_klystron_enld(tao: Tao, element_name: str) -> float:
    """
    Get the energy loss per turn (ENLD) of a klystron in MeV.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the klystron element.

    Returns:
    --------
    float
        The energy gain of the klystron in MeV.
    """
    return tao.ele(element_name).control_vars["ENLD_MEV"] * 1e-6

def set_klystron_enld(tao: Tao, element_name: str, value: float):
    """
    Set the energy loss per turn (ENLD) of a klystron in MeV.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the klystron element.
    value : float
        The energy gain to set for the klystron in MeV.
    """
    set_element_attribute(tao, element_name, "ENLD_MEV", value * 1e6)


def get_klystron_pdes(tao: Tao, element_name: str) -> float:
    """
    Get the phase of the klystron in degrees.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the klystron element.

    Returns:
    --------
    float
        The phase of the klystron in degrees.
    """
    return tao.ele(element_name).control_vars["PHASE_DEG"]

def set_klystron_pdes(tao: Tao, element_name: str, value: float):
    """
    Set the phase of the klystron in degrees.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the klystron element.
    value : float
        The phase to set for the klystron in degrees.
    """
    set_element_attribute(tao, element_name, "PHASE_DEG", value)

def set_klystron_stat(tao: Tao, element_name: str, value: int):
    """
    Set the status of the klystron.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the klystron element.
    value : int
        The status to set for the klystron (0 for off, 1 for on).
    """

    _logic_mapping = {0: True, 1: False}  

    if value not in [0, 1]:
        raise ValueError("Status must be 0 (off) or 1 (on)")
    set_element_attribute(tao, element_name, "STAT", _logic_mapping[value])

def get_klystron_stat(tao: Tao, element_name: str) -> int:
    """
    Get the status of the klystron.

    Parameters:
    -----------
    tao : Tao
        An instance of the Tao class.
    element_name : str
        The name of the klystron element.

    Returns:
    --------
    int
        The status of the klystron (0 for off, 1 for on).
    """
    _logic_mapping = {True: 0, False: 1}  
    stat_value = tao.ele(element_name).control_vars["STAT"]
    return _logic_mapping[stat_value]