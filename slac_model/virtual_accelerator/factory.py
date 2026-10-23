"""
Virtual accelerator factory module. Takes in beam dynamics simulations and 
creates virtual accelerator instances based on the provided simulation data and info from slac-db

"""

from pytao import Tao
from typing import Any
from slac_db.devices import get_device_accessors, get_device_types
from slac_model.model_tools.bmad import get_bmad_device_names
from slac_model.virtual_accelerator import actions as bmad_actions
from lume_bmad.model import LUMEBmadModel
from lume.model import LUMEModel
from . import ACCESSOR_MAPPING


def get_simulator_device_names(simulator: Any):
    """ get all the slac-db device names for the given simulator instance """
    if isinstance(simulator, Tao):
        return get_bmad_device_names(simulator)
    else:
        raise ValueError("Unsupported simulator type")

def get_lume_variable_module(simulator: Any):
    """ Identify the appropriate LUME variable module for the given simulator instance """
    if isinstance(simulator, Tao):
        return bmad_actions
    else:
        raise ValueError("Unsupported simulator type")

def get_lume_model_module(simulator: Any):
    """ Identify the appropriate LUME model module for the given simulator instance """
    if isinstance(simulator, Tao):
        return LUMEBmadModel
    else:
        raise ValueError("Unsupported simulator type")

def create_virtual_accelerator(simulator: Any) -> LUMEModel:
    """ Create a virtual accelerator instance based on the provided simulator instance 

    Parameters:
    -----------
    simulator : Any
        The simulator instance (e.g., Tao for BMAD simulations).

    Returns:
    --------
    LUMEModel
        The virtual accelerator instance based on the simulator.
    """

    # get device names, lume variable module, and lume model module for the simulator
    device_names = get_simulator_device_names(simulator)
    lume_variable_module = get_lume_variable_module(simulator)
    lume_model_module = get_lume_model_module(simulator)

    # get all slac-db accessor dicts
    all_device_accessors = {name: get_device_accessors(name) for name in device_names}

    # for each device in the BMAD simulation create the lume variables needed to support that device
    lume_variables = []
    for device_name in device_names:
        device_accessors = all_device_accessors.get(device_name)
        device_type = get_device_types(device_name)

        # for each accessor in the device type's accessor mapping create the corresponding LUME variable
        for accessor, variable_class in ACCESSOR_MAPPING.get(device_type, {}).items():
            # get the variable class for the accessor
            _cls = getattr(lume_variable_module, variable_class)

            # get pv for the accessor
            pv = device_accessors.get(accessor)

            # create the LUME variable for this accessor and add it to the list
            lume_variables.append(_cls(pv))
            
    virtual_accelerator = lume_model_module(
        tao=simulator,
        action_variables=lume_variables,
    )

    return virtual_accelerator

