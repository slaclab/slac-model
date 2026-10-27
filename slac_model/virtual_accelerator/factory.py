"""
Virtual accelerator factory module. Takes in beam dynamics simulations and 
creates virtual accelerator instances based on the provided simulation data and info from slac-db

"""

import logging

from pytao import Tao
from typing import Any
from slac_db.device import get_all_accessors, get_attribute, get_devices
from slac_model.model_tools.bmad import get_bmad_device_names
from slac_model.virtual_accelerator import bmad_actions as bmad_actions
from lume_bmad.model import LUMEBmadModel
from lume.model import LUMEModel
from . import ACCESSOR_MAPPING

logger = logging.getLogger(__name__)


def get_device_type(device_name: str) -> str:
    """ Get the device type for the given device name from slac-db """
    return get_attribute(device_name, "device_type")

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

    known_devices = set(get_devices(include_inactive=True))

    # for each device in the BMAD simulation create the lume variables needed to support that device
    lume_variables = []
    for device_name in device_names:
        if device_name not in known_devices:
            continue
        device_type = get_device_type(device_name)
        accessor_mapping = ACCESSOR_MAPPING.get(device_type, {})
        if not accessor_mapping:
            continue
        device_accessors = get_all_accessors(device_name)
        missing_accessors = accessor_mapping.keys() - device_accessors.keys()
        if missing_accessors:
            logger.warning(
                "Skipping accessors missing from slac-db for %s: %s",
                device_name,
                ", ".join(sorted(missing_accessors)),
            )

        # for each accessor in the device type's accessor mapping create the corresponding LUME variable
        for accessor, variable_class in accessor_mapping.items():
            if accessor not in device_accessors:
                continue
            # get the variable class for the accessor
            _cls = getattr(lume_variable_module, variable_class)

            # get pv for the accessor
            pv = device_accessors[accessor]

            # create the LUME variable for this accessor and add it to the list
            lume_variables.append(_cls(name=pv, element_name=device_name))
            
    virtual_accelerator = lume_model_module(
        tao=simulator,
        action_variables=lume_variables,
    )

    return virtual_accelerator
