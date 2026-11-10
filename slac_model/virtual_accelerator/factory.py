"""
Virtual accelerator factory module. Takes in beam dynamics simulations and 
creates virtual accelerator instances based on the provided simulation data and info from slac-db

"""

import logging

from typing import Any
from slac_db.device import get_all_accessors, get_attribute, get_devices
from lume.model import LUMEModel
from slac_model.virtual_accelerator.backend_utils import (
    get_variable_class,
    get_model_class,
    get_simulator_device_names
)
from . import ACCESSOR_MAPPING

logger = logging.getLogger(__name__)


def get_device_type(device_name: str) -> str:
    """ Get the device type for the given device name from slac-db """
    return get_attribute(device_name, "device_type")

def create_virtual_accelerator(simulator: Any, beampath: str) -> LUMEModel:
    """ Create a virtual accelerator instance based on the provided simulator instance and beampath name

    Parameters:
    -----------
    simulator : Any
        The simulator instance (e.g., Tao for BMAD simulations).
    beampath : str
        The name of the beampath for which to create the virtual accelerator.

    Returns:
    --------
    LUMEModel
        The virtual accelerator instance based on the simulator.
    """

    # get device names and model class for the simulator
    simulated_device_names = get_simulator_device_names(simulator)
    lume_model_module = get_model_class(simulator)

    known_devices = set(get_devices(include_inactive=True, beampath=beampath))

    # for each device in the simulation create the lume variables needed to support that device
    lume_variables = []
    for device_name in simulated_device_names:
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
            # get the variable class for the accessor, with backend already configured
            _cls = get_variable_class(simulator, variable_class)

            # get pv for the accessor
            pv = device_accessors[accessor]

            # create the LUME variable for this accessor and add it to the list
            lume_variables.append(_cls(name=pv, element_name=device_name))
            
    virtual_accelerator = lume_model_module(
        simulator,
        action_variables=lume_variables,
    )

    return virtual_accelerator
