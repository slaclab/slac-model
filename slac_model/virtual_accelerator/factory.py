"""
Virtual accelerator factory module. Takes in beam dynamics simulations and 
creates virtual accelerator instances based on the provided simulation data and info from slac-db

"""

import logging

from typing import Any
from slac_db.device import get_all_accessors, get_attribute, get_devices
from lume.model import LUMEModel
from slac_model.virtual_accelerator.backend_utils import (
    get_model_class,
    get_simulator_device_names,
    _get_backend_config,
    _validate_backend_functions,
)
from . import ACCESSOR_MAPPING

logger = logging.getLogger(__name__)


def get_device_type(device_name: str) -> str:
    """ Get the device type for the given device name from slac-db """
    return get_attribute(device_name, "device_type")


def _get_variable_class_importable(actions_module: Any, variable_class_name: str, conversion_module: Any = None) -> type:
    """Get an importable Variable class from the actions module.
    
    Returns the class as-is without creating a dynamic subclass. The class
    automatically loads its conversion module via model_post_init().
    
    If conversion_module is provided, validates that required backend functions exist.
    
    Parameters
    ----------
    actions_module : module
        The actions module (e.g., actions.py or torch_actions.py)
    variable_class_name : str
        Name of the variable class to retrieve
    conversion_module : module, optional
        The backend conversion module for validating required functions
        
    Returns
    -------
    type
        The Variable class
        
    Raises
    ------
    AttributeError
        If the variable class doesn't exist in the actions module
        or required backend functions don't exist in conversion_module
    """
    if not hasattr(actions_module, variable_class_name):
        raise AttributeError(
            f"Variable class '{variable_class_name}' not found in {actions_module.__name__}"
        )
    
    variable_class = getattr(actions_module, variable_class_name)
    
    # Validate backend functions if conversion module is provided
    if conversion_module is not None:
        _validate_backend_functions(variable_class, conversion_module, variable_class_name)
    
    return variable_class


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
    backend_config = _get_backend_config(simulator)
    actions_module = backend_config['actions_module']
    conversion_module = backend_config['conversion_module']

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
        for accessor, variable_class_name in accessor_mapping.items():
            if accessor not in device_accessors:
                continue
            
            # get pv for the accessor
            pv = device_accessors[accessor]
            
            try:
                # Get the importable variable class directly (no dynamic subclass needed).
                # The class automatically loads its conversion module via model_post_init().
                # Backend functions are validated to ensure they exist in the conversion module.
                _cls = _get_variable_class_importable(actions_module, variable_class_name, conversion_module)
                # create the LUME variable for this accessor and add it to the list
                lume_variables.append(_cls(name=pv, element_name=device_name))
            except AttributeError as e:
                # Skip this action if the class doesn't exist or required backend functions are missing
                logger.warning(
                    "Skipping variable %s for device %s: %s",
                    variable_class_name,
                    device_name,
                    str(e),
                )
            
    virtual_accelerator = lume_model_module(
        simulator,
        action_variables=lume_variables,
    )

    return virtual_accelerator
