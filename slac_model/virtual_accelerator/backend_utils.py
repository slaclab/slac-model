"""Utilities for managing backend-specific variable classes and models.

This module provides functions to select the appropriate variable classes and
model classes based on the simulator type. Variable classes are created on-demand
for each backend using the generic definitions from actions.py, with conversion
modules bound per instance to prevent cross-backend interference.
"""

from typing import Any, Type
from pytao import Tao
from lume.model import LUMEModel
from lume.variables import Variable


def _get_backend_config(simulator: Any) -> dict[str, Any]:
    """Get backend configuration based on simulator type.
    
    This function identifies the simulator backend and returns the appropriate
    model class, device name getter function, and conversion module.
    
    Args:
        simulator: The simulator instance (e.g., Tao, CheetahSimulator).
        
    Returns:
        Dictionary containing:
            - 'model_class': LUME model class to use
            - 'get_device_names': Function to get device names from simulator
            - 'conversion_module': The conversion module for get/set functions
            - 'backend_name': String identifier for the backend
            
    Raises:
        ValueError: If the simulator type is not supported.
    """
    if isinstance(simulator, Tao):
        from lume_bmad.model import LUMEBmadModel
        from slac_model.model_tools.bmad import get_bmad_device_names
        from slac_model.pv_conversion import bmad
        
        return {
            'model_class': LUMEBmadModel,
            'get_device_names': get_bmad_device_names,
            'conversion_module': bmad,
            'backend_name': 'bmad',
        }
    else:
        raise ValueError(f"Unsupported simulator type: {type(simulator).__name__}")


def get_variable_class(
    simulator: Any,
    variable_class_name: str,
) -> Type[Variable]:
    """Get a backend-specific subclass for the given simulator.
    
    This function creates a subclass of the generic variable class that has the
    conversion module bound per instance. This ensures that variables created for
    different simulators (e.g., Bmad and Cheetah) don't interfere with each other.
    
    Args:
        simulator: The simulator instance.
        variable_class_name: Name of the variable class to retrieve.
        
    Returns:
        A subclass of the generic variable class that automatically binds the
        conversion module per instance.
        
    Raises:
        ValueError: If simulator type is unsupported.
        AttributeError: If variable class doesn't exist in actions.py or 
                       required backend functions are missing.
    """
    from slac_model.virtual_accelerator import actions
    
    backend_config = _get_backend_config(simulator)
    conversion_module = backend_config['conversion_module']
    backend_name = backend_config['backend_name']
    
    if not hasattr(actions, variable_class_name):
        raise AttributeError(
            f"Variable class '{variable_class_name}' not found in "
            f"slac_model.virtual_accelerator.actions"
        )
    
    generic_class = getattr(actions, variable_class_name)
    
    # Validate that backend functions exist before creating the subclass
    _validate_backend_functions(generic_class, conversion_module, variable_class_name)
    
    # Create a backend-specific subclass that binds the conversion module at instantiation
    class BackendSpecificVariable(generic_class):
        """Backend-specific subclass with bound conversion module."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_instance_conversion_module(conversion_module)
    
    # Preserve the original class name for debugging and introspection
    BackendSpecificVariable.__name__ = f"{generic_class.__name__}_{backend_name}"
    BackendSpecificVariable.__qualname__ = f"{generic_class.__qualname__}_{backend_name}"
    
    return BackendSpecificVariable


def get_model_class(simulator: Any) -> Type[LUMEModel]:
    """Get the LUME model class for the given simulator backend.
    
    Args:
        simulator: The simulator instance.
        
    Returns:
        The LUME model class to use for this simulator.
        
    Raises:
        ValueError: If simulator type is unsupported.
    """
    backend_config = _get_backend_config(simulator)
    return backend_config['model_class']


def _validate_backend_functions(
    variable_class: Type[Variable],
    conversion_module: Any,
    variable_class_name: str,
) -> None:
    """Validate that required backend functions exist in the conversion module.
    
    Args:
        variable_class: The variable class to check.
        conversion_module: The backend conversion module.
        variable_class_name: Name of the variable class (for error messages).
        
    Raises:
        AttributeError: If required functions don't exist in the conversion module.
    """
    # Check for get_function in model_fields
    if hasattr(variable_class, 'model_fields'):
        get_field = variable_class.model_fields.get('get_function')
        if get_field and get_field.default:
            get_func_name = get_field.default
            if not hasattr(conversion_module, get_func_name):
                raise AttributeError(
                    f"Backend function '{get_func_name}' required by {variable_class_name} "
                    f"not found in conversion module {conversion_module.__name__}"
                )
        
        # Check for set_function in model_fields
        set_field = variable_class.model_fields.get('set_function')
        if set_field and set_field.default:
            set_func_name = set_field.default
            if not hasattr(conversion_module, set_func_name):
                raise AttributeError(
                    f"Backend function '{set_func_name}' required by {variable_class_name} "
                    f"not found in conversion module {conversion_module.__name__}"
                )


def get_simulator_device_names(simulator: Any) -> list[str]:
    """Get all device names from the simulator.
    
    Args:
        simulator: The simulator instance.
        
    Returns:
        List of device names in the simulator.
        
    Raises:
        ValueError: If simulator type is unsupported.
    """
    backend_config = _get_backend_config(simulator)
    device_name_getter = backend_config['get_device_names']
    return device_name_getter(simulator)
