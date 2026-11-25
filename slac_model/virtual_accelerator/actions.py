"""Generic lume `Variable` subclasses for non-torch simulator backends.

This module provides only the base ScalarVariable and EnumVariable classes
without any backend binding. These are designed to be used through 
get_variable_class(simulator, "VariableClassName") which automatically 
detects the backend and binds the appropriate conversion module.

Device-specific classes (QuadrupoleBCTRLVariable, etc.) are defined in:
- bmad_actions.py for Bmad backend with auto-loading
- torch_actions.py for Cheetah backend with auto-loading  
- Generated on-the-fly by get_variable_class() for any backend

Usage - Recommended (auto-detects backend):
    from slac_model.virtual_accelerator.backend_utils import get_variable_class
    quad_var = get_variable_class(simulator, "QuadrupoleBCTRLVariable")(
        name="Q1", element_name="Q1_element"
    )

"""

from lume.variables import ScalarVariable as LumeScalarVariable
from lume.variables import EnumVariable as LumeEnumVariable

from slac_model.virtual_accelerator._actions_base import AgnosticSimulatorMixin


class ScalarVariable(LumeScalarVariable, AgnosticSimulatorMixin):
    """Base scalar variable without backend binding.
    
    Use get_variable_class() to automatically create backend-specific subclasses
    with proper conversion module binding based on simulator type.
    """
    pass


class EnumVariable(LumeEnumVariable, AgnosticSimulatorMixin):
    """Base enum variable without backend binding.
    
    Use get_variable_class() to automatically create backend-specific subclasses
    with proper conversion module binding based on simulator type.
    """
    pass
