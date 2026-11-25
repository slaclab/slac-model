"""Generic lume `Variable` subclasses for Bmad (non-torch) simulator backend.

This module provides action classes based on ScalarVariable from lume that work
with the Bmad/Tao simulator. These variables automatically load the Bmad
conversion module on instantiation through model_post_init().

The device-specific base mixins are defined in _actions_base.py. This module
combines them with the backend-specific ScalarVariable base class to create
the final variable classes.

Usage:
    from slac_model.virtual_accelerator.bmad_actions import QuadrupoleBCTRLVariable
    
    # Create a variable - conversion module is auto-loaded
    quad_var = QuadrupoleBCTRLVariable(name="Q1", element_name="Q1_element")
    
    # Backend is already set up, ready to use
    value = quad_var._get(bmad_simulator)
"""

from lume.variables import ScalarVariable as LumeScalarVariable
from lume.variables import EnumVariable as LumeEnumVariable

from slac_model.virtual_accelerator._actions_base import (
    AgnosticSimulatorMixin,
    _QuadrupoleBCTRLVariableMixin,
    _QuadrupoleBACTVariableMixin,
    _SolenoidBCTRLVariableMixin,
    _SolenoidBACTVariableMixin,
    _CavityAMPLVariableMixin,
    _CavityPHASVariableMixin,
    _HKickerBCTRLVariableMixin,
    _HKickerBACTVariableMixin,
    _VKickerBCTRLVariableMixin,
    _VKickerBACTVariableMixin,
    _BminVariableMixin,
    _BmaxVariableMixin,
    _ControlStateVariableMixin,
    _BPMXVariableMixin,
    _BPMYVariableMixin,
    _BPMTMITDummyVariableMixin,
    _KlystronENLDVariableMixin,
    _KlystronPDESVariableMixin,
    _KlystronPACTVariableMixin,
    _KlystronBeamcode1StatVariableMixin,
    _KlystronBeamcode2StatVariableMixin,
    _CavityAREQVariableMixin,
    _CavityAREQReadbackVariableMixin,
    _CavityPREQVariableMixin,
    _CavityPREQReadbackVariableMixin,
    _DummyEnumVariableMixin,
    _CavityMODECFGVariableMixin,
)


class ScalarVariable(LumeScalarVariable, AgnosticSimulatorMixin):
    """Base Bmad scalar variable with Bmad conversion module support.
    
    Automatically loads the Bmad conversion module during initialization
    via model_post_init(), enabling direct variable creation without get_variable_class().
    """
    
    def model_post_init(self, __context) -> None:
        """Load Bmad conversion module on instantiation."""
        try:
            from slac_model.pv_conversion import bmad
            self.set_instance_conversion_module(bmad)
        except ImportError as e:
            raise RuntimeError(
                f"Cannot load Bmad conversion module for {type(self).__name__}. "
                f"Error: {e}"
            ) from e


class EnumVariable(LumeEnumVariable, AgnosticSimulatorMixin):
    """Base Bmad enum variable with Bmad conversion module support.
    
    Automatically loads the Bmad conversion module during initialization
    via model_post_init(), enabling direct variable creation without get_variable_class().
    """
    
    def model_post_init(self, __context) -> None:
        """Load Bmad conversion module on instantiation."""
        try:
            from slac_model.pv_conversion import bmad
            self.set_instance_conversion_module(bmad)
        except ImportError as e:
            raise RuntimeError(
                f"Cannot load Bmad conversion module for {type(self).__name__}. "
                f"Error: {e}"
            ) from e


# ============================================================================
# Quadrupole Variables
# ============================================================================


class QuadrupoleBCTRLVariable(ScalarVariable, _QuadrupoleBCTRLVariableMixin):
    """Action that operates on the BCTRL/BDES property of Quadrupoles"""
    read_only: bool = False


class QuadrupoleBACTVariable(QuadrupoleBCTRLVariable, _QuadrupoleBACTVariableMixin):
    """Action that operates on the BACT property of Quadrupoles"""


# ============================================================================
# Solenoid Variables
# ============================================================================


class SolenoidBCTRLVariable(ScalarVariable, _SolenoidBCTRLVariableMixin):
    """Action that operates on the BCTRL/BDES property of Solenoids"""


class SolenoidBACTVariable(SolenoidBCTRLVariable, _SolenoidBACTVariableMixin):
    """Action that operates on the BACT property of Solenoids"""


# ============================================================================
# Kicker Variables (Horizontal)
# ============================================================================


class HKickerBCTRLVariable(ScalarVariable, _HKickerBCTRLVariableMixin):
    """Action that operates on the BCTRL/BDES property of Horizontal Kicker magnets"""


class HKickerBACTVariable(HKickerBCTRLVariable, _HKickerBACTVariableMixin):
    """Action that operates on the BACT property of Horizontal Kicker magnets"""


# ============================================================================
# Kicker Variables (Vertical)
# ============================================================================


class VKickerBCTRLVariable(ScalarVariable, _VKickerBCTRLVariableMixin):
    """Action that operates on the BCTRL/BDES property of Vertical Kicker magnets"""


class VKickerBACTVariable(VKickerBCTRLVariable, _VKickerBACTVariableMixin):
    """Action that operates on the BACT property of Vertical Kicker magnets"""


# ============================================================================
# Limit Variables
# ============================================================================


class BminVariable(ScalarVariable, _BminVariableMixin):
    """Action that operates on the BMIN/DRVL property of a device"""


class BmaxVariable(ScalarVariable, _BmaxVariableMixin):
    """Action that operates on the BMAX/DRVH property of a device"""


# ============================================================================
# Control State Variables
# ============================================================================


class ControlStateVariable(EnumVariable, _ControlStateVariableMixin):
    """Action that operates on the control state (e.g. CTRL) of a device"""


# ============================================================================
# BPM Variables
# ============================================================================


class BPMXVariable(ScalarVariable, _BPMXVariableMixin):
    """Action that operates on the X position of a BPM"""


class BPMYVariable(ScalarVariable, _BPMYVariableMixin):
    """Action that operates on the Y position of a BPM"""


class BPMTMITDummyVariable(ScalarVariable, _BPMTMITDummyVariableMixin):
    """Dummy variable for BPM TMIT (total beam intensity) for testing purposes."""


# ============================================================================
# Klystron Variables
# ============================================================================


class KlystronENLDVariable(ScalarVariable, _KlystronENLDVariableMixin):
    """
    Action that operates on the amplitude of a klystron which acts on an overlay.
    """


class KlystronPDESVariable(ScalarVariable, _KlystronPDESVariableMixin):
    """
    Action that operates on the phase of a klystron which acts on an overlay.
    """


class KlystronPACTVariable(KlystronPDESVariable, _KlystronPACTVariableMixin):
    """
    Action that operates on the actual phase of a klystron which acts on an overlay.
    """


class KlystronBeamcode1StatVariable(ScalarVariable, _KlystronBeamcode1StatVariableMixin):
    """
    Klystron BEAMCODE1 status (CU_HXR): PV == 1 -> on (IN_USE=1), else off.
    element_name must be the klystron's overlay element.
    """


class KlystronBeamcode2StatVariable(ScalarVariable, _KlystronBeamcode2StatVariableMixin):
    """
    Klystron BEAMCODE2 status (CU_SXR): PV == 2 -> on (IN_USE=1), else off.
    element_name must be the klystron's overlay element.
    """


# ============================================================================
# Cavity Variables
# ============================================================================


class CavityAREQVariable(ScalarVariable, _CavityAREQVariableMixin):
    """
    Action that operates on the amplitude property of a cavity.
    """


class CavityAREQReadbackVariable(CavityAREQVariable, _CavityAREQReadbackVariableMixin):
    """Read-only variant of cavity amplitude request variable."""


class CavityPREQVariable(ScalarVariable, _CavityPREQVariableMixin):
    """
    Action that operates on the phase property of a cavity.
    """


class CavityPREQReadbackVariable(CavityPREQVariable, _CavityPREQReadbackVariableMixin):
    """Read-only variant of cavity phase request variable."""


class DummyEnumVariable(_DummyEnumVariableMixin, EnumVariable):
    """
    Dummy variable for testing purposes. This variable does not correspond to any real element or property.
    It is used to test the behavior of the model when a variable is requested that does not exist in the model.
    """


class CavityMODECFGVariable(EnumVariable, _CavityMODECFGVariableMixin):
    """
    Action that operates on the mode configuration property of a cavity.
    """
