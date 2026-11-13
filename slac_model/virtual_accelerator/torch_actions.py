"""Generic lume `Variable` subclasses for Cheetah (torch-based) simulator backend.

This module provides action classes based on TorchScalarVariable from lume_torch
that work with the Cheetah simulator. These variables use torch-based operations
and can be instantiated with a conversion module bound per instance.

The device-specific base mixins are defined in _actions_base.py. This module
combines them with the backend-specific TorchScalarVariable base class to create
the final variable classes.

Usage:
    from slac_model.virtual_accelerator.torch_actions import QuadrupoleBCTRLVariable
    from slac_model.virtual_accelerator import backend_utils
    
    # Create a Cheetah simulator instance
    cheetah_sim = create_cheetah_simulator()
    
    # Create and configure a variable
    quad_var = QuadrupoleBCTRLVariable(name="Q1", element_name="Q1")
    quad_var.set_instance_conversion_module(backend_utils.cheetah_conversion)
    
    # Now you can get/set values
    value = quad_var._get(cheetah_sim)
"""

from lume.variables import EnumVariable as LumeEnumVariable
from lume_torch.variables import TorchScalarVariable as LumeTorchScalarVariable

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
    _StatusVariableMixin,
    _BminVariableMixin,
    _BmaxVariableMixin,
    _ControlStateVariableMixin,
    _BPMXVariableMixin,
    _BPMYVariableMixin,
    _BPMTMITDummyVariableMixin,
    _KlystronENLDVariableMixin,
    _KlystronPDESVariableMixin,
    _KlystronPACTVariableMixin,
    _KlystronStatVariableMixin,
    _CavityAREQVariableMixin,
    _CavityAREQReadbackVariableMixin,
    _CavityPREQVariableMixin,
    _CavityPREQReadbackVariableMixin,
    _DummyEnumVariableMixin,
    _CavityMODECFGVariableMixin,
)


class ScalarVariable(LumeTorchScalarVariable, AgnosticSimulatorMixin):
    """Base torch scalar variable with conversion module support for Cheetah simulator."""


class EnumVariable(LumeEnumVariable, AgnosticSimulatorMixin):
    """Base enum variable with conversion module support for Cheetah simulator."""


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
# Status Variables
# ============================================================================


class StatusVariable(ScalarVariable, _StatusVariableMixin):
    """Action that operates on the status of a device (e.g. STATCTRLSUB.T)"""


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


class KlystronStatVariable(ScalarVariable, _KlystronStatVariableMixin):
    """
    Action that operates on the status of a klystron which acts on an overlay.
    Note: the element_name for this variable should be set for an overlay element.

    Values other than 1.0 (e.g. 0.0, 4.0 for bypassed/standby beamcodes) are
    treated as "off" and set the overlay's IN_USE to 0.
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


class DummyEnumVariable(EnumVariable, _DummyEnumVariableMixin):
    """
    Dummy variable for testing purposes. This variable does not correspond to any real element or property.
    It is used to test the behavior of the model when a variable is requested that does not exist in the model.
    """


class CavityMODECFGVariable(EnumVariable, _CavityMODECFGVariableMixin):
    """
    Action that operates on the mode configuration property of a cavity.
    """
