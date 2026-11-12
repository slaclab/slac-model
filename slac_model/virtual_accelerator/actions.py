"""Generic lume `Variable` subclasses for non-torch simulator backends.

This module provides action classes based on ScalarVariable from lume that work
with Bmad/Tao, Impact, and other non-torch simulators. These variables use
standard numpy-based operations and can be instantiated with a conversion module
bound per instance.

Usage:
    from slac_model.virtual_accelerator.actions import QuadrupoleBCTRLVariable
    from slac_model.virtual_accelerator import backend_utils
    
    # Create a simulator instance
    sim = create_simulator()
    
    # Create and configure a variable
    quad_var = QuadrupoleBCTRLVariable(name="Q1", element_name="Q1")
    quad_var.set_instance_conversion_module(backend_utils.conversion_module)
    
    # Now you can get/set values
    value = quad_var._get(sim)
"""

from typing import Any, Optional
from abc import ABC

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable as LumeScalarVariable
from lume.variables import EnumVariable as LumeEnumVariable


class AgnosticSimulatorMixin(ABC):
    """Mixin that provides instance-level conversion module binding."""
    element_name: str

    def get_conversion_module(self) -> Any:
        """Get the conversion module for this instance.
        
        Returns:
            The instance-level conversion module.
            
        Raises:
            RuntimeError: If no conversion module has been set.
        """
        if not hasattr(self, '_instance_conversion_module'):
            raise RuntimeError(
                f"Conversion module not set for {type(self).__name__}. "
                f"Create instances via get_variable_class() for proper backend binding."
            )
        return self._instance_conversion_module

    def set_instance_conversion_module(self, module: Any) -> None:
        """Set the instance-level conversion module.
        
        Args:
            module: The conversion module for this instance.
        """
        self._instance_conversion_module = module

    def _get(self, simulator: Any) -> Any:
        get_func_name = getattr(self, 'get_function', None)
        if get_func_name is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _get() or define get_function"
            )
        module = self.get_conversion_module()
        return getattr(module, get_func_name)(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        set_func_name = getattr(self, 'set_function', None)
        if set_func_name is None:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _set() or define set_function"
            )
        module = self.get_conversion_module()
        return getattr(module, set_func_name)(simulator, self.element_name, value)


class ScalarVariable(LumeScalarVariable, AgnosticSimulatorMixin):
    """Base scalar variable with conversion module support."""


class EnumVariable(LumeEnumVariable, AgnosticSimulatorMixin):
    """Base enum variable with conversion module support."""


class _ReadbackFromControlMixin(ReadOnlyActionMixin):
    """Common readback behavior for variables that share control get logic."""

    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        # Skip ReadOnlyActionMixin's abstract _get and delegate to the next class.
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator: Any, value: Any) -> None:
        raise RuntimeError(f"{self.name} is read-only")


# ============================================================================
# Quadrupole Variables
# ============================================================================


class QuadrupoleBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Quadrupoles"""

    read_only: bool = False
    unit: str = "kG-m"
    get_function: Optional[str] = "get_quadrupole_bctrl"
    set_function: Optional[str] = "set_quadrupole_bctrl"


class QuadrupoleBACTVariable(QuadrupoleBCTRLVariable, _ReadbackFromControlMixin):
    """Action that operates on the BACT property of Quadrupoles"""


# ============================================================================
# Solenoid Variables
# ============================================================================


class SolenoidBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Solenoids"""
    unit: str = "kG-m"
    get_function: str = "get_solenoid_bctrl"
    set_function: str = "set_solenoid_bctrl"

class SolenoidBACTVariable(SolenoidBCTRLVariable, _ReadbackFromControlMixin):
    """Action that operates on the BACT property of Solenoids"""


# ============================================================================
# SBend (Dipole) Variables
# ============================================================================


class SBendBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of SBends"""
    unit: str = "GeV/c"
    get_function: str = "get_sbend_bctrl"
    set_function: str = "set_sbend_bctrl"


class SBendBACTVariable(SBendBCTRLVariable, _ReadbackFromControlMixin):
    """Action that operates on the BACT property of SBends"""


# ============================================================================
# Kicker Variables (Horizontal)
# ============================================================================


class HKickerBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Horizontal Kicker magnets"""
    unit: str = "kG"
    get_function: str = "get_hkicker_bctrl"
    set_function: str = "set_hkicker_bctrl"


class HKickerBACTVariable(HKickerBCTRLVariable, _ReadbackFromControlMixin):
    """Action that operates on the BACT property of Horizontal Kicker magnets"""


# ============================================================================
# Kicker Variables (Vertical)
# ============================================================================


class VKickerBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Vertical Kicker magnets"""
    unit: str = "kG"
    get_function: str = "get_vkicker_bctrl"
    set_function: str = "set_vkicker_bctrl"


class VKickerBACTVariable(VKickerBCTRLVariable, _ReadbackFromControlMixin):
    """Action that operates on the BACT property of Vertical Kicker magnets"""


# ============================================================================
# Status Variables
# ============================================================================


class StatusVariable(ScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the status of a device (e.g. STATCTRLSUB.T)"""

    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        return 0  # TODO: add logic for status of device


# ============================================================================
# Limit Variables
# ============================================================================


class BminVariable(ScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMIN/DRVL property of a device"""

    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        return -100  # TODO: add logic for these limits


class BmaxVariable(ScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMAX/DRVH property of a device"""

    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        return 100  # TODO: add logic for these limits


# ============================================================================
# Control State Variables
# ============================================================================


class ControlStateVariable(EnumVariable, ReadOnlyActionMixin):
    """Action that operates on the control state (e.g. CTRL) of a device"""

    read_only: bool = True
    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"

    def _get(self, simulator: Any) -> Any:
        return "Ready"


# ============================================================================
# BPM Variables
# ============================================================================


class BPMXVariable(ScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the X position of a BPM"""

    unit: str = "mm"
    get_function: str = "get_bpm_x"


class BPMYVariable(ScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the Y position of a BPM"""

    unit: str = "mm"
    get_function: str = "get_bpm_y"


class BPMTMITDummyVariable(ScalarVariable, ReadOnlyActionMixin):
    """Dummy variable for BPM TMIT (total beam intensity) for testing purposes."""

    unit: str = "arbitrary units"
    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        # Return a dummy value for TMIT
        return 1.0  # This can be adjusted as needed for testing


# ============================================================================
# Klystron Variables
# ============================================================================


class KlystronENLDVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the amplitude of a klystron which acts on an overlay.
    """

    unit: str = "MeV"
    get_function: str = "get_klystron_enld"
    set_function: str = "set_klystron_enld"


class KlystronPDESVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase of a klystron which acts on an overlay.
    """

    unit: str = "degrees"
    get_function: str = "get_klystron_pdes"
    set_function: str = "set_klystron_pdes"


class KlystronPACTVariable(KlystronPDESVariable, _ReadbackFromControlMixin):
    """
    Action that operates on the actual phase of a klystron which acts on an overlay.
    """


class KlystronStatVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the status of a klystron which acts on an overlay.
    Note: the element_name for this variable should be set for an overlay element.

    Values other than 1.0 (e.g. 0.0, 4.0 for bypassed/standby beamcodes) are
    treated as "off" and set the overlay's IN_USE to 0.
    """

    read_only: bool = False
    default_value: float = 0.0
    description: str = "Klystron on/off status"

    def _get(self, simulator: Any) -> Any:
        return float(simulator.ele(self.element_name).control_vars["IN_USE"])

    def _set(self, simulator: Any, value: Any) -> None:
        in_use = 1 if value == 1.0 else 0
        simulator.cmd(f"set ele {self.element_name} IN_USE = {in_use}")


# ============================================================================
# Cavity Variables
# ============================================================================


class CavityAREQVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the amplitude property of a cavity.
    """

    unit: str = "MV"
    get_function: str = "get_cavity_areq"
    set_function: str = "set_cavity_areq"


class CavityAREQReadbackVariable(CavityAREQVariable, _ReadbackFromControlMixin):
    """Read-only variant of cavity amplitude request variable."""


class CavityPREQVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase property of a cavity.
    """

    unit: str = "degrees"
    get_function: str = "get_cavity_preq"
    set_function: str = "set_cavity_preq"


class CavityPREQReadbackVariable(CavityPREQVariable, _ReadbackFromControlMixin):
    """Read-only variant of cavity phase request variable."""


class DummyEnumVariable(EnumVariable, WritableActionMixin):
    """
    Dummy variable for testing purposes. This variable does not correspond to any real element or property.
    It is used to test the behavior of the model when a variable is requested that does not exist in the model.
    """

    options: list[str] = ["0", "1"]
    default_value: str = "0"

    _value: str = "0"

    def _get(self, simulator: Any) -> Any:
        return self._value

    def _set(self, simulator: Any, value: Any) -> None:
        self._value = value


class CavityMODECFGVariable(EnumVariable, WritableActionMixin):
    """
    Action that operates on the mode configuration property of a cavity.
    """

    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"
    get_function: str = "get_cavity_modecfg"
    set_function: str = "set_cavity_modecfg"
