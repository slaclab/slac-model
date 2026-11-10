"""Generic lume `Variable` subclasses for virtual accelerator elements.

This module provides generic action classes that work with any conversion module
(bmad, cheetah, impact, etc.). Instead of duplicating class definitions for each
backend, this module defines all classes with configurable conversion modules.

Usage:
    # For bmad backend
    from slac_model.virtual_accelerator.actions import (
        QuadrupoleBCTRLVariable,
        QuadrupoleBACTVariable,
    )
    from slac_model.pv_conversion import bmad

    QuadrupoleBCTRLVariable.set_conversion_module(bmad)
    
    # For cheetah backend
    from slac_model.pv_conversion import cheetah
    QuadrupoleBCTRLVariable.set_conversion_module(cheetah)

Alternatively, create backend-specific modules:
    # cheetah_actions.py
    from slac_model.virtual_accelerator.actions import (
        QuadrupoleBCTRLVariable as _QuadrupoleBCTRLVariable,
    )
    from slac_model.pv_conversion import cheetah

    _QuadrupoleBCTRLVariable.set_conversion_module(cheetah)
    QuadrupoleBCTRLVariable = _QuadrupoleBCTRLVariable
"""

from typing import Any, Optional
from abc import ABC

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, EnumVariable


class ConversionModuleMixin(ABC):
    """Mixin that provides a class-level conversion module reference."""

    _conversion_module = None

    @classmethod
    def set_conversion_module(cls, module: Any) -> None:
        """Set the conversion module for this variable class.
        
        Args:
            module: The conversion module (e.g., slac_model.pv_conversion.bmad)
        """
        cls._conversion_module = module

    @classmethod
    def get_conversion_module(cls) -> Any:
        """Get the conversion module for this variable class.
        
        Raises:
            RuntimeError: If no conversion module has been set.
        """
        if cls._conversion_module is None:
            raise RuntimeError(
                f"Conversion module not set for {cls.__name__}. "
                f"Call {cls.__name__}.set_conversion_module() first."
            )
        return cls._conversion_module


class ScalarVariable(ScalarVariable, ConversionModuleMixin):
    """Base scalar variable with conversion module support."""

    element_name: str


class EnumVariable(EnumVariable, ConversionModuleMixin):
    """Base enum variable with conversion module support."""

    element_name: str


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
    unit: str = "kG"

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_quadrupole_bctrl(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_quadrupole_bctrl(simulator, self.element_name, value)


class QuadrupoleBACTVariable(_ReadbackFromControlMixin, QuadrupoleBCTRLVariable):
    """Action that operates on the BACT property of Quadrupoles"""


# ============================================================================
# Solenoid Variables
# ============================================================================


class SolenoidBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Solenoids"""

    conversion_name: str = "solenoid"

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_solenoid_bctrl(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_solenoid_bctrl(simulator, self.element_name, value)


class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
    """Action that operates on the BACT property of Solenoids"""


# ============================================================================
# SBend (Dipole) Variables
# ============================================================================


class SBendBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of SBends"""

    read_only: bool = False
    unit: str = "GeV/c"

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_sbend_bctrl(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_sbend_bctrl(simulator, self.element_name, value)


class SBendBACTVariable(_ReadbackFromControlMixin, SBendBCTRLVariable):
    """Action that operates on the BACT property of SBends"""


# ============================================================================
# Kicker Variables (Horizontal)
# ============================================================================


class HKickerBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Horizontal Kicker magnets"""

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_hkicker_bctrl(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_hkicker_bctrl(simulator, self.element_name, value)


class HKickerBACTVariable(_ReadbackFromControlMixin, HKickerBCTRLVariable):
    """Action that operates on the BACT property of Horizontal Kicker magnets"""


# ============================================================================
# Kicker Variables (Vertical)
# ============================================================================


class VKickerBCTRLVariable(ScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Vertical Kicker magnets"""

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_vkicker_bctrl(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_vkicker_bctrl(simulator, self.element_name, value)


class VKickerBACTVariable(_ReadbackFromControlMixin, VKickerBCTRLVariable):
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
    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_bpm_x(simulator, self.element_name)


class BPMYVariable(ScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the Y position of a BPM"""

    unit: str = "mm"
    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_bpm_y(simulator, self.element_name)


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

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_klystron_enld(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_klystron_enld(simulator, self.element_name, value)


class KlystronPDESVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase of a klystron which acts on an overlay.
    """

    unit: str = "degrees"

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_klystron_pdes(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_klystron_pdes(simulator, self.element_name, value)


class KlystronPACTVariable(_ReadbackFromControlMixin, KlystronPDESVariable):
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

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_cavity_areq(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_cavity_areq(simulator, self.element_name, value)


class CavityAREQReadbackVariable(_ReadbackFromControlMixin, CavityAREQVariable):
    """Read-only variant of cavity amplitude request variable."""


class CavityPREQVariable(ScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase property of a cavity.
    """

    unit: str = "degrees"

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_cavity_preq(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_cavity_preq(simulator, self.element_name, value)


class CavityPREQReadbackVariable(_ReadbackFromControlMixin, CavityPREQVariable):
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

    def _get(self, simulator: Any) -> Any:
        module = self.get_conversion_module()
        return module.get_cavity_modecfg(simulator, self.element_name)

    def _set(self, simulator: Any, value: Any) -> None:
        module = self.get_conversion_module()
        module.set_cavity_modecfg(simulator, self.element_name, value)
