"""Shared base classes, mixins, and device definitions for action variables.

This module contains:

1. AgnosticSimulatorMixin - Handles instance-level conversion module binding
2. _ReadbackFromControlMixin - Common readback behavior for readback variables
3. Device-specific base classes (mixins) - Define get/set functions, units, and other
   attributes for each device type. These are meant to be combined with a backend-
   specific ScalarVariable base class (from lume or lume_torch).

Usage pattern:
    # In a backend-specific module (actions.py or torch_actions.py):
    
    class ScalarVariable(LumeScalarVariable, AgnosticSimulatorMixin):
        pass
    
    class QuadrupoleBCTRLVariable(ScalarVariable, _QuadrupoleBCTRLVariableMixin):
        pass
"""

from typing import Any
from abc import ABC

from lume.actions import ReadOnlyActionMixin, WritableActionMixin


class AgnosticSimulatorMixin(ABC):
    """Mixin that provides instance-level conversion module binding.
    
    This mixin allows variable instances to have their conversion module bound
    at instantiation time rather than at class definition time. This enables
    the same variable classes to work with different backends by just changing
    which conversion module is bound to each instance.
    """
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


class _ReadbackFromControlMixin(ReadOnlyActionMixin):
    """Common readback behavior for variables that share control get logic.
    
    This mixin is used for variables that represent the actual/readback value
    of a control property (e.g., BACT vs BCTRL). It marks the variable as
    read-only and provides the standard readback behavior.
    """

    read_only: bool = True

    def _get(self, simulator: Any) -> Any:
        # Skip ReadOnlyActionMixin's abstract _get and delegate to the next class.
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator: Any, value: Any) -> None:
        raise RuntimeError(f"{self.name} is read-only")


# ============================================================================
# Device-Specific Base Classes (Mixins)
# ============================================================================
# Each device class below defines the attributes and behavior for a specific
# device type. They are intended to be combined with a backend-specific
# ScalarVariable or EnumVariable base class using multiple inheritance.
#
# Example:
#     class QuadrupoleBCTRLVariable(ScalarVariable, _QuadrupoleBCTRLVariableMixin):
#         pass


class _QuadrupoleBCTRLVariableMixin(WritableActionMixin):
    """Mixin for quadrupole strength control variables."""
    unit: str = "kG-m"
    get_function: str = "get_quadrupole_bctrl"
    set_function: str = "set_quadrupole_bctrl"


class _QuadrupoleBACTVariableMixin(_ReadbackFromControlMixin):
    """Mixin for quadrupole strength readback variables."""
    pass


class _SolenoidBCTRLVariableMixin(WritableActionMixin):
    """Mixin for solenoid strength control variables."""
    unit: str = "T"
    get_function: str = "get_solenoid_bctrl"
    set_function: str = "set_solenoid_bctrl"


class _SolenoidBACTVariableMixin(_ReadbackFromControlMixin):
    """Mixin for solenoid strength readback variables."""
    pass


class _CavityAMPLVariableMixin(WritableActionMixin):
    """Mixin for cavity amplitude control variables."""
    unit: str = "MV"
    get_function: str = "get_cavity_ampl"
    set_function: str = "set_cavity_ampl"


class _CavityPHASVariableMixin(WritableActionMixin):
    """Mixin for cavity phase control variables."""
    unit: str = "degrees"
    get_function: str = "get_cavity_phas"
    set_function: str = "set_cavity_phas"


class _HKickerBCTRLVariableMixin(WritableActionMixin):
    """Mixin for horizontal kicker strength control variables."""
    unit: str = "T"
    get_function: str = "get_hkicker_bctrl"
    set_function: str = "set_hkicker_bctrl"


class _HKickerBACTVariableMixin(_ReadbackFromControlMixin):
    """Mixin for horizontal kicker strength readback variables."""
    pass


class _VKickerBCTRLVariableMixin(WritableActionMixin):
    """Mixin for vertical kicker strength control variables."""
    unit: str = "T"
    get_function: str = "get_vkicker_bctrl"
    set_function: str = "set_vkicker_bctrl"


class _VKickerBACTVariableMixin(_ReadbackFromControlMixin):
    """Mixin for vertical kicker strength readback variables."""
    pass


class _StatusVariableMixin(_ReadbackFromControlMixin):
    """Mixin for device status variables."""
    get_function: str = "get_status"


class _BminVariableMixin(_ReadbackFromControlMixin):
    """Mixin for minimum field strength variables."""
    unit: str = ""
    get_function: str = "get_bmin"


class _BmaxVariableMixin(_ReadbackFromControlMixin):
    """Mixin for maximum field strength variables."""
    unit: str = ""
    get_function: str = "get_bmax"


class _ControlStateVariableMixin(_ReadbackFromControlMixin):
    """Mixin for control state enum variables."""
    options: list[str] = ["Local", "Remote", "Off"]
    default_value: str = "Off"
    get_function: str = "get_control_state"


class _BPMXVariableMixin(_ReadbackFromControlMixin):
    """Mixin for BPM X position variables."""
    unit: str = "m"
    get_function: str = "get_bpm_x"


class _BPMYVariableMixin(_ReadbackFromControlMixin):
    """Mixin for BPM Y position variables."""
    unit: str = "m"
    get_function: str = "get_bpm_y"


class _BPMTMITDummyVariableMixin(_ReadbackFromControlMixin):
    """Mixin for BPM TMIT (total intensity) variables."""
    unit: str = ""
    get_function: str = "get_bpm_tmit"


class _KlystronENLDVariableMixin(WritableActionMixin):
    """Mixin for klystron ENLD (energy load) control variables."""
    unit: str = "MV"
    get_function: str = "get_klystron_enld"
    set_function: str = "set_klystron_enld"


class _KlystronPDESVariableMixin(WritableActionMixin):
    """Mixin for klystron PDES (phase desired) control variables."""
    unit: str = "degrees"
    get_function: str = "get_klystron_pdes"
    set_function: str = "set_klystron_pdes"


class _KlystronPACTVariableMixin(_ReadbackFromControlMixin):
    """Mixin for klystron PACT (phase actual) readback variables."""
    pass


class _KlystronStatVariableMixin(WritableActionMixin):
    """Mixin for klystron STAT (status) variables."""
    unit: str = "bit"
    get_function: str = "get_klystron_stat"
    set_function: str = "set_klystron_stat"


class _CavityAREQVariableMixin(WritableActionMixin):
    """Mixin for cavity AREQ (amplitude request) control variables."""
    unit: str = "MV"
    get_function: str = "get_cavity_areq"
    set_function: str = "set_cavity_areq"


class _CavityAREQReadbackVariableMixin(_ReadbackFromControlMixin):
    """Mixin for cavity AREQ (amplitude request) readback variables."""
    pass


class _CavityPREQVariableMixin(WritableActionMixin):
    """Mixin for cavity PREQ (phase request) control variables."""
    unit: str = "degrees"
    get_function: str = "get_cavity_preq"
    set_function: str = "set_cavity_preq"


class _CavityPREQReadbackVariableMixin(_ReadbackFromControlMixin):
    """Mixin for cavity PREQ (phase request) readback variables."""
    pass


class _DummyEnumVariableMixin(WritableActionMixin):
    """Mixin for dummy test enum variables."""
    options: list[str] = ["0", "1"]
    default_value: str = "0"
    _value: str = "0"

    def _get(self, simulator: Any) -> Any:
        return self._value

    def _set(self, simulator: Any, value: Any) -> None:
        self._value = value


class _CavityMODECFGVariableMixin(WritableActionMixin):
    """Mixin for cavity mode configuration enum variables."""
    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"
    get_function: str = "get_cavity_modecfg"
    set_function: str = "set_cavity_modecfg"

