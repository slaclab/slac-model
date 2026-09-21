"""lume-bmad `Variable` subclasses for CU virtual accelerator elements.

Each class wires a PV-facing variable (quadrupoles, solenoids, bends,
correctors, BPMs, klystrons, cavities, and related status/limit fields) to
the corresponding getter/setter in `slac_model.pv_conversion.bmad`, using a
Tao instance as the simulator backend.
"""

from typing import Any

from lume.actions import ReadOnlyActionMixin, WritableActionMixin
from lume.variables import ScalarVariable, EnumVariable
from pytao import Tao

from slac_model.pv_conversion import bmad


class BmadScalarVariable(ScalarVariable):
    """all bmad variables should have the bmad element name associated with them"""

    element_name: str


class BmadEnumVariable(EnumVariable):
    """Base class for Bmad variables that have a discrete set of options."""

    element_name: str


class _ReadbackFromControlMixin(ReadOnlyActionMixin):
    """Common readback behavior for variables that share control get logic."""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        # Skip ReadOnlyActionMixin's abstract _get and delegate to the next class.
        return super(ReadOnlyActionMixin, self)._get(simulator)

    def _set(self, simulator: Tao, value: Any) -> None:
        raise RuntimeError(f"{self.name} is read-only")


class QuadrupoleBCTRLVariable(BmadScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Quadrupoles"""

    read_only: bool = False
    unit: str = "kG"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_quadrupole_bctrl(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_quadrupole_bctrl(simulator, self.element_name, value)


class QuadrupoleBACTVariable(_ReadbackFromControlMixin, QuadrupoleBCTRLVariable):
    """Action that operates on the BACT property of Quadrupoles"""


class SolenoidBCTRLVariable(BmadScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Solenoids"""

    conversion_name: str = "solenoid"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_solenoid_bctrl(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_solenoid_bctrl(simulator, self.element_name, value)


class SolenoidBACTVariable(_ReadbackFromControlMixin, SolenoidBCTRLVariable):
    """Action that operates on the BACT property of Solenoids"""


class SBendBCTRLVariable(BmadScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of SBends"""

    read_only: bool = False
    unit: str = "GeV/c"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_sbend_bctrl(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_sbend_bctrl(simulator, self.element_name, value)


class SBendBACTVariable(_ReadbackFromControlMixin, SBendBCTRLVariable):
    """Action that operates on the BACT property of SBends"""


class HKickerBCTRLVariable(BmadScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Horizontal Kicker magnets"""

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_hkicker_bctrl(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_hkicker_bctrl(simulator, self.element_name, value)


class HKickerBACTVariable(_ReadbackFromControlMixin, HKickerBCTRLVariable):
    """Action that operates on the BACT property of Horizontal Kicker magnets"""


class VKickerBCTRLVariable(BmadScalarVariable, WritableActionMixin):
    """Action that operates on the BCTRL/BDES property of Vertical Kicker magnets"""

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_vkicker_bctrl(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_vkicker_bctrl(simulator, self.element_name, value)


class VKickerBACTVariable(_ReadbackFromControlMixin, VKickerBCTRLVariable):
    """Action that operates on the BACT property of Vertical Kicker magnets"""


class StatusVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the status of a device (e.g. STATCTRLSUB.T)"""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return 0  # TODO: add logic for status of device


class BminVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMIN/DRVL property of a device"""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return -100  # TODO: add logic for these limits


class BmaxVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the BMAX/DRVH property of a device"""

    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return 100  # TODO: add logic for these limits


class ControlStateVariable(BmadEnumVariable, ReadOnlyActionMixin):
    """Action that operates on the control state (e.g. CTRL) of a device"""

    read_only: bool = True
    options: list[str] = ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"]
    default_value: str = "Ready"

    def _get(self, simulator: Tao) -> Any:
        return "Ready"


class BPMXVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the X position of a BPM"""

    unit: str = "mm"
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_bpm_x(simulator, self.element_name)


class BPMYVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Action that operates on the Y position of a BPM"""

    unit: str = "mm"
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_bpm_y(simulator, self.element_name)


class BPMTMITDummyVariable(BmadScalarVariable, ReadOnlyActionMixin):
    """Dummy variable for BPM TMIT (total beam intensity) for testing purposes."""

    unit: str = "arbitrary units"
    read_only: bool = True

    def _get(self, simulator: Tao) -> Any:
        # Return a dummy value for TMIT
        return 1.0  # This can be adjusted as needed for testing


class KlystronENLDVariable(BmadScalarVariable, WritableActionMixin):
    """
    Action that operates on the amplitude of a klystron which acts on an overlay in Bmad
    """

    unit: str = "MeV"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_klystron_enld(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_klystron_enld(simulator, self.element_name, value)


class KlystronPDESVariable(BmadScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase of a klystron which acts on an overlay in Bmad
    """

    unit: str = "degrees"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_klystron_pdes(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_klystron_pdes(simulator, self.element_name, value)


class KlystronPACTVariable(_ReadbackFromControlMixin, KlystronPDESVariable):
    """
    Action that operates on the actual phase of a klystron which acts on an overlay in Bmad

    """


class KlystronStatVariable(BmadEnumVariable, WritableActionMixin):
    """
    Action that operates on the status of a klystron which acts on an overlay in Bmad

    """

    read_only: bool = True
    options: list[str] = ["0", "1"]
    default_value: str = "0"

    def _get(self, simulator: Tao) -> Any:
        return str(bmad.get_klystron_stat(simulator, self.element_name))

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_klystron_stat(simulator, self.element_name, int(value))


class CavityAREQVariable(BmadScalarVariable, WritableActionMixin):
    """
    Action that operates on the amplitude property of a cavity

    """

    unit: str = "MV"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_cavity_areq(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_cavity_areq(simulator, self.element_name, value)


class CavityAREQReadbackVariable(_ReadbackFromControlMixin, CavityAREQVariable):
    """Read-only variant of cavity amplitude request variable."""


class CavityPREQVariable(BmadScalarVariable, WritableActionMixin):
    """
    Action that operates on the phase property of a cavity

    """

    unit: str = "degrees"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_cavity_preq(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_cavity_preq(simulator, self.element_name, value)


class CavityPREQReadbackVariable(_ReadbackFromControlMixin, CavityPREQVariable):
    """Read-only variant of cavity phase request variable."""


class DummyEnumVariable(BmadEnumVariable, WritableActionMixin):
    """
    Dummy variable for testing purposes. This variable does not correspond to any real element or property in the Bmad model.
    It is used to test the behavior of the model when a variable is requested that does not exist in the model.

    """

    options: list[str] = ["0", "1"]
    default_value: str = "0"

    _value: str = "0"

    def _get(self, simulator: Tao) -> Any:
        return self._value

    def _set(self, simulator: Tao, value: Any) -> None:
        self._value = value


class CavityMODECFGVariable(BmadEnumVariable, WritableActionMixin):
    """
    Action that operates on the mode configuration property of a cavity
    """

    options: list[str] = ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"]
    default_value: str = "ACCEL_STDBY"

    def _get(self, simulator: Tao) -> Any:
        return bmad.get_cavity_modecfg(simulator, self.element_name)

    def _set(self, simulator: Tao, value: Any) -> None:
        bmad.set_cavity_modecfg(simulator, self.element_name, value)
