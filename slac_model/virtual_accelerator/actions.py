import types
from types import ModuleType
from typing import Any, Callable, Optional, Union

from lume.variables import EnumVariable, NDVariable, ScalarVariable
from lume.actions import WritableActionMixin, ReadOnlyActionMixin


def create_classes(module: ModuleType) -> dict:
    """Build `Variable` subclasses whose accessor functions are looked up on `module`."""

    def resolve_func(func: Union[str, Callable]) -> Callable:
        """Look up `func` on `module` by name, or use it directly if already callable."""
        return getattr(module, func) if isinstance(func, str) else func

    def make_variable_class(
        class_name: str,
        variable_type: type,
        get_func: Union[str, Callable],
        set_func: Optional[Union[str, Callable]] = None,
        required_fields: Optional[dict[str, type]] = None,
        **field_defaults: Any,
    ):
        mixin = WritableActionMixin if set_func else ReadOnlyActionMixin
        required_fields = required_fields or {}

        def _get(self, simulator):
            extra_args = [getattr(self, field_name) for field_name in required_fields]
            return resolve_func(get_func)(simulator, self.element_name, *extra_args)

        def exec_body(ns: dict):
            defaults = {"read_only": set_func is None, **field_defaults}
            annotations = {"element_name": str, **required_fields, **{k: type(v) for k, v in defaults.items()}}
            ns["__annotations__"] = annotations
            ns.update(defaults)
            ns["_get"] = _get
            if set_func:
                def _set(self, simulator, value):
                    extra_args = [getattr(self, field_name) for field_name in required_fields]
                    resolve_func(set_func)(simulator, self.element_name, *extra_args, value)

                ns["_set"] = _set

        # `types.new_class` builds the class via the real `class` statement machinery,
        # so it correctly picks up `variable_type`'s (e.g. pydantic) metaclass.
        _Variable = types.new_class(class_name, (variable_type, mixin), exec_body=exec_body)
        return _Variable

    # (class_name, variable_type, get_func, set_func, required_fields, field_defaults) —
    # get_func/set_func can be a name to look up on `module`, or a callable (e.g. lambda)
    # taking the same (simulator, element_name, *required_fields[, value]) signature.
    # `required_fields` declares extra per-instance constructor args (e.g. screen shape)
    # that are passed positionally to get_func/set_func after element_name.
    specs = [
        ("QuadrupoleBCTRLVariable", ScalarVariable, "get_quadrupole_bctrl", "set_quadrupole_bctrl", None, {"unit": "kG"}),
        ("QuadrupoleBACTVariable", ScalarVariable, "get_quadrupole_bact", None, None, {"unit": "kG"}),
        ("SolenoidBCTRLVariable", ScalarVariable, "get_solenoid_bctrl", "set_solenoid_bctrl", None, {"unit": "kG"}),
        ("SolenoidBACTVariable", ScalarVariable, "get_solenoid_bact", None, None, {"unit": "kG"}),
        ("SBendBCTRLVariable", ScalarVariable, "get_sbend_bctrl", "set_sbend_bctrl", None, {"unit": "GeV/c"}),
        ("SBendBACTVariable", ScalarVariable, "get_sbend_bact", None, None, {"unit": "GeV/c"}),
        ("KickerBCTRLVariable", ScalarVariable, "get_kicker_bctrl", "set_kicker_bctrl", None, {"unit": "kG"}),
        ("KickerBACTVariable", ScalarVariable, "get_kicker_bact", None, None, {"unit": "kG"}),
        ("BmaxVariable", ScalarVariable, lambda s,e: 100.0, None, None, {"unit": "kG"}),
        ("BminVariable", ScalarVariable, lambda s,e: -100.0, None, None, {"unit": "kG"}),
        ("ControlStateVariable", EnumVariable, lambda s, e: "Ready", None, None, {
            "read_only": True,
            "options": ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"],
            "default_value": "Ready",
        }),
        ("StatusVariable", EnumVariable, lambda s, e: 0, None, None, {"options": [0, 1]}),
        ("BPMXVariable", ScalarVariable, "get_bpm_x", None, None, {"unit": "mm"}),
        ("BPMYVariable", ScalarVariable, "get_bpm_y", None, None, {"unit": "mm"}),
        ("KlystronENLDVariable", ScalarVariable, "get_klystron_enld", "set_klystron_enld", None, {"unit": "MeV"}),
        ("KlystronPDESVariable", ScalarVariable, "get_klystron_pdes", "set_klystron_pdes", None, {"unit": "degrees"}),
        ("KlystronPACTVariable", ScalarVariable, "get_klystron_pact", None, None, {"unit": "degrees"}),
        ("KlystronStatVariable", EnumVariable, "get_klystron_stat", "set_klystron_stat", None, {"options": [0, 1]}),
        ("CavityAREQVariable", ScalarVariable, "get_cavity_areq", "set_cavity_areq", None, {"unit": "MV"}),
        ("CavityAREQReadbackVariable", ScalarVariable, "get_cavity_areq_readback", None, None, {"unit": "MV"}),
        ("CavityPREQVariable", ScalarVariable, "get_cavity_preq", "set_cavity_preq", None, {"unit": "degrees"}),
        ("CavityPREQReadbackVariable", ScalarVariable, "get_cavity_preq_readback", None, None, {"unit": "degrees"}),
        # Read-only across all backends: only Bmad's backend can actually toggle is_on.
        ("CavityMODECFGVariable", EnumVariable, "get_cavity_modecfg", None, None, {
            "options": ["Disable", "ACCEL", "STDBY", "ACCEL_STDBY"],
            "default_value": "ACCEL_STDBY",
        }),
        ("ScreenImageVariable", NDVariable, "get_screen_image", None, {"shape": tuple, "pixel_size": float}, {}),
        ("ScreenResolutionVariable", ScalarVariable, "get_screen_resolution", None, {"pixel_size": float}, {}),
        ("ScreenImageArraySizeVariable", ScalarVariable, "get_screen_image_array_size", None, {"shape": tuple, "index": int}, {}),
        ("ScreenXVariable", ScalarVariable, "get_screen_x", None, None, {"unit": "mm"}),
        ("ScreenYVariable", ScalarVariable, "get_screen_y", None, None, {"unit": "mm"}),
    ]

    return {
        class_name: make_variable_class(
            class_name, variable_type, get_func, set_func, required_fields, **field_defaults
        )
        for class_name, variable_type, get_func, set_func, required_fields, field_defaults in specs
    }