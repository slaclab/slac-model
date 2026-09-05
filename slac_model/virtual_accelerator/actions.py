import types
from types import ModuleType
from typing import Any, Callable, Optional, Union

from lume_base.variables import EnumVariable, ScalarVariable
from lume_base.actions import WritableVariableMixIn, ReadOnlyVariableMixIn


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
        **field_defaults: Any,
    ):
        mixin = WritableVariableMixIn if set_func else ReadOnlyVariableMixIn

        def _get(self, simulator):
            return resolve_func(get_func)(simulator, self.element_name)

        def exec_body(ns: dict):
            ns["__annotations__"] = {"element_name": str, **{k: type(v) for k, v in field_defaults.items()}}
            ns.update(field_defaults)
            ns["_get"] = _get
            if set_func:
                def _set(self, simulator, value):
                    resolve_func(set_func)(simulator, self.element_name, value)

                ns["_set"] = _set

        # `types.new_class` builds the class via the real `class` statement machinery,
        # so it correctly picks up `variable_type`'s (e.g. pydantic) metaclass.
        _Variable = types.new_class(class_name, (variable_type, mixin), exec_body=exec_body)
        return _Variable

    # (class_name, variable_type, get_func, set_func, field_defaults) — get_func/set_func
    # can be a name to look up on `module`, or a callable (e.g. lambda) taking the same
    # (simulator, element_name[, value]) signature. Add a tuple here for each new class.
    specs = [
        ("QuadrupoleBCTRLVariable", ScalarVariable, "get_quadrupole_bctrl", "set_quadrupole_bctrl", {"unit": "kG"}),
        ("QuadrupoleBACTVariable", ScalarVariable, "get_quadrupole_bctrl", None, {"unit": "kG"}),
        ("BmaxVariable", ScalarVariable, lambda s,e: 100.0, None, {"unit": "kG"}),
        ("BminVariable", ScalarVariable, lambda s,e: -100.0, None, {"unit": "kG"}),
        ("ControlStateVariable", EnumVariable, lambda s, e: "Ready", None, {
            "read_only": True,
            "options": ["Ready", "TRIM", "PERTURB", "BCON_TO_BDES", "BACT_TO_BDES"],
            "default_value": "Ready",
        }),
        ("StatusVariable", EnumVariable, lambda s, e: 0, None, {"options": [0, 1]}),
        ("BPMXVariable", ScalarVariable, "get_bpm_x", None, {"unit": "mm"}),
        ("BPMYVariable", ScalarVariable, "get_bpm_y", None, {"unit": "mm"}),
        ("KlystronENLDVariable", ScalarVariable, "get_klystron_enld", "set_klystron_enld", {"unit": "MeV"}),
        ("KlystronPDESVariable", ScalarVariable, "get_klystron_pdes", "set_klystron_pdes", {"unit": "degrees"}),
        ("KlystronStatVariable", EnumVariable, "get_klystron_stat", "set_klystron_stat", {"options": [0, 1]}),
    ]

    return {
        class_name: make_variable_class(class_name, variable_type, get_func, set_func, **field_defaults)
        for class_name, variable_type, get_func, set_func, field_defaults in specs
    }