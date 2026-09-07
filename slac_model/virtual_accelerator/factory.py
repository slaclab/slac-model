from typing import Any, Optional, Union

from lume.variables import Variable

from ..utils import _isinstance_if_importable, get_all_element_types
from .actions import create_classes


def _resolve_backend(simulator: Any):
    """Return the `pv_conversion` accessor module matching `simulator`'s type."""
    # Import backend submodules lazily, only once we know which one is needed: each
    # eagerly imports its simulator package, and those optional deps may not all be installed.
    if _isinstance_if_importable(simulator, "pytao", "Tao"):
        from ..pv_conversion import bmad

        return bmad
    if _isinstance_if_importable(simulator, "impact", "Impact"):
        from ..pv_conversion import impact as impact_backend

        return impact_backend
    if _isinstance_if_importable(simulator, "lume_cheetah.simulator", "CheetahSimulator"):
        from ..pv_conversion import cheetah

        return cheetah

    raised_type = type(simulator).__name__
    raised_module = type(simulator).__module__
    raise TypeError(
        f"Unsupported simulator type {raised_module}.{raised_type}; expected "
        "a Bmad Tao, Impact simulator, or Cheetah CheetahSimulator."
    )


def build_model(
    simulator: Any,
    variable_config: dict[str, dict[str, Union[str, dict[str, Any]]]],
    screen_config: Optional[dict[str, dict[str, Any]]] = None,
) -> list[Variable]:
    """
    Build the list of action variables for `simulator`.

    Parameters:
    -----------
    simulator : Tao | Impact | Segment
        The live simulator instance to bind the constructed variables to.
    variable_config : dict[str, dict[str, str | dict]]
        Mapping of element type -> attribute suffix -> variable class name
        (e.g. loaded from `slac_variable_config.yaml`), or a dict
        `{"variable_class": name, **extra_kwargs}` for classes that need
        extra per-attribute constructor args (e.g. a Screen array-size
        variable's `index`).
    screen_config : dict[str, dict[str, Any]], optional
        Mapping of screen element name -> extra constructor args shared by
        all of that screen's variables (e.g. `shape`, `pixel_size`).

    Returns:
    --------
    list[Variable]
        The instantiated action variables, bound to the backend module
        matching `simulator`'s type.
    """
    element_types = get_all_element_types(simulator)
    backend = _resolve_backend(simulator)
    variable_classes = create_classes(backend)
    screen_config = screen_config or {}

    variables = []
    for element_name, element_type in element_types.items():
        if element_type not in variable_config:
            continue

        for attr, class_spec in variable_config[element_type].items():
            if isinstance(class_spec, dict):
                class_name = class_spec["variable_class"]
                extra_kwargs = {k: v for k, v in class_spec.items() if k != "variable_class"}
            else:
                class_name = class_spec
                extra_kwargs = {}

            var_class = variable_classes.get(class_name)
            if var_class is None:
                raise ValueError(
                    f"Unknown variable class {class_name!r} for {element_name}.{attr}"
                )

            # Fill any remaining required fields (e.g. screen shape/pixel_size) from screen_config.
            element_spec = screen_config.get(element_name, {})
            for field_name, value in element_spec.items():
                if field_name in var_class.model_fields and field_name not in extra_kwargs:
                    extra_kwargs[field_name] = value

            variables.append(
                var_class(
                    name=f"{element_name}:{attr}",
                    element_name=element_name,
                    **extra_kwargs,
                )
            )

    return variables
