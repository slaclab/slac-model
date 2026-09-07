import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from lume.variables import Variable

from ..utils import _isinstance_if_importable, get_all_element_types
from .actions import ImpactGroupVariable, create_classes


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


def build_actions(
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


@lru_cache(maxsize=1)
def _load_default_variable_config() -> dict:
    """Load the element_type -> attr -> variable_class mapping bundled with this package."""
    config_path = Path(__file__).parent / "slac_variable_config.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Bmad
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BmadModelSpec:
    lattice_env_var: str
    tao_init_relpath: str
    variable_config: Optional[dict] = None
    screen_config: Optional[dict] = None
    default_track_start: Optional[str] = None
    default_beam_relpath: Optional[str] = None


def build_bmad_model(
    spec: BmadModelSpec,
    start_element: str,
    end_element: str,
    track_beam: bool = False,
    custom_beam_path: Optional[str] = None,
    custom_tao_commands: Optional[list[str]] = None,
    custom_aliases: Optional[dict[str, str]] = None,
):
    """Build a lattice-specific `LUMEBmadModel`, using `build_actions` for variable construction."""
    from pytao import Tao
    from lume_bmad.model import LUMEBmadModel

    lattice_root = os.environ[spec.lattice_env_var]
    init_file = os.path.join(lattice_root, spec.tao_init_relpath)
    tao = Tao(f"-init {init_file} -noplot -slice_lattice {start_element}:{end_element}")

    tao.cmd(f"set beam track_start = {start_element}")

    if custom_tao_commands is not None:
        for cmd in custom_tao_commands:
            tao.cmd(cmd)

    if custom_aliases is not None:
        for element, alias in custom_aliases.items():
            tao.cmd(f"set ele {element} alias = {alias}")

    variables = build_actions(
        tao, spec.variable_config or _load_default_variable_config(), spec.screen_config
    )

    element_types = get_all_element_types(tao)
    active_screens = tuple(
        element for element, element_type in element_types.items() if element_type == "Screen"
    )

    model = LUMEBmadModel(
        tao=tao,
        action_variables=variables,
        dump_locations=list(active_screens),
    )

    if track_beam:
        beam_path = custom_beam_path
        if (
            beam_path is None
            and spec.default_track_start is not None
            and spec.default_beam_relpath is not None
            and start_element == spec.default_track_start
        ):
            beam_path = os.path.join(lattice_root, spec.default_beam_relpath)

        if beam_path is None:
            warnings.warn(
                "track_beam=True for start_element "
                f"!= {spec.default_track_start} without providing custom_beam_path; "
                "beam tracking was not enabled."
            )
        else:
            model.tao.cmd(f"set beam_init position_file = {beam_path}")
            model.set({"track_type": "beam"})

    return model


# ---------------------------------------------------------------------------
# Impact
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactModelSpec:
    lattice_env_var: str
    distgen_file: str
    n_particles: int
    variable_config: Optional[dict] = None
    screen_config: Optional[dict] = None
    stop_location: Optional[Union[str, float]] = None
    impact_file: Optional[str] = None
    impact_yaml_file: Optional[str] = None
    numprocs: int = 1
    space_charge: bool = False


def _get_impact_and_distgen(spec: ImpactModelSpec):
    """Load the `Impact` and `distgen.Generator` objects referenced by `spec`."""
    from impact import Impact
    from distgen import Generator

    lattice_root = os.environ[spec.lattice_env_var]
    distgen_file = os.path.join(lattice_root, spec.distgen_file)

    impact_file = os.path.join(lattice_root, spec.impact_file) if spec.impact_file else None
    impact_yaml_file = (
        os.path.join(lattice_root, spec.impact_yaml_file) if spec.impact_yaml_file else None
    )

    if impact_file is None and impact_yaml_file is None:
        raise ValueError("Either an impact file or an impact YAML file must be provided")

    impact = (
        Impact.from_yaml(impact_yaml_file) if impact_yaml_file is not None else Impact(impact_file)
    )
    distgen = Generator(distgen_file)

    return impact, distgen


def _set_stop_location(impact, stop_location: Union[str, float]):
    """Truncate `impact`'s lattice at `stop_location` (an element name or z position)."""
    if isinstance(stop_location, str):
        try:
            element = impact.ele[stop_location]
            stop_location_z = element["s"]
        except KeyError:
            raise ValueError(f"Element {stop_location!r} not found in the impact model.")
    else:
        stop_location_z = float(stop_location)

    impact.stop = stop_location_z
    impact.ele = {k: v for k, v in impact.ele.items() if v["s"] <= impact.stop}
    impact.input["lattice"] = [
        elem for elem in impact.lattice if elem.get("s", float("inf")) <= impact.stop
    ]
    return impact


def _get_actions_from_groups(impact, spec: ImpactModelSpec) -> list[ImpactGroupVariable]:
    """Build one `ImpactGroupVariable` per `impact_yaml_file` group whose elements are all present."""
    lattice_root = os.environ[spec.lattice_env_var]
    with open(
        os.path.join(lattice_root, spec.impact_yaml_file), "r", encoding="utf-8"
    ) as f:
        impact_config_dict = yaml.safe_load(f)

    actions = []
    for group_name, group_info in impact_config_dict.get("group", {}).items():
        if not all(ele_name in impact.ele for ele_name in group_info.get("ele_names", [])):
            continue
        actions.append(
            ImpactGroupVariable(
                name=f"group:{group_name}",
                group_name=group_name,
                group_key=group_info["var_name"],
            )
        )
    return actions


def build_impact_model(spec: ImpactModelSpec):
    """Build and return the Impact model, using `build_actions` for element-level variable construction."""
    from impact.model.distgen.distgen_impact_model import LUMEDistgenImpactModel

    impact, distgen = _get_impact_and_distgen(spec)

    impact.header["Np"] = spec.n_particles
    impact.numprocs = spec.numprocs
    impact.header["Bcurr"] = 1 if spec.space_charge else 0

    if spec.stop_location is not None:
        impact = _set_stop_location(impact, spec.stop_location)

    impact.run()

    distgen["n_particle"] = spec.n_particles

    model = LUMEDistgenImpactModel.from_objects(distgen, impact)

    variables = build_actions(
        impact, spec.variable_config or _load_default_variable_config(), spec.screen_config
    )
    for var in variables:
        model.register_impact_action_variable(var)

    if spec.impact_yaml_file is not None:
        for action in _get_actions_from_groups(impact, spec):
            model.register_impact_action_variable(action)

    return model


# ---------------------------------------------------------------------------
# Cheetah
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheetahModelSpec:
    lattice_env_var: str
    lattice_relpath: str
    initial_beam_relpath: Optional[str] = None
    energy: Optional[float] = None
    variable_config: Optional[dict] = None
    screen_config: Optional[dict] = None


def build_cheetah_model(spec: CheetahModelSpec):
    """Build a lattice-specific `LUMECheetahModel`, using `build_actions` for variable construction."""
    import torch
    from cheetah.accelerator import Segment
    from cheetah.particles import ParticleBeam
    from lume_cheetah import LUMECheetahModel
    from lume_cheetah.simulator import CheetahSimulator

    lattice_root = os.environ[spec.lattice_env_var]
    segment = Segment.from_lattice_json(os.path.join(lattice_root, spec.lattice_relpath))

    if spec.initial_beam_relpath is not None:
        from beamphysics import ParticleGroup

        particle_group = ParticleGroup(h5=os.path.join(lattice_root, spec.initial_beam_relpath))
        simulator = CheetahSimulator(segment=segment, initial_particle_group=particle_group)
    else:
        # No beam file provided: default to a single particle at the origin.
        incoming_beam = ParticleBeam(torch.zeros(1, 7), energy=torch.tensor(spec.energy))
        simulator = CheetahSimulator(segment=segment, initial_beam_distribution=incoming_beam)

    simulator.track()

    variables = build_actions(
        simulator, spec.variable_config or _load_default_variable_config(), spec.screen_config
    )

    return LUMECheetahModel(simulator=simulator, action_variables=variables)