import pytest

torch = pytest.importorskip("torch")
cheetah = pytest.importorskip("cheetah")
lume_cheetah = pytest.importorskip("lume_cheetah")

from cheetah.accelerator import BPM, Drift, Quadrupole, Screen, Segment
from cheetah.particles import ParticleBeam
from lume_cheetah.simulator import CheetahSimulator

from slac_model.virtual_accelerator.factory import build_actions, build_cheetah_model, CheetahModelSpec


@pytest.fixture
def cheetah_simulator():
    segment = Segment(
        elements=[
            Quadrupole(name="Q1", length=torch.tensor(0.1), k1=torch.tensor(1.0)),
            BPM(name="BPM1"),
            Screen(
                name="OTR1",
                resolution=(4, 4),
                pixel_size=torch.tensor((1e-5, 1e-5)),
                is_active=True,
            ),
            Drift(name="D1", length=torch.tensor(0.5)),
        ]
    )
    beam = ParticleBeam.from_parameters(num_particles=100, energy=torch.tensor(1e9))
    simulator = CheetahSimulator(segment=segment, initial_beam_distribution=beam)
    simulator.track()
    return simulator


def test_build_model_scalar_variables(cheetah_simulator):
    variable_config = {
        "Quadrupole": {"BCTRL": "QuadrupoleBCTRLVariable", "BACT": "QuadrupoleBACTVariable"},
        "BPM": {"X": "BPMXVariable", "Y": "BPMYVariable"},
    }

    variables = build_actions(cheetah_simulator, variable_config)

    names = {v.name for v in variables}
    assert names == {"Q1:BCTRL", "Q1:BACT", "BPM1:X", "BPM1:Y"}

    by_name = {v.name: v for v in variables}
    assert by_name["Q1:BCTRL"].read_only is False
    assert by_name["Q1:BACT"].read_only is True
    assert by_name["Q1:BCTRL"]._get(cheetah_simulator) == pytest.approx(
        by_name["Q1:BACT"]._get(cheetah_simulator)
    )


def test_build_model_screen_variables(cheetah_simulator):
    variable_config = {
        "Screen": {
            "Image:ArrayData": "ScreenImageVariable",
            "RESOLUTION": "ScreenResolutionVariable",
            "Image:ArraySize0_RBV": {"variable_class": "ScreenImageArraySizeVariable", "index": 1},
            "Image:ArraySize1_RBV": {"variable_class": "ScreenImageArraySizeVariable", "index": 0},
        },
    }
    screen_config = {"OTR1": {"shape": (4, 4), "pixel_size": 1e-5}}

    variables = build_actions(cheetah_simulator, variable_config, screen_config)
    by_name = {v.name: v for v in variables}

    assert by_name["OTR1:Image:ArrayData"]._get(cheetah_simulator).shape == (4, 4)
    assert by_name["OTR1:Image:ArraySize0_RBV"]._get(cheetah_simulator) == 4
    assert by_name["OTR1:Image:ArraySize1_RBV"]._get(cheetah_simulator) == 4


def test_build_model_unknown_class_raises(cheetah_simulator):
    with pytest.raises(ValueError, match="Unknown variable class"):
        build_actions(cheetah_simulator, {"Quadrupole": {"BCTRL": "NotARealClass"}})


def test_build_model_unsupported_simulator_raises():
    with pytest.raises(TypeError, match="Unsupported simulator type"):
        build_actions(object(), {})


def test_build_cheetah_model(tmp_path, monkeypatch):
    segment = Segment(
        elements=[
            Quadrupole(name="Q1", length=torch.tensor(0.1), k1=torch.tensor(1.0)),
            BPM(name="BPM1"),
        ]
    )
    lattice_path = tmp_path / "lattice.json"
    segment.to_lattice_json(str(lattice_path))
    monkeypatch.setenv("TEST_LATTICE_ROOT", str(tmp_path))

    spec = CheetahModelSpec(
        lattice_env_var="TEST_LATTICE_ROOT",
        lattice_relpath="lattice.json",
        energy=1e9,
        variable_config={
            "Quadrupole": {"BCTRL": "QuadrupoleBCTRLVariable"},
            "BPM": {"X": "BPMXVariable"},
        },
    )

    model = build_cheetah_model(spec)

    names = set(model.supported_variables)
    assert names == {"Q1:BCTRL", "BPM1:X"}
    # defaults to a single particle at the origin when no initial_beam_relpath is given
    assert model.simulator.beam_distribution.num_particles == 1
