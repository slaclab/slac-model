"""Tests for slac_model.pv_conversion.impact accessor functions."""

import pytest

impact = pytest.importorskip("impact")

from impact import Impact

import slac_model.pv_conversion.impact as impact_cnv


def _build_lattice():
    return [
        {"type": "rotationally_symmetric_to_3d", "name": "2d_to_3d_spacecharge", "s": -1000.0},
        {
            "type": "quadrupole",
            "name": "Q1",
            "zedge": 0.0,
            "L": 0.5,
            "s": 0.5,
            "b1_gradient": 2.0,
            "L_effective": 0.5,
            "radius": 10.0,
        },
        {"type": "write_beam", "name": "OTR1", "s": 1.0, "filename": "fort.71", "sample_frequency": 1},
        {"type": "stop", "name": "stop_1", "s": 1.0},
    ]


@pytest.fixture
def simulator():
    """Lattice-only Impact object (not run) for tests that don't need tracked particles."""
    I = Impact()
    I.header["Np"] = 500
    I.header["Bcurr"] = 0
    I.header["Xrad"] = 0.5
    I.header["Yrad"] = 0.5
    I.input["lattice"] = _build_lattice()
    I.configure()
    return I


@pytest.fixture
def tracked_simulator(simulator):
    """Same lattice, actually run through ImpactTexe so `.particles` is populated."""
    simulator.run()
    return simulator


def test_validate_element_raises_for_wrong_type(simulator):
    with pytest.raises(ValueError):
        impact_cnv.get_quadrupole_bctrl(simulator, "OTR1")


class TestQuadrupole:
    def test_get_bctrl(self, simulator):
        b1_gradient = simulator.ele["Q1"]["b1_gradient"]
        l_effective = simulator.ele["Q1"]["L_effective"]
        expected = -b1_gradient * l_effective * 10
        assert impact_cnv.get_quadrupole_bctrl(simulator, "Q1") == pytest.approx(expected)

    def test_set_get_roundtrip(self, simulator):
        impact_cnv.set_quadrupole_bctrl(simulator, "Q1", 3.5)
        assert impact_cnv.get_quadrupole_bctrl(simulator, "Q1") == pytest.approx(3.5)

    def test_bact_is_bctrl_alias(self):
        assert impact_cnv.get_quadrupole_bact is impact_cnv.get_quadrupole_bctrl


class TestScreenImage:
    def test_get_screen_image(self, tracked_simulator):
        image = impact_cnv.get_screen_image(tracked_simulator, "OTR1", (64, 64), 2e-4)
        assert image.shape == (64, 64)
        assert image.sum() > 0

    def test_get_screen_image_wrong_type_raises(self, tracked_simulator):
        with pytest.raises(ValueError):
            impact_cnv.get_screen_image(tracked_simulator, "Q1", (64, 64), 2e-4)

    def test_get_screen_x_y(self, tracked_simulator):
        beam = tracked_simulator.particles["OTR1"]
        assert impact_cnv.get_screen_x(tracked_simulator, "OTR1") == pytest.approx(beam.x.mean() * 1e3)
        assert impact_cnv.get_screen_y(tracked_simulator, "OTR1") == pytest.approx(beam.y.mean() * 1e3)

    def test_get_screen_image_array_size(self, simulator):
        assert impact_cnv.get_screen_image_array_size(simulator, "OTR1", (80, 60), 0) == 80
        assert impact_cnv.get_screen_image_array_size(simulator, "OTR1", (80, 60), 1) == 60

    def test_get_screen_resolution(self, simulator):
        assert impact_cnv.get_screen_resolution(simulator, "OTR1", 2e-4) == pytest.approx(200.0)
