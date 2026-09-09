"""Tests for slac_model.pv_conversion.cheetah accessor functions."""

import math

import pytest

cheetah = pytest.importorskip("cheetah")
pytest.importorskip("lume_cheetah")

import torch
from cheetah.accelerator import (
    BPM,
    Drift,
    HorizontalCorrector,
    Quadrupole,
    Screen,
    Segment,
    Solenoid,
    TransverseDeflectingCavity,
)
from cheetah.particles import ParticleBeam
from lume_cheetah.simulator import CheetahSimulator

import slac_model.pv_conversion.cheetah as cheetah_cnv

ENERGY = 1e8


@pytest.fixture
def simulator():
    segment = Segment(
        [
            Quadrupole(name="Q1", length=torch.tensor(0.5), k1=torch.tensor(1.0)),
            Solenoid(name="SOL1", length=torch.tensor(0.2), k=torch.tensor(0.5)),
            HorizontalCorrector(
                name="KICK1", length=torch.tensor(0.1), angle=torch.tensor(0.0)
            ),
            Drift(name="D1", length=torch.tensor(1.0)),
            BPM(name="BPM1", is_active=True),
            TransverseDeflectingCavity(
                name="TCAV1",
                length=torch.tensor(0.3),
                voltage=torch.tensor(1e6),
                phase=torch.tensor(0.25),
                frequency=torch.tensor(2.856e9),
            ),
            Screen(
                name="OTR1",
                is_active=True,
                resolution=(64, 64),
                pixel_size=torch.tensor((1e-4, 1e-4)),
            ),
        ]
    )
    beam = ParticleBeam.from_twiss(
        beta_x=torch.tensor(1.0),
        beta_y=torch.tensor(1.0),
        emittance_x=torch.tensor(1e-8),
        emittance_y=torch.tensor(1e-8),
        num_particles=1000,
        energy=torch.tensor(ENERGY),
    )
    return CheetahSimulator(segment=segment, initial_beam_distribution=beam)


def test_get_magnetic_rigidity():
    assert math.isclose(cheetah_cnv.get_magnetic_rigidity(1e9), 33.356, rel_tol=1e-9)
    assert math.isclose(cheetah_cnv.get_magnetic_rigidity(2e9), 66.712, rel_tol=1e-9)


def test_validate_element_raises_for_wrong_type(simulator):
    with pytest.raises(ValueError):
        cheetah_cnv.get_quadrupole_bctrl(simulator, "D1")


class TestQuadrupole:
    def test_get_bctrl(self, simulator):
        expected = 1.0 * 0.5 * cheetah_cnv.get_magnetic_rigidity(ENERGY)
        assert math.isclose(cheetah_cnv.get_quadrupole_bctrl(simulator, "Q1"), expected, rel_tol=1e-6)

    def test_set_get_roundtrip(self, simulator):
        cheetah_cnv.set_quadrupole_bctrl(simulator, "Q1", 0.75)
        assert math.isclose(cheetah_cnv.get_quadrupole_bctrl(simulator, "Q1"), 0.75, abs_tol=1e-8)

    def test_bact_is_bctrl_alias(self):
        assert cheetah_cnv.get_quadrupole_bact is cheetah_cnv.get_quadrupole_bctrl


class TestSolenoid:
    def test_get_bctrl(self, simulator):
        expected = 0.5 * cheetah_cnv.get_magnetic_rigidity(ENERGY)
        assert math.isclose(cheetah_cnv.get_solenoid_bctrl(simulator, "SOL1"), expected, rel_tol=1e-6)

    def test_set_get_roundtrip(self, simulator):
        cheetah_cnv.set_solenoid_bctrl(simulator, "SOL1", 2.0)
        assert math.isclose(cheetah_cnv.get_solenoid_bctrl(simulator, "SOL1"), 2.0, rel_tol=1e-6)

    def test_bact_is_bctrl_alias(self):
        assert cheetah_cnv.get_solenoid_bact is cheetah_cnv.get_solenoid_bctrl

    def test_wrong_type_raises(self, simulator):
        with pytest.raises(ValueError):
            cheetah_cnv.get_solenoid_bctrl(simulator, "Q1")


class TestKicker:
    def test_set_get_roundtrip(self, simulator):
        cheetah_cnv.set_kicker_bctrl(simulator, "KICK1", 0.1)
        assert math.isclose(cheetah_cnv.get_kicker_bctrl(simulator, "KICK1"), 0.1, rel_tol=1e-6)

    def test_bact_is_bctrl_alias(self):
        assert cheetah_cnv.get_kicker_bact is cheetah_cnv.get_kicker_bctrl


class TestBPM:
    def test_get_x_y(self, simulator):
        bpm = simulator.segment.BPM1
        assert cheetah_cnv.get_bpm_x(simulator, "BPM1") == pytest.approx(bpm.reading[0].item() * 1e3)
        assert cheetah_cnv.get_bpm_y(simulator, "BPM1") == pytest.approx(bpm.reading[1].item() * 1e3)

    def test_wrong_type_raises(self, simulator):
        with pytest.raises(ValueError):
            cheetah_cnv.get_bpm_x(simulator, "Q1")


class TestCavity:
    def test_get_areq(self, simulator):
        assert cheetah_cnv.get_cavity_areq(simulator, "TCAV1") == pytest.approx(1.0)

    def test_set_get_areq_roundtrip(self, simulator):
        cheetah_cnv.set_cavity_areq(simulator, "TCAV1", 2.5)
        assert cheetah_cnv.get_cavity_areq(simulator, "TCAV1") == pytest.approx(2.5)

    def test_areq_readback_is_areq_alias(self):
        assert cheetah_cnv.get_cavity_areq_readback is cheetah_cnv.get_cavity_areq

    def test_get_preq(self, simulator):
        assert cheetah_cnv.get_cavity_preq(simulator, "TCAV1") == pytest.approx(0.25 * 360.0)

    def test_set_get_preq_roundtrip(self, simulator):
        cheetah_cnv.set_cavity_preq(simulator, "TCAV1", 90.0)
        assert cheetah_cnv.get_cavity_preq(simulator, "TCAV1") == pytest.approx(90.0)

    def test_preq_readback_is_preq_alias(self):
        assert cheetah_cnv.get_cavity_preq_readback is cheetah_cnv.get_cavity_preq

    def test_wrong_type_raises(self, simulator):
        with pytest.raises(ValueError):
            cheetah_cnv.get_cavity_areq(simulator, "Q1")


class TestScreen:
    def test_get_image(self, simulator):
        screen = simulator.segment.OTR1
        expected = screen.reading.mT * 65535
        assert torch.equal(cheetah_cnv.get_screen_image(simulator, "OTR1"), expected)

    def test_get_image_array_size(self, simulator):
        assert cheetah_cnv.get_screen_image_array_size(simulator, "OTR1", index=0) == 64
        assert cheetah_cnv.get_screen_image_array_size(simulator, "OTR1", index=1) == 64

    def test_get_resolution(self, simulator):
        assert cheetah_cnv.get_screen_resolution(simulator, "OTR1") == pytest.approx(100.0)

    def test_pneumatic_roundtrip(self, simulator):
        assert cheetah_cnv.get_screen_pneumatic(simulator, "OTR1") == 1.0
        cheetah_cnv.set_screen_pneumatic(simulator, "OTR1", 0.0)
        assert cheetah_cnv.get_screen_pneumatic(simulator, "OTR1") == 0.0
        cheetah_cnv.set_screen_pneumatic(simulator, "OTR1", 1.0)
        assert cheetah_cnv.get_screen_pneumatic(simulator, "OTR1") == 1.0

    def test_get_x_y(self, simulator):
        beam = simulator.segment.OTR1.get_read_beam()
        assert cheetah_cnv.get_screen_x(simulator, "OTR1") == pytest.approx(beam.x.mean().item() * 1e3)
        assert cheetah_cnv.get_screen_y(simulator, "OTR1") == pytest.approx(beam.y.mean().item() * 1e3)


def test_get_cavity_modecfg(simulator):
    assert cheetah_cnv.get_cavity_modecfg(simulator, "TCAV1") == "ACCEL_STDBY"
    assert cheetah_cnv.get_cavity_modecfg(simulator, "Q1") == "ACCEL_STDBY"
