"""Tests for slac_model.pv_conversion.bmad accessor functions."""

import os

import pytest

pytao = pytest.importorskip("pytao")

from pytao import Tao

import slac_model.pv_conversion.bmad as bmad_cnv

LATTICE_INIT = os.path.join(
    os.path.dirname(__file__), "..", "resources", "bmad", "test_lattice.init"
)


@pytest.fixture
def simulator():
    return Tao(init_file=LATTICE_INIT, noplot=True)


def test_get_set_element_attribute(simulator):
    assert bmad_cnv.get_element_attribute(simulator, "QF", "K1") == pytest.approx(1.2)
    bmad_cnv.set_element_attribute(simulator, "QF", "K1", 2.0)
    assert bmad_cnv.get_element_attribute(simulator, "QF", "K1") == pytest.approx(2.0)


def test_validate_element_raises_for_wrong_type(simulator):
    with pytest.raises(ValueError):
        bmad_cnv.get_quadrupole_bctrl(simulator, "SOL1")


class TestOverlay:
    def test_get_set_roundtrip(self, simulator):
        bmad_cnv.set_overlay_attribute(simulator, "KLYS_OVL", "ENLD_MEV", 3.0)
        assert bmad_cnv.get_overlay_attribute(simulator, "KLYS_OVL", "ENLD_MEV") == pytest.approx(3.0)

    def test_wrong_type_raises(self, simulator):
        with pytest.raises(ValueError):
            bmad_cnv.get_overlay_attribute(simulator, "QF", "ENLD_MEV")


class TestQuadrupole:
    def test_get_bctrl(self, simulator):
        b1_gradient = bmad_cnv.get_element_attribute(simulator, "QF", "B1_GRADIENT")
        length = bmad_cnv.get_element_attribute(simulator, "QF", "L")
        expected = -b1_gradient * length * 10
        assert bmad_cnv.get_quadrupole_bctrl(simulator, "QF") == pytest.approx(expected)

    def test_set_get_roundtrip(self, simulator):
        bmad_cnv.set_quadrupole_bctrl(simulator, "QF", 5.0)
        assert bmad_cnv.get_quadrupole_bctrl(simulator, "QF") == pytest.approx(5.0)

    def test_bact_is_bctrl_alias(self):
        assert bmad_cnv.get_quadrupole_bact is bmad_cnv.get_quadrupole_bctrl


class TestSolenoid:
    def test_get_bctrl(self, simulator):
        bs_field = bmad_cnv.get_element_attribute(simulator, "SOL1", "BS_FIELD")
        expected = -bs_field * 10
        assert bmad_cnv.get_solenoid_bctrl(simulator, "SOL1") == pytest.approx(expected)

    def test_set_get_roundtrip(self, simulator):
        bmad_cnv.set_solenoid_bctrl(simulator, "SOL1", 1.0)
        assert bmad_cnv.get_solenoid_bctrl(simulator, "SOL1") == pytest.approx(1.0)

    def test_bact_is_bctrl_alias(self):
        assert bmad_cnv.get_solenoid_bact is bmad_cnv.get_solenoid_bctrl

    def test_wrong_type_raises(self, simulator):
        with pytest.raises(ValueError):
            bmad_cnv.get_solenoid_bctrl(simulator, "QF")


class TestKicker:
    def test_get_hkicker_bctrl(self, simulator):
        bl_hkick = bmad_cnv.get_element_attribute(simulator, "KICK1", "BL_HKICK")
        expected = -bl_hkick * 10
        assert bmad_cnv.get_hkicker_bctrl(simulator, "KICK1") == pytest.approx(expected)

    def test_set_get_hkicker_roundtrip(self, simulator):
        bmad_cnv.set_hkicker_bctrl(simulator, "KICK1", 0.05)
        assert bmad_cnv.get_hkicker_bctrl(simulator, "KICK1") == pytest.approx(0.05)

    def test_hkicker_bact_is_bctrl_alias(self):
        assert bmad_cnv.get_hkicker_bact is bmad_cnv.get_hkicker_bctrl

    def test_get_vkicker_bctrl(self, simulator):
        bl_vkick = bmad_cnv.get_element_attribute(simulator, "KICK1", "BL_VKICK")
        expected = -bl_vkick * 10
        assert bmad_cnv.get_vkicker_bctrl(simulator, "KICK1") == pytest.approx(expected)

    def test_set_get_vkicker_roundtrip(self, simulator):
        bmad_cnv.set_vkicker_bctrl(simulator, "KICK1", 0.02)
        assert bmad_cnv.get_vkicker_bctrl(simulator, "KICK1") == pytest.approx(0.02)

    def test_vkicker_bact_is_bctrl_alias(self):
        assert bmad_cnv.get_vkicker_bact is bmad_cnv.get_vkicker_bctrl


class TestSBend:
    def test_get_bctrl(self, simulator):
        g = bmad_cnv.get_element_attribute(simulator, "BND1", "G")
        dg = bmad_cnv.get_element_attribute(simulator, "BND1", "DG")
        p0c = bmad_cnv.get_element_attribute(simulator, "BND1", "P0C")
        expected = p0c * (1 + dg / g) * 1e-9
        assert bmad_cnv.get_sbend_bctrl(simulator, "BND1") == pytest.approx(expected)

    def test_set_get_roundtrip(self, simulator):
        bmad_cnv.set_sbend_bctrl(simulator, "BND1", 0.15)
        assert bmad_cnv.get_sbend_bctrl(simulator, "BND1") == pytest.approx(0.15, rel=1e-4)

    def test_bact_is_bctrl_alias(self):
        assert bmad_cnv.get_sbend_bact is bmad_cnv.get_sbend_bctrl

    def test_wrong_type_raises(self, simulator):
        with pytest.raises(ValueError):
            bmad_cnv.get_sbend_bctrl(simulator, "QF")


class TestBPMScreenPosition:
    def test_get_bpm_x_y(self, simulator):
        orbit = simulator.ele("BPM1").orbit
        assert bmad_cnv.get_bpm_x(simulator, "BPM1") == pytest.approx(orbit.x * 1e3)
        assert bmad_cnv.get_bpm_y(simulator, "BPM1") == pytest.approx(orbit.y * 1e3)

    def test_get_screen_x_y(self, simulator):
        orbit = simulator.ele("OTR1").orbit
        assert bmad_cnv.get_screen_x(simulator, "OTR1") == pytest.approx(orbit.x * 1e3)
        assert bmad_cnv.get_screen_y(simulator, "OTR1") == pytest.approx(orbit.y * 1e3)

    def test_screen_is_bpm_alias(self):
        assert bmad_cnv.get_screen_x is bmad_cnv.get_bpm_x
        assert bmad_cnv.get_screen_y is bmad_cnv.get_bpm_y

    def test_invalid_coordinate_raises(self, simulator):
        with pytest.raises(ValueError):
            bmad_cnv.get_bpm_x.func(simulator, "BPM1", coordinate="z")


class TestCavity:
    def test_get_areq(self, simulator):
        voltage = bmad_cnv.get_element_attribute(simulator, "CAV1", "VOLTAGE")
        assert bmad_cnv.get_cavity_areq(simulator, "CAV1") == pytest.approx(voltage * 1e-6)

    def test_set_areq_raises_since_overlay_controlled(self, simulator):
        # CAV1's voltage is driven by the KLYS_OVL overlay, as in the real lattice.
        with pytest.raises(pytao.TaoCommandError):
            bmad_cnv.set_cavity_areq(simulator, "CAV1", 3.0)

    def test_areq_readback_is_areq_alias(self):
        assert bmad_cnv.get_cavity_areq_readback is bmad_cnv.get_cavity_areq

    def test_get_preq(self, simulator):
        phi0 = bmad_cnv.get_element_attribute(simulator, "CAV1", "PHI0")
        assert bmad_cnv.get_cavity_preq(simulator, "CAV1") == pytest.approx(phi0 * 360.0)

    def test_set_preq_raises_since_overlay_controlled(self, simulator):
        # CAV1's phi0 is driven by the KLYS_OVL overlay, as in the real lattice.
        with pytest.raises(pytao.TaoCommandError):
            bmad_cnv.set_cavity_preq(simulator, "CAV1", 90.0)

    def test_preq_readback_is_preq_alias(self):
        assert bmad_cnv.get_cavity_preq_readback is bmad_cnv.get_cavity_preq

    def test_modecfg_roundtrip(self, simulator):
        assert bmad_cnv.get_cavity_modecfg(simulator, "CAV1") == "ACCEL_STDBY"
        bmad_cnv.set_cavity_modecfg(simulator, "CAV1", "STDBY")
        assert bmad_cnv.get_cavity_modecfg(simulator, "CAV1") == "STDBY"
        bmad_cnv.set_cavity_modecfg(simulator, "CAV1", "ACCEL_STDBY")
        assert bmad_cnv.get_cavity_modecfg(simulator, "CAV1") == "ACCEL_STDBY"

    def test_modecfg_invalid_value_raises(self, simulator):
        with pytest.raises(ValueError):
            bmad_cnv.set_cavity_modecfg(simulator, "CAV1", "BOGUS")


class TestKlystron:
    def test_enld_roundtrip(self, simulator):
        bmad_cnv.set_klystron_enld(simulator, "KLYS_OVL", 12.0)
        assert bmad_cnv.get_klystron_enld(simulator, "KLYS_OVL") == pytest.approx(12.0)

    def test_enld_drives_cavity_areq(self, simulator):
        # mirrors the real lattice, where the klystron overlay sets the cavity's voltage.
        bmad_cnv.set_klystron_enld(simulator, "KLYS_OVL", 8.0)
        assert bmad_cnv.get_cavity_areq(simulator, "CAV1") == pytest.approx(8.0)

    def test_pdes_roundtrip(self, simulator):
        bmad_cnv.set_klystron_pdes(simulator, "KLYS_OVL", 45.0)
        assert bmad_cnv.get_klystron_pdes(simulator, "KLYS_OVL") == pytest.approx(45.0)

    def test_pdes_drives_cavity_preq(self, simulator):
        bmad_cnv.set_klystron_pdes(simulator, "KLYS_OVL", 30.0)
        assert bmad_cnv.get_cavity_preq(simulator, "CAV1") == pytest.approx(30.0)

    def test_pact_is_pdes_alias(self):
        assert bmad_cnv.get_klystron_pact is bmad_cnv.get_klystron_pdes

    def test_stat_roundtrip(self, simulator):
        bmad_cnv.set_klystron_stat(simulator, "KLYS_OVL", 0)
        assert bmad_cnv.get_klystron_stat(simulator, "KLYS_OVL") == 0
        bmad_cnv.set_klystron_stat(simulator, "KLYS_OVL", 1)
        assert bmad_cnv.get_klystron_stat(simulator, "KLYS_OVL") == 1

    def test_stat_invalid_value_raises(self, simulator):
        with pytest.raises(ValueError):
            bmad_cnv.set_klystron_stat(simulator, "KLYS_OVL", 5)


class TestScreenImage:
    def test_get_screen_image_in_beam_mode(self, simulator):
        simulator.cmd("set beam saved_at = OTR1,BEGINNING,END")
        simulator.cmd("set global track_type = beam")
        image = bmad_cnv.get_screen_image(simulator, "OTR1", (64, 64), 2e-4)
        assert image.shape == (64, 64)
        assert image.sum() > 0

    def test_get_screen_image_zero_when_not_beam_mode(self, simulator):
        image = bmad_cnv.get_screen_image(simulator, "OTR1", (10, 10), 1e-4)
        assert image.shape == (10, 10)
        assert (image == 0).all()

    def test_get_screen_image_array_size(self, simulator):
        assert bmad_cnv.get_screen_image_array_size(simulator, "OTR1", (80, 60), 0) == 80
        assert bmad_cnv.get_screen_image_array_size(simulator, "OTR1", (80, 60), 1) == 60

    def test_get_screen_resolution(self, simulator):
        assert bmad_cnv.get_screen_resolution(simulator, "OTR1", 2e-4) == pytest.approx(200.0)
