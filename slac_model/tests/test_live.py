"""Tests for `slac_model.live.get_model_pvs` against real slac_db devices."""

import slac_db.device as device

from slac_model.live import get_model_pvs


class TestQuad:
    def test_bdes_pv(self):
        cs_name = device.get_attribute("QM01", "cs_name")
        assert get_model_pvs(["QM01"], beampath="CU_HXR") == [f"{cs_name}:BDES"]


class TestCavity:
    def test_amplitude_phase_beamcode_pvs(self):
        # K21_3 has no `amplitude_des_beamcode1` / `phase_desired_beamcode1`
        # accessors, so the resolver falls back to `amplitude` / `phase_desired`.
        assert get_model_pvs(["K21_3"], beampath="CU_HXR") == [
            "KLYS:LI21:31:ENLD",
            "KLYS:LI21:31:PDES",
            "KLYS:LI21:31:BEAMCODE1_STAT",
        ]
