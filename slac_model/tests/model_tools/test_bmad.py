"""Tests for `cu_energy_match` and `cu_energy_fudge` against the CU_HXR lattice."""

import os
from pathlib import Path

import pytest

pytest.importorskip("pytao")
pytest.importorskip("lume_bmad")

import slac_db.device as device

from slac_model.models.cu import get_cu_hxr_bmad_model
from slac_model.model_tools import bmad as bmad_tools
from slac_model.model_tools.bmad import cu_energy_fudge, cu_energy_match


REGION_END_ELES = {'L1': 'ENDBC1', 'L2': 'ENDBC2', 'L3': 'ENDL3'}


@pytest.fixture
def simulator():
    """Load the CU_HXR OTR2:END slice afresh for each test.

    `cu_energy_match` and `cu_energy_fudge` mutate `linac_fudge`,
    `O_L2`, every `SBST_*`, and every L2/L3 LCAV, so a module-scoped
    Tao would leak state between tests.
    """
    lattice_root = os.environ.get("LCLS_LATTICE")
    if not lattice_root:
        pytest.skip("Set LCLS_LATTICE to run the CU_HXR integration tests")
    init_file = Path(lattice_root) / "bmad" / "models" / "cu_hxr" / "tao.init"
    assert init_file.is_file(), f"CU_HXR lattice initialization file not found: {init_file}"
    return get_cu_hxr_bmad_model(start_element="OTR2", end_element="END")


@pytest.fixture
def baseline_edes(simulator):
    """Energies (GeV) read from the fresh model — a known-reachable target."""
    return {
        region: simulator.ele_gen_attribs(ele)['E_TOT'] / 1e9
        for region, ele in REGION_END_ELES.items()
    }


def _region_energy(simulator, region):
    return simulator.ele_gen_attribs(REGION_END_ELES[region])['E_TOT'] / 1e9


def test_match_hits_targets(simulator, baseline_edes):
    """`cu_energy_match` should drive each region's end-energy to EDES."""
    edes = {k: v * 1.01 for k, v in baseline_edes.items()}

    cu_energy_match(edes, simulator)

    for region, target in edes.items():
        assert _region_energy(simulator, region) == pytest.approx(target, rel=1e-3)


def test_match_moves_fudge_vars(simulator, baseline_edes):
    """A non-identity match should perturb each `linac_fudge[i]` from unity."""
    edes = {k: v * 1.01 for k, v in baseline_edes.items()}

    cu_energy_match(edes, simulator)

    for i in (1, 2, 3):
        var = simulator.var(f'linac_fudge[{i}]')
        assert var['model_value'] != pytest.approx(1.0, abs=1e-6), (
            f"linac_fudge[{i}] did not move: model_value={var['model_value']}"
        )


def test_match_identity_is_stable(simulator, baseline_edes):
    """Matching to the current energies should leave them within tolerance."""
    cu_energy_match(dict(baseline_edes), simulator)

    for region, target in baseline_edes.items():
        assert _region_energy(simulator, region) == pytest.approx(target, rel=1e-3)


def test_fudge_hits_region_energies(simulator, baseline_edes):
    """`cu_energy_fudge` should land `ENDL2` and `ENDL3` on the EDES values."""
    edes = dict(baseline_edes)
    edes['L2'] = baseline_edes['L2'] * 1.005
    edes['L3'] = baseline_edes['L3'] * 1.005

    cu_energy_fudge(edes, simulator)

    assert _region_energy(simulator, 'L2') == pytest.approx(edes['L2'], rel=5e-4)
    assert _region_energy(simulator, 'L3') == pytest.approx(edes['L3'], rel=5e-4)


def test_fudge_zeros_overlays(simulator, baseline_edes):
    """After the fudge, O_L2 and every touched SBST overlay must be zero."""
    expected_sbst = {
        f'SBST_{stn[1:3]}'
        for region in ('L2', 'L3')
        for stn in device.get_devices(area=region, device_type='LCAV')
    }

    cu_energy_fudge(dict(baseline_edes), simulator)

    assert simulator.ele_control_var('O_L2')['PHASE_DEG'] == pytest.approx(0.0, abs=1e-9)
    for sbst in expected_sbst:
        assert simulator.ele_control_var(sbst)['PHASE_DEG'] == pytest.approx(0.0, abs=1e-9), (
            f"{sbst} PHASE_DEG not zeroed before writeback"
        )


def test_fudge_is_idempotent(simulator, baseline_edes):
    """A second call with the same EDES should not drift the region energies."""
    edes = dict(baseline_edes)
    edes['L2'] = baseline_edes['L2'] * 1.005
    edes['L3'] = baseline_edes['L3'] * 1.005

    cu_energy_fudge(edes, simulator)
    cu_energy_fudge(edes, simulator)

    assert _region_energy(simulator, 'L2') == pytest.approx(edes['L2'], rel=5e-4)
    assert _region_energy(simulator, 'L3') == pytest.approx(edes['L3'], rel=5e-4)


def test_fudge_raises_when_region_is_empty(simulator, baseline_edes, monkeypatch):
    """An empty L2 station list with a nonzero expected gain must raise."""
    real_get_devices = device.get_devices

    def fake_get_devices(**kwargs):
        if kwargs.get('area') == 'L2':
            return []
        return real_get_devices(**kwargs)

    monkeypatch.setattr(bmad_tools.device, 'get_devices', fake_get_devices)

    edes = dict(baseline_edes)
    edes['L2'] = baseline_edes['L1'] + 1.0

    with pytest.raises(ValueError, match='Cannot match L2'):
        cu_energy_fudge(edes, simulator)
