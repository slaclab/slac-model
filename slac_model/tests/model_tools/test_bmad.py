"""Tests for `cu_energy_match` and `cu_energy_fudge` against the CU_HXR lattice."""

import os
from pathlib import Path

import pytest

pytest.importorskip("pytao")
pytest.importorskip("lume_bmad")

import slac_db.device as device

from slac_model.models.cu import get_cu_hxr_bmad_model
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


def test_fudge_preserves_disabled_station_phase_and_enld(simulator, baseline_edes):
    """A disabled L3 station must retain its drive and have the SBST offset
    folded into its own PHASE_DEG once the overlay is zeroed."""
    stn = device.get_devices(area='L3', device_type='LCAV')[0]
    orig = simulator.ele_control_var(stn)
    orig_phase = orig['PHASE_DEG']
    orig_enld = orig['ENLD_MEV']
    sbst_key = f'SBST_{stn[1:3]}'
    sbst_phase = simulator.ele_control_var(sbst_key)['PHASE_DEG']

    simulator.cmd(f'set ele {stn} IN_USE = 0')

    cu_energy_fudge(dict(baseline_edes), simulator)

    after = simulator.ele_control_var(stn)
    assert after['ENLD_MEV'] == pytest.approx(orig_enld, abs=1e-9), (
        f'{stn} ENLD_MEV was overwritten while disabled'
    )
    assert after['PHASE_DEG'] == pytest.approx(orig_phase + sbst_phase, abs=1e-6), (
        f'{stn} PHASE_DEG did not absorb the SBST offset before overlay zeroing'
    )
    assert simulator.ele_control_var(sbst_key)['PHASE_DEG'] == pytest.approx(0.0, abs=1e-9)


def test_fudge_preserves_disabled_l2_station_folds_l2_overlay(simulator, baseline_edes):
    """A disabled L2 station must carry both the O_L2 and SBST offsets after a fudge."""
    stn = device.get_devices(area='L2', device_type='LCAV')[0]
    orig = simulator.ele_control_var(stn)
    orig_phase = orig['PHASE_DEG']
    orig_enld = orig['ENLD_MEV']
    sbst_key = f'SBST_{stn[1:3]}'
    sbst_phase = simulator.ele_control_var(sbst_key)['PHASE_DEG']
    l2_phase_init = simulator.ele_control_var('O_L2')['PHASE_DEG']

    if l2_phase_init == 0 and sbst_phase == 0:
        pytest.skip('O_L2 and SBST both zero — nothing to fold in; test vacuous')

    simulator.cmd(f'set ele {stn} IN_USE = 0')

    cu_energy_fudge(dict(baseline_edes), simulator)

    after = simulator.ele_control_var(stn)
    assert after['ENLD_MEV'] == pytest.approx(orig_enld, abs=1e-9)
    assert after['PHASE_DEG'] == pytest.approx(
        orig_phase + l2_phase_init + sbst_phase, abs=1e-6
    )


def test_fudge_zero_amp_region_bakes_in_station_phases(simulator, baseline_edes):
    """When a region's expected gain is zero and all its stations are disabled,
    the early `continue` must not leave stations stranded without the overlay
    offset — the other region's overlay-zero still runs."""
    l3_stations = device.get_devices(area='L3', device_type='LCAV')
    pre = {
        stn: (
            simulator.ele_control_var(stn)['PHASE_DEG'],
            simulator.ele_control_var(f'SBST_{stn[1:3]}')['PHASE_DEG'],
        )
        for stn in l3_stations
    }

    for stn in l3_stations:
        simulator.cmd(f'set ele {stn} IN_USE = 0')

    edes = dict(baseline_edes)
    edes['L3'] = edes['L2']

    cu_energy_fudge(edes, simulator)

    for stn, (orig_phase, sbst_phase) in pre.items():
        after = simulator.ele_control_var(stn)['PHASE_DEG']
        assert after == pytest.approx(orig_phase + sbst_phase, abs=1e-6), (
            f'{stn} phase lost SBST offset through zero-amp early return'
        )
    for stn in l3_stations:
        sbst_key = f'SBST_{stn[1:3]}'
        assert simulator.ele_control_var(sbst_key)['PHASE_DEG'] == pytest.approx(0.0, abs=1e-9)
