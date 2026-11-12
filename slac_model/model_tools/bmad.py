from pytao import Tao
import logging
import re

import numpy as np
import slac_db.device as device

logger = logging.getLogger(__name__)
KLYSTRON_PATTERN = re.compile(r"^K\d+_\d+[A-Z]$")

def get_bmad_device_names(tao: Tao):
    """Return slac-db device names for the given BMAD Tao instance.

    Parameters
    ----------
    tao : Tao
        Active Tao instance containing the currently loaded lattice.

    Returns
    -------
    list[str]
        Ordered element names with slac-db device names.
    """
    elements = tao.lat_list("*", "ele.name")
    indices = tao.lat_list("*", "ele.ix_ele")

    normalized_elements = []
    for elem, ix in zip(elements, indices):
        if elem in ("BEGINNING", "END"):
            continue
        # Deferred elements (type starts with "@") are placeholders for
        # hardware not yet installed in the accelerator.
        if tao.ele_head(int(ix))["type"].startswith("@"):
            continue
        # Strip the "#<number>" split suffix; duplicates are removed below.
        elem = elem.split("#")[0]
        match = KLYSTRON_PATTERN.match(elem)
        if match:
            normalized_elements.append(
                elem[:-1]
            )  # klystron segment -> overlay element name
        else:
            normalized_elements.append(elem)

    # remove duplicates while preserving order
    return list(dict.fromkeys(normalized_elements))


def cu_energy_match(edes, tao):
    """Drive linac_fudge[1..3] so the Bmad model's L1/L2/L3 energies
    match the given EDES values.

    Parameters
    ----------
    edes : dict
        ``{'L1': float, 'L2': float, 'L3': float}`` — ENDBC1, ENDBC2 and
        ENDL3 EDES energies in GeV.
        Callers typically build this with a small ``cu_get_edes(line)``
        helper (see the example scripts in ``examples/``).
    tao : pytao.Tao
        Active Tao instance for the model being matched. Must carry
        ``linac_fudge[1..3]`` variables and
        ``BC1.energy[2]`` / ``BC2.energy[2]`` / ``L3.energy[2]`` datums
        (standard CU_HXR tao init).
    """
    stages = [
        ('L1', 'ENDBC1', 'BC1.energy[2]', 'linac_fudge[1]'),
        ('L2', 'ENDBC2', 'BC2.energy[2]', 'linac_fudge[2]'),
        ('L3', 'ENDL3',  'L3.energy[2]',  'linac_fudge[3]'),
    ]
    for label, end_ele, dat, var in stages:
        meas = edes[label]
        before = tao.ele_gen_attribs(end_ele)['E_TOT'] / 1e9
        logger.debug('%s Energy before (meas, model): (%s, %0.4g)', label, meas, before)
        tao.cmds([
            'veto var *',
            'veto dat *',
            f'set dat {dat}|meas = {meas}E9',
            f'use dat {dat}',
            f'use var {var}',
        ])
        tao.cmd('run')
        after = tao.ele_gen_attribs(end_ele)['E_TOT'] / 1e9
        logger.debug('%s Energy after  (meas, model): (%s, %0.4g)', label, meas, after)
    for line in tao.cmd('show var linac_fudge'):
        logger.debug('%s', line)


def cu_energy_fudge(edes, tao):
    """Compute per-station amplitude/phase 'fudge' for L2 and L3 so each
    region's total gain matches the EDES values, and push the fudged
    ``ENLD_MEV`` / ``PHASE_DEG`` back into the Bmad model.

    NOTE: Overlays set the element directly, overriding the previous
    setting. So we need to take into account the L2 Phase and SBST
    overlays values and then zero them in order to write to the
    individual KLYS overlays.

    Parameters
    ----------
    edes : dict
        ``{'L1': float, 'L2': float, 'L3': float}`` — ENDBC1, ENDBC2 and
        ENDL3 EDES energies in GeV, same shape as :func:`cu_energy_match`'s
        input.
    tao : pytao.Tao
        Active Tao instance for the CU_HXR (or compatible) model.
    """
    regions = [
        ('L2', edes['L2'] - edes['L1']),
        ('L3', edes['L3'] - edes['L2']),
    ]

    l2_phase_init = tao.ele_control_var('O_L2')['PHASE_DEG']

    region_stations = {
        region: device.get_devices(area=region, device_type='LCAV')
        for region, _ in regions
    }

    sbst_phases = {}
    for stations in region_stations.values():
        for stn in stations:
            key = f'SBST_{stn[1:3]}'
            if key not in sbst_phases:
                sbst_phases[key] = tao.ele_control_var(key)['PHASE_DEG']

    # Bake the shared O_L2 and SBST_* phase offsets into every station's
    # own PHASE_DEG before zeroing those overlays. Without this, stations
    # we don't rewrite in the per-region writeback (disabled stations, or
    # any station in a region taken by the zero-amplitude early return)
    # silently lose the overlay contribution to their effective RF phase.
    tao.cmd("set global lattice_calc_on = F")
    for region, stations in region_stations.items():
        l2_phase = l2_phase_init if region == 'L2' else 0
        for stn in stations:
            vals = tao.ele_control_var(stn)
            baked = vals['PHASE_DEG'] + l2_phase + sbst_phases[f'SBST_{stn[1:3]}']
            tao.cmd(f'set ele {stn} PHASE_DEG = {baked}')
    tao.cmd('set ele O_L2 PHASE_DEG = 0')
    for sbst in sbst_phases:
        tao.cmd(f'set ele {sbst} PHASE_DEG = 0')
    tao.cmd("set global lattice_calc_on = T")
    l2_phase_init = 0
    sbst_phases = {k: 0 for k in sbst_phases}

    for region, expected_gain in regions:
        l2_phase = l2_phase_init if region == 'L2' else 0

        stations = region_stations[region]
        ampl, phas = [], []
        for stn in stations:
            vals = tao.ele_control_var(stn)
            sbst_phase = sbst_phases[f'SBST_{stn[1:3]}']
            ampl.append(vals['ENLD_MEV'] * vals['IN_USE'])
            phas.append(vals['PHASE_DEG'] + l2_phase + sbst_phase)

        ampl = np.array(ampl)
        phas = np.deg2rad(np.array(phas))
        gain_measured = ampl * np.cos(phas)
        total_ampl = ampl.sum()
        if total_ampl == 0:
            if expected_gain == 0:
                continue
            raise ValueError(f'Cannot match {region}: no nonzero in-use station amplitudes')
        dF = (1000 * expected_gain - gain_measured.sum()) / total_ampl
        complex_gain_fudged = ampl * np.exp(1j * phas) + ampl * dF
        amplF = np.abs(complex_gain_fudged)
        phasF = np.degrees(np.angle(complex_gain_fudged))

        logger.debug('%s fudge dF=%0.4g (expected_gain=%s GeV)', region, dF, expected_gain)
        tao.cmd("set global lattice_calc_on = F")
        for indx, stn in enumerate(stations):
            if not tao.ele_control_var(stn)['IN_USE']:
                continue
            tao.cmd(f'set ele {stn} ENLD_MEV = {amplF[indx]}')
            tao.cmd(f'set ele {stn} PHASE_DEG = {phasF[indx]}')
        tao.cmd("set global lattice_calc_on = T")


def _apply_live_rf_quad_pvs(tao, beampath, end_device=None):
    """Fetch live LCAV/QUAD PVs from K21_3 (optionally to ``end_device``) on
    ``beampath`` and push them into ``tao`` via ``apply_pv_data``. Also
    disables ``K26_3`` in the model (not an operational station)."""
    from slac_model.pv_conversion.bmad import apply_pv_data
    from slac_tools import epics_data
    import slac_db.device as dev
    
    devices = dev.get_devices(beampath=beampath, device_type=['LCAV', 'QUAD'])
    start = devices.index('K21_3')
    stop = devices.index(end_device) + 1 if end_device is not None else None
    devices = devices[start:stop]
    tao.cmd('set ele K26_3 IN_USE = 0')
    pvs = dev.get_model_pvs(devices, beampath=beampath)
    pv_data = epics_data.get_live(pvs)
    apply_pv_data(tao, pv_data, beampath=beampath)


def sxr_dual_energy_match():
    # Match cu_sxr as we keep cu_hxr energy gain profile
    from slac_model.models.cu import get_cu_sxr_bmad_model

    logger.info("Generating Cu Linac Beamcode 1 LIVE model")
    tao = get_cu_sxr_bmad_model(start_element="BEGL2", end_element="END")
    _apply_live_rf_quad_pvs(tao, 'CU_HXR', end_device='Q50Q2')
    cu_energy_fudge(cu_get_edes('CU_HXR'), tao)

    #Change energy to SXR. Do not scale magnets
    logger.info("Updating energy gain from Beamcode 2")
    tao.cmd('set ele BEGINNING:ENDCLTH_1 field_master=True')
    _apply_live_rf_quad_pvs(tao, 'CU_SXR')
    cu_energy_fudge(cu_get_edes('CU_SXR'), tao)

    # match
    beta_match(tao, 'WS32B', 'q_LTUS', '1:4', '1:5')
    return tao


def beta_match(tao, data, var, data_range, var_range):
    match_cmds = [
        f'show lat {data} -attr E_TOT',
        'veto var *',
        'veto dat *@*',
        f'use data {data}[{data_range}]',
        f'use var {var}[{var_range}]',
        'set global n_opti_cycles = 912',
        f'set data {data}|meas = {data}|design',
        f'show data {data}',
    ]
    for cmd in match_cmds:
        tao.cmd(cmd)
    logger.info('beta_match starting optimizer')
    tao.cmd('run')
    
def show_var(tao, tao_variable,  write_bdes = False):
    from slac_model.pv_conversion.bmad import get_quadrupole_bctrl
    var = tao.var_v1_array(tao_variable)['data']
    for line in var:
        quad = line['ele_name']
        bdes = get_quadrupole_bctrl(tao,quad)
        device =  tao.ele_head(quad)['alias']
        logger.info("%s %s BDES:%7.2f", quad, device, bdes)
        if write_bdes:
            epics.caput(device + ':BCTRL', bdes)
        

def cu_get_edes(beampath):
    import epics

    L1_edes = epics.caget('BEND:LI21:231:EDES')
    L2_edes = epics.caget('BEND:LI24:790:EDES')
    L3_edes = None
    if beampath == 'CU_HXR':
        L3_edes = epics.caget('BEND:DMPH:400:EDES')
    if beampath == 'CU_SXR':
        L3_edes = epics.caget('BEND:DMPS:400:EDES')
    return {'L1': L1_edes, 'L2': L2_edes, 'L3': L3_edes}

