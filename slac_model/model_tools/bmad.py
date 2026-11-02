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

    # Remove sentinel elements and preserve Tao's unique element IDs to avoid
    # ambiguous lookups when multiple elements share the same base name.
    elements = list(
        dict.fromkeys(elem for elem in elements if elem not in ("BEGINNING", "END"))
    )

    normalized_elements = []

    # if an element has "#<number>" suffix then it is a split element, remove the suffix -- duplicates will be removed later while preserving order
    elements = [elem.split("#")[0] for elem in elements]

    # if an element matches the klystron segment pattern similar to K21_1D#1, normalize it to the base klystron name without the segment suffix K21_1
    for elem in elements:
        match = KLYSTRON_PATTERN.match(elem)
        if match:
            normalized_elements.append(
                elem[:-1]
            )  # remove the segment suffix to get the overlay element name

        else:
            normalized_elements.append(elem)

    # remove duplicates while preserving order
    normalized_elements = list(dict.fromkeys(normalized_elements))

    return normalized_elements


def cu_energy_match(edes, tao):
    """Drive linac_fudge[1..3] so the Bmad model's L1/L2/L3 energies
    match the given EDES values.

    Parameters
    ----------
    edes : dict
        ``{'L1': float, 'L2': float, 'L3': float}`` — energies in GeV.
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

    Parameters
    ----------
    edes : dict
        ``{'L1': float, 'L2': float, 'L3': float}`` - energies in GeV,
        same shape as :func:`cu_energy_match`'s input.
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
        tao.cmd(f'set ele O_L2 PHASE_DEG = 0')
        for sbst in sbst_phases:
            tao.cmd(f'set ele {sbst} PHASE_DEG = 0')
        tao.cmd("set global lattice_calc_on = F")
        for indx, stn in enumerate(stations):
            if not tao.ele_control_var(stn)['IN_USE']:
                continue
            tao.cmd(f'set ele {stn} ENLD_MEV = {amplF[indx]}')
            tao.cmd(f'set ele {stn} PHASE_DEG = {phasF[indx]}')
        tao.cmd("set global lattice_calc_on = T")   
