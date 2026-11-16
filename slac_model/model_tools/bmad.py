from pytao import Tao
import logging
import re

import numpy as np
import slac_db.device as device

logger = logging.getLogger(__name__)
KLYSTRON_PATTERN = re.compile(r"^K\d+_\d+[A-Z]$")


_CAVITY_CS_PREFIXES = {"KLYS", "ACCL", "LCAV"}

_BEAM_PATH_BEAMCODE = {
    "CU_HXR": 1,
    "CU_SXR": 2,
}

_MAGNET_DES_ATTRS = {
    "BEND": ["BDES"],
    "BTRM": ["BDES"],
    "QUAD": ["BDES"],
    "SOLE": ["BDES"],
    "XCOR": ["BDES"],
    "YCOR": ["BDES"],
}


def _klys_pvs(mad_name, beamcode):
    """Return (amplitude_pv, phase_pv, beamcode_pv) for a cavity device.

    Looks up the beamcode-specific DES accessors on the device,
    falling back to the plain amplitude/phase_desired entries if the
    beamcode-specific keys are absent. The beamcode PV comes from the
    'beamcode{N}' accessor entry. Any return slot may be None.

    Args:
        mad_name (str): MAD device name.
        beamcode (int): Klystron beamcode (e.g. 1 for CU_HXR).

    Returns:
        tuple[str | None, str | None, str | None]:
            (amplitude_pv, phase_pv, beamcode_pv)
    """
    acc = device.get_all_accessors(mad_name) or {}
    ampl_pv = (
        acc.get(f"amplitude_des_beamcode{beamcode}")
        or acc.get("amplitude_des")
        or acc.get("amplitude")
    )
    phase_pv = (
        acc.get(f"phase_desired_beamcode{beamcode}")
        or acc.get("phase_desired")
    )
    beamcode_pv = acc.get(f"beamcode{beamcode}")
    return ampl_pv, phase_pv, beamcode_pv


def get_model_pvs(devices, beampath=None):
    """Return a flat list of DES PVs for a list of MAD devices.

    Cavity-shaped entries (cs_name prefix in KLYS/ACCL/LCAV) contribute
    their resolved amplitude, phase, and beamcode PVs when present.
    Magnet entries contribute cs_name:BDES per device_type.

    Args:
        devices (list[str]): MAD device names (e.g. the return of
            `get_devices(beampath=..., device_type=...)`).
        beampath (str) optional: Beam path name used to resolve the
            klystron beamcode (CU_HXR -> 1, CU_SXR -> 2). Required if
            any device in the list is cavity-shaped.

    Returns:
        list[str]: PV names in input order; cavity entries contribute
            up to three PVs (amplitude, phase, beamcode); magnet
            entries contribute one (BDES).

    Raises:
        ValueError: If a cavity-shaped device is encountered without
            a beampath beamcode mapping, or a device's `device_type`
            is neither cavity-shaped nor in `_MAGNET_DES_ATTRS`.
    """
    beamcode = (
        _BEAM_PATH_BEAMCODE.get(beampath.upper()) if beampath else None
    )
    pvs = []
    for mad_name in devices:
        cs_name = device.get_attribute(mad_name, "cs_name")
        device_type = device.get_attribute(mad_name, "device_type")
        cs_prefix = cs_name.split(":")[0] if cs_name else ""
        if cs_prefix in _CAVITY_CS_PREFIXES:
            if beamcode is None:
                raise ValueError(
                    f"beampath {beampath!r} is required for beamcode "
                    f"resolution on cavity-shaped device {mad_name!r}. "
                    f"Supported: {list(_BEAM_PATH_BEAMCODE)}."
                )
            for pv in _klys_pvs(mad_name, beamcode):
                if pv:
                    pvs.append(pv)
        elif device_type in _MAGNET_DES_ATTRS:
            for suffix in _MAGNET_DES_ATTRS[device_type]:
                pvs.append(f"{cs_name}:{suffix}")
        else:
            raise ValueError(
                f"Unsupported device type {device_type!r} for device "
                f"{mad_name!r} (cs_name={cs_name!r}). Supported: "
                f"{', '.join(sorted(_MAGNET_DES_ATTRS))}."
            )
    return pvs

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
