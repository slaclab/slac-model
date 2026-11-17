"""SXR dual-energy predictive matching driven through a CU_SXR VA.

Builds a live CU_SXR virtual accelerator seeded from CU_HXR PVs
through the common linac, then re-seeded from CU_SXR PVs and matched
through the LTU. Bundles its own ``get_model_pvs``, ``cu_get_edes``,
and ``beta_match`` helpers so ``slac_model.model_tools.bmad`` on this
branch stays identical to ``main``; only :func:`cu_energy_fudge` is
imported from there.
"""

import logging

import slac_db.device as dev
from slac_tools import epics_data

from slac_model.models.cu import get_cu_sxr_bmad_va
from slac_model.model_tools.bmad import cu_energy_fudge


logger = logging.getLogger(__name__)


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
    acc = dev.get_all_accessors(mad_name) or {}
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
        cs_name = dev.get_attribute(mad_name, "cs_name")
        device_type = dev.get_attribute(mad_name, "device_type")
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


def cu_get_edes(beampath):
    """Read L1/L2/L3 EDES energies (GeV) from EPICS for the given beampath."""
    import epics

    L1_edes = epics.caget('BEND:LI21:231:EDES')
    L2_edes = epics.caget('BEND:LI24:790:EDES')
    L3_edes = None
    if beampath == 'CU_HXR':
        L3_edes = epics.caget('BEND:DMPH:400:EDES')
    if beampath == 'CU_SXR':
        L3_edes = epics.caget('BEND:DMPS:400:EDES')
    return {'L1': L1_edes, 'L2': L2_edes, 'L3': L3_edes}


def beta_match(tao, data, var, data_range, var_range):
    """Run Tao's optimizer to match `data[data_range]` to design using
    `var[var_range]`."""
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


def sxr_dual_energy_match():
    """Match cu_sxr while preserving the cu_hxr energy gain profile.

    Builds a CU_SXR VA, applies a live CU_HXR PV snapshot (K21_3..Q50Q2)
    via ``va.set``, energy-fudges to the HXR EDES, then overlays the
    CU_SXR PV snapshot (K21_3..) and fudges to the SXR EDES. Final step
    is a ``beta_match`` through WS32B using ``q_LTUS``.
    """
    logger.info("Generating Cu Linac Beamcode 1 LIVE model")
    va = get_cu_sxr_bmad_va(start_element="BEGL2", end_element="END")

    hxr_devices = dev.get_devices(beampath="CU_HXR", device_type=["LCAV", "QUAD"])
    hxr_devices = hxr_devices[
        hxr_devices.index("K21_3"): hxr_devices.index("Q50Q2") + 1
    ]
    va.tao.cmd("set ele K26_3 IN_USE = 0")
    hxr_pvs = get_model_pvs(hxr_devices, beampath="CU_HXR")
    va.set(epics_data.get_live(hxr_pvs))
    cu_energy_fudge(cu_get_edes("CU_HXR"), va.tao)

    logger.info("Updating energy gain from Beamcode 2")
    va.tao.cmd("set ele BEGINNING:ENDCLTH_1 field_master=False")
    sxr_devices = dev.get_devices(beampath="CU_SXR", device_type=["LCAV"])
    sxr_devices = sxr_devices[sxr_devices.index("K21_3"):]
    sxr_pvs = get_model_pvs(sxr_devices, beampath="CU_SXR")
    va.set(epics_data.get_live(sxr_pvs))
    cu_energy_fudge(cu_get_edes("CU_SXR"), va.tao)

    beta_match(va.tao, "WS32B", "q_LTUS", "1:4", "1:5")

    for quad in get_tao_var_ele_name(va, "q_LTUS"):
        quad_accessor = dev.build_accessor(quad)
        pv_bctrl = quad_accessor['bctrl']['cs_address']
        quad_bctrl = va.get([pv_bctrl])[pv_bctrl]
        logger.info("   %s %s: %7.2f", quad, pv_bctrl, quad_bctrl)
        if write_bctrl:
            import epics
            epics.caput(pv.replace(pv_bctrl, quad_bctrl)
    return va


def get_tao_var_ele_name(va, tao_variable, useit_opt=True):
    """Return Bmad element names for entries in Tao variable
    ``tao_variable`` whose ``useit_opt`` flag matches input."""
    return [
        row["ele_name"]
        for row in va.tao.var_v1_array(tao_variable)["data"]
        if row.get("useit_opt") == useit_opt
    ]
