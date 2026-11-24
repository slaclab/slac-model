"""SXR dual-energy predictive matching driven through a CU_SXR VA.

Builds a live CU_SXR virtual accelerator seeded from CU_HXR PVs
through the common linac, then re-seeded from CU_SXR PVs and matched
through the LTU. Bundles its own ``cu_get_edes`` and ``beta_match``
helpers; :func:`cu_energy_fudge` is imported from
:mod:`slac_model.model_tools.bmad` and :func:`get_model_pvs` from
:mod:`slac_model.live`.
"""

import logging

import slac_db.device as dev
from slac_tools import epics_data

from slac_model.models.cu import get_cu_sxr_bmad_va
from slac_model.model_tools.bmad import cu_energy_fudge
from slac_model.live import get_model_pvs


logger = logging.getLogger(__name__)


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


def sxr_dual_energy_match(write_bctrl=False):
    """Match cu_sxr while preserving the cu_hxr energy gain profile.

    Builds a CU_SXR VA, applies a live CU_HXR PV snapshot (K21_3..Q50Q2)
    via ``va.set``, energy-fudges to the HXR EDES, then overlays the
    CU_SXR PV snapshot (K21_3..) and fudges to the SXR EDES. Final step
    is a ``beta_match`` through WS32B using ``q_LTUS``. When
    ``write_bctrl`` is True, each post-match q_LTUS quad BCTRL is pushed
    back to its EPICS BCTRL record.
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
            epics.caput(pv_bctrl, quad_bctrl)
    return va


def get_tao_var_ele_name(va, tao_variable, useit_opt=True):
    """Return Bmad element names for entries in Tao variable
    ``tao_variable`` whose ``useit_opt`` flag matches input."""
    return [
        row["ele_name"]
        for row in va.tao.var_v1_array(tao_variable)["data"]
        if row.get("useit_opt") == useit_opt
    ]
