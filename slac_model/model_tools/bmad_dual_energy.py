"""SXR dual-energy predictive matching driven through a CU_SXR VA.

Builds a live CU_SXR virtual accelerator seeded from CU_HXR PVs
through the common linac, then re-seeded from CU_SXR PVs and matched
through the LTU. Bundles its own ``cu_get_edes`` and ``beta_match``
helpers
"""

import logging

import numpy as np
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
    """Run Tao's optimizer to match a Tao datum in  `data[data_range]` 
    to design using using tao variable `var[var_range]`."""
    match_cmds = [
        f'show lat {data} -attr E_TOT',
        'veto var *',
        'veto dat *@*',
        f'use data {data}[{data_range}]',
        f'use var {var}[{var_range}]',
        'set global n_opti_cycles = 1212',
        f'set data {data}|meas = {data}|design',
        f'show data {data}',
    ]
    for cmd in match_cmds:
        tao.cmd(cmd)
    logger.info('beta_match starting optimizer')
    tao.cmd('run')
    tao.cmd('show merit')


def sxr_dual_energy_match(write_bctrl=False, plot=False):
    """Match cu_sxr while preserving the cu_hxr energy gain profile.

    Builds a CU_SXR VA, applies a live CU_HXR PV snapshot (K21_3..Q50Q2),
    energy-fudges to the HXR EDES, then overlays the
    CU_SXR PV snapshot (K21_3..) and fudges to the SXR EDES. Final step
    is a ``beta_match`` through WS32B using ``q_LTUS`` tao variable. When
    ``write_bctrl`` is True, each post-match q_LTUS quad BCTRL is pushed
    back to its EPICS BCTRL record. When ``plot`` is True, β_x(s),
    β_y(s), and E_tot(s) [GeV] are captured at three stages ("Design" /
    "CU_HXR" / "CU_SXR") and rendered via :func:`plot_beta_stages`.
    """
    logger.info("Generating Cu Linac Beamcode 1 LIVE model")
    va = get_cu_sxr_bmad_va(start_element="BEGL2", end_element="END")
    va.tao.cmd("set ele * field_master = False")
    snapshots = []
    if plot:
        snapshots.append(("Design", va.get(["s_ele", "a.beta", "b.beta", "e_tot"])))

    hxr_devices = dev.get_devices(beampath="CU_HXR", device_type=["LCAV"])
    hxr_devices = hxr_devices[
        hxr_devices.index("K21_3"): hxr_devices.index("K30_8") + 1
    ]
    va.tao.cmd("set ele K26_3 IN_USE = 0")
    hxr_pvs = get_model_pvs(hxr_devices, beampath="CU_HXR")
    va.set(epics_data.get_live(hxr_pvs))
    cu_energy_fudge(cu_get_edes("CU_HXR"), va.tao)
    if plot:
        va.update_state()  
        snapshots.append(("CU_HXR", va.get(["s_ele", "a.beta", "b.beta", "e_tot"])))

    logger.info("Updating energy gain from Beamcode 2")
    va.tao.cmd("set ele BEGINNING:ENDCLTH_1 field_master=False")
    sxr_devices = dev.get_devices(beampath="CU_SXR", device_type=["LCAV"])
    sxr_devices = sxr_devices[sxr_devices.index("K21_3"):]
    sxr_pvs = get_model_pvs(sxr_devices, beampath="CU_SXR")
    va.set(epics_data.get_live(sxr_pvs))
    cu_energy_fudge(cu_get_edes("CU_SXR"), va.tao)

    beta_match(va.tao, "WS32B", "q_LTUS", "1:4", "1:5")
    if plot:
        va.update_state()        
        snapshots.append(("CU_SXR", va.get(["s_ele", "a.beta", "b.beta", "e_tot"])))
        plot_beta_stages(snapshots)

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


def plot_beta_stages(snapshots):
    """Plot β_x(s), β_y(s), and E_tot(s) for a sequence of named snapshots.

    `e_tot` is reported by the VA in eV; it is scaled by 1e-9 for the
    Energy panel so the y-axis reads in GeV.

    Parameters
    ----------
    snapshots : list[tuple[str, dict]]
        Ordered list of ``(label, out)`` pairs where ``out`` is the dict
        returned by ``va.get(["s_ele", "a.beta", "b.beta", "e_tot"])``.

    Returns
    -------
    tuple[matplotlib.figure.Figure, ...]
        ``(fig_x, fig_y, fig_e)`` — β_x, β_y, and Energy figures.
    """
    import matplotlib.pyplot as plt

    panels = [
        ("a.beta", "β_x [m]", "β_x vs s", 1.0),
        ("b.beta", "β_y [m]", "β_y vs s", 1.0),
        ("e_tot", "Energy [GeV]", "Energy", 1e-9),
    ]
    figs = []
    for key, ylabel, title, scale in panels:
        fig = plt.figure(figsize=(10, 6))
        for label, out in snapshots:
            plt.plot(out["s_ele"], np.asarray(out[key]) * scale, label=label)
        plt.xlabel("s [m]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        figs.append(fig)
    return tuple(figs)
