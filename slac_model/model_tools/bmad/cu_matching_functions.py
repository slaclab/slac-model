"""Match a Cu linac Bmad model's energy profile to live/EDES values.

Workflow:
    from slac_model.model_tools.bmad import cu_matching_functions
    cu_matching_functions.cu_energy_match(model_instance)

The two public helpers are:
    - cu_get_edes(line): read the desired-energy EPICS PVs for the
      three Cu linac stages (L1, L2, L3) and return them as a dict.
    - cu_energy_match(model_instance): drive the Bmad model's
      ``linac_fudge`` variables (via Tao) until each stage's modeled
      E_TOT matches the measured EDES.
"""

import epics
import slac_db.device as device


def cu_get_edes(line):
    """Read desired-energy (EDES) PVs for the Cu linac stages.

    L1 and L2 are shared between HXR and SXR. L3 depends on which
    dumpline is of interest and is selected by *line*.

    Parameters
    ----------
    line : str
        Either ``'HXR'`` or ``'SXR'``. Any other value leaves
        ``L3`` as ``None`` instead of raising.

    Returns
    -------
    dict
        ``{'L1': float|None, 'L2': float|None, 'L3': float|None}`` —
        values are whatever ``epics.caget`` returns (so ``None`` on
        read failure).
    """
    L1_edes = epics.caget('BEND:LI21:231:EDES')
    L2_edes = epics.caget('BEND:LI24:790:EDES')
    L3_edes = None
    if line == 'HXR':
        L3_edes = epics.caget('BEND:DMPH:400:EDES')
    if line == 'SXR':
        L3_edes = epics.caget('BEND:DMPS:400:EDES')
    return {'L1': L1_edes, 'L2': L2_edes, 'L3': L3_edes}


def cu_energy_match(model_instance):
    """Match the Bmad model's L1/L2/L3 energies to measured EDES.

    For each of the three Cu linac stages, this function:
        1. reads the live EDES value via :func:`cu_get_edes`;
        2. prints the pre-match (meas, model) energy pair at the
           corresponding ``END{BC1,BC2,L3}`` element;
        3. tells Tao to minimise the mismatch by varying the
           corresponding ``linac_fudge`` variable against the
           ``{BC1,BC2,L3}.energy[2]`` datum;
        4. prints the post-match (meas, model) energy pair;
    and finally prints the ``show var linac_fudge`` summary so the
    operator can see the fudge values Tao landed on.

    This mutates *model_instance* (via Tao commands). It does not
    return anything.

    Parameters
    ----------
    model_instance : lume_bmad.model.LUMEBmadModel
        A live model whose ``.tao`` is a connected pytao instance.
        The model is expected to carry ``linac_fudge[1..3]``
        variables and ``BC1.energy[2]`` / ``BC2.energy[2]`` /
        ``L3.energy[2]`` datums (standard CU_HXR tao init).
    """
    edes = cu_get_edes('HXR')
    stages = [
        ('L1', 'ENDBC1', 'BC1.energy[2]', 'linac_fudge[1]'),
        ('L2', 'ENDBC2', 'BC2.energy[2]', 'linac_fudge[2]'),
        ('L3', 'ENDL3',  'L3.energy[2]',  'linac_fudge[3]'),
    ]
    for label, end_ele, dat, var in stages:
        meas = edes[label]
        before = model_instance.tao.ele_gen_attribs(end_ele)['E_TOT'] / 1e9
        print(f'{label} Energy before (meas, model): ({meas}, {before:0.4})')
        model_instance.tao.cmds([
            'veto var *',
            'veto dat *',
            f'set dat {dat}|meas = {meas}E9',
            f'use dat {dat}',
            f'use var {var}',
        ])
        model_instance.tao.cmd('run')
        after = model_instance.tao.ele_gen_attribs(end_ele)['E_TOT'] / 1e9
        print(f'{label} Energy after  (meas, model): ({meas}, {after:0.4})')
    for line in model_instance.tao.cmd('show var linac_fudge'):
        print(line)
