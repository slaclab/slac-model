from pytao import Tao
import logging
import re

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
