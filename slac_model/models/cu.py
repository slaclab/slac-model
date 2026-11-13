"""Build Tao and lume-bmad virtual accelerator instances for the CU_HXR and
CU_SXR lattices.

Provides `get_cu_hxr_bmad_model` / `get_cu_sxr_bmad_model` to construct a
sliced Tao instance for a lattice segment, and `get_cu_hxr_bmad_va` /
`get_cu_sxr_bmad_va` to wrap it in a `LUMEBmadModel` virtual accelerator
with variables and screen dump locations configured.
"""

import os
import logging
import warnings
from slac_model.virtual_accelerator.factory import create_virtual_accelerator

def get_cu_hxr_bmad_model(start_element: str, end_element: str):
    """
    Get the Tao instance for the CU_HXR lattice from `start_element` to `end_element`.

    Parameters
    -------------
    start_element: str
        The starting element for the model.
    end_element: str
        The ending element for the model.

    Returns
    -------
    Tao
        Instance of the Tao class for the CU_HXR lattice.
    """
    from pytao import Tao

    init_path = os.path.join(os.environ["LCLS_LATTICE"], "bmad/models/cu_hxr/tao.init")
    tao = Tao(f"-init {init_path} -noplot -slice_lattice {start_element}:{end_element}")

    # set tracking to start_element
    tao.cmd(f"set beam track_start = {start_element}")
    tao.cmd("set ele * field_master = True")

    return tao

def get_cu_hxr_cheetah_model(start_element: str, end_element: str):
    import torch
    from cheetah.accelerator import Segment
    from cheetah.particles import ParticleBeam
    from lume_cheetah.simulator import CheetahSimulator

    segment = Segment.from_lattice_json(
        os.path.join(os.environ["LCLS_LATTICE"], "cheetah/nc_hxr.json")
    )
    
    # Handle 'end' to mean the last element in the segment
    if end_element.lower() == 'end':
        end_element = 'enddmph_2'
    
    segment = segment.subcell(start=start_element.lower(), end=end_element.lower())

    # Create a default single particle beam at 1 MeV
    initial_beam = ParticleBeam(
        particles = torch.zeros(1,7),
        energy = torch.tensor([1.0e6]),
        particle_charges = torch.tensor([1.0]),
    )

    simulator = CheetahSimulator(segment=segment, initial_beam_distribution=initial_beam)
    return simulator

def get_cu_hxr_bmad_va(
    start_element="OTR2", 
    end_element="END", 
):
    """
    Get the LUMEBmadModel virtual accelerator for the CU_HXR lattice from `start_element` to `end_element`.

    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "OTR2".
    end_element: str, optional
        The ending element for the model. Default is "END".

    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the CU_HXR lattice.
    """
    # Suppress factory warnings during VA creation
    factory_logger = logging.getLogger("slac_model.virtual_accelerator.factory")
    original_level = factory_logger.level
    factory_logger.setLevel(logging.ERROR)
    
    try:
        # get the Tao instance for the CU_HXR lattice from start_element to end_element
        tao = get_cu_hxr_bmad_model(start_element=start_element, end_element=end_element)
        return create_virtual_accelerator(tao, beampath="CU_HXR")
    finally:
        factory_logger.setLevel(original_level)

def get_cu_sxr_bmad_model(start_element: str, end_element: str):
    """
    Get the Tao instance for the CU_SXR lattice from `start_element` to `end_element`.

    Parameters
    -------------
    start_element: str
        The starting element for the model.
    end_element: str
        The ending element for the model.

    Returns
    -------
    Tao
        Instance of the Tao class for the CU_SXR lattice.
    """

    init_path = os.path.join(os.environ["LCLS_LATTICE"], "bmad/models/cu_sxr/tao.init")
    tao = Tao(f"-init {init_path} -noplot -slice_lattice {start_element}:{end_element}")

    # set tracking to start_element
    tao.cmd(f"set beam track_start = {start_element}")
    tao.cmd("set ele * field_master = True")

    return tao


def get_cu_sxr_bmad_va(
    start_element="OTR2",
    end_element="END",
):
    """
    Get the LUMEBmadModel virtual accelerator for the CU_SXR lattice from `start_element` to `end_element`.

    Parameters
    -------------
    start_element: str
        The starting element for the model.
    end_element: str
        The ending element for the model.

    Returns
    -------
    LUMEBmadModel
        Instance of the LUMEBmadModel for the CU_SXR lattice.
    """
    tao = get_cu_sxr_bmad_model(start_element=start_element, end_element=end_element)

    return create_virtual_accelerator(tao, beampath="CU_SXR")


def get_cu_hxr_cheetah_va(
    start_element="OTR2", 
    end_element="END", 
):
    """
    Get the CheetahSimulator virtual accelerator for the CU_HXR lattice from `start_element` to `end_element`.


    Parameters
    -------------
    start_element: str, optional
        The starting element for the model. Default is "OTR2".
    end_element: str, optional
        The ending element for the model. Default is "END".

    Returns
    -------
    CheetahSimulator
        Instance of the CheetahSimulator for the CU_HXR lattice.
    """
    # Suppress factory warnings during VA creation
    factory_logger = logging.getLogger("slac_model.virtual_accelerator.factory")
    original_level = factory_logger.level
    factory_logger.setLevel(logging.ERROR)
    
    try:
        simulator = get_cu_hxr_cheetah_model(start_element=start_element, end_element=end_element)
        return create_virtual_accelerator(simulator, beampath="CU_HXR")
    finally:
        factory_logger.setLevel(original_level)

