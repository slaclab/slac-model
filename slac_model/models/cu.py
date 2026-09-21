import os
from pytao import Tao
from lume_bmad.model import LUMEBmadModel

from slac_model.virtual_accelerator.variables import get_variables, get_all_element_types
from slac_model.models import CU_HXR_PROFMON_INFO

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

    init_path = os.path.join(os.environ["LCLS_LATTICE"], "bmad/models/cu_hxr/tao.init")
    tao = Tao(f"-init {init_path} -noplot -slice_lattice {start_element}:{end_element}")

    # set tracking to start_element
    tao.cmd(f"set beam track_start = {start_element}")

    return tao

def get_cu_hxr_bmad_va(
    start_element="OTR2", 
    end_element="END", 
):
    """
    Get the LUMEBmadModel for the CU_HXR lattice from `start_element` to `end_element`.

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
    # get the Tao instance for the CU_HXR lattice from start_element to end_element
    tao = get_cu_hxr_bmad_model(start_element=start_element, end_element=end_element)

    # get the screen configuration for the CU_HXR lattice from the YAML file
    screen_config_dict = CU_HXR_PROFMON_INFO

    # get variables for all elements in the lattice based on the 
    # element attribute mapping and screen configuration for the model
    variables = get_variables(tao, screen_config_dict)

    # get list of screens that are present in the lattice
    element_types = get_all_element_types(tao)
    active_screens = tuple(
        element
        for element, element_type in element_types.items()
        if element_type == "Screen"
    )

    # create LUMEBmadModel-based VA with the Tao instance, variables, and active screens for beam dumping
    va = LUMEBmadModel(
        tao=tao,
        action_variables=variables,
        dump_locations=list(active_screens),
    )

    return va