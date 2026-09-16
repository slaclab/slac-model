import slac_db.device as dev
from slac_model import bmad_util
from pytao import Tao

OPTIONS = '-noplot '
INIT = f'-init $LCLS_LATTICE/bmad/models/cu_hxr/tao.init {OPTIONS}'
tao = Tao(INIT)
tao.cmd('set ele BEGINNING:END field_master=True')


elements = bmad_util.get_modeled_elements(beampath='CU_HXR')
pvs = bmad_util.get_model_pvs(elements, beam_path='CU_HXR')
pv_data = bmad_util.get_pv_data(pvs)

#TODO update tao with pv_data

