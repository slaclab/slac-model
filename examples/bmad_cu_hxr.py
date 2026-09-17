import slac_db.device as dev
from slac_model.bmad import model_tools
from pytao import Tao

OPTIONS = '-noplot '
INIT = f'-init $LCLS_LATTICE/bmad/models/cu_hxr/tao.init {OPTIONS}'
tao = Tao(INIT)
tao.cmd('set ele BEGINNING:END field_master=True')


elements = model_tooks.get_modeled_elements(beampath='CU_HXR')
pvs = model_tools.get_model_pvs(elements, beam_path='CU_HXR')
pv_data = model_tools.get_pv_data(pvs)

#TODO update tao with pv_data from VA code.  For now use bmad_util code

tao_cmds = model_tools.pv_to_tao_cmd(pv_data, tao)


