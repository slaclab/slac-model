import slac_db.device as dev
from slac_model.bmad import model_tools
from slac_model.models.cu import get_cu_hxr_bmad_va
from slac_model.models.cu_matching_functions import cu_get_edes
import epics 
import numpy as np

cu_hxr = get_cu_hxr_bmad_va(start_element='DL00', end_element='END')
cu_hxr.tao.cmd('set ele DL00:END field_master=True')

elements = model_tools.get_modeled_elements(beampath='CU_HXR')
pvs = model_tools.get_model_pvs(elements, beam_path='CU_HXR')



pv_data = model_tools.get_pv_data(pvs)

remove_list = ['QUAD:IN20:121:BDES', 'QUAD:IN20:122:BDES', 'ACCL:LI21:1:L1S_ADES_DS0',
               'ACCL:LI21:1:L1S_PDES_DS0', 'ACCL:LI24:100:KLY_PDES:SETDATA_1',
               'ACCL:LI24:200:KLY_PDES:SETDATA_1', 'ACCL:LI24:300:KLY_PDES:SETDATA_1']
[pv_data.pop(k, None) for k in remove_list]

cu_hxr.set(pv_data)


import slac_model.bmad.model_plots as model_plots
output = model_tools.get_output(cu_hxr.tao)
model_plots.plot_twiss(cu_hxr.tao,output)
