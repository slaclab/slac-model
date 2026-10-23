import slac_db.device as dev
from slac_model.bmad import model_tools
from slac_model import live_data
from slac_model.models.cu import get_cu_hxr_bmad_va
import epics 
import numpy as np

cu_hxr = get_cu_hxr_bmad_va(start_element='DL00', end_element='END')
#cu_hxr.tao.cmd('set ele DL00:END field_master=True')

RF_QUAD = ['LCAV','QUAD']
devices = dev.get_model_devices(beampath='CU_HXR', device_types = RF_QUAD)
pvs = dev.get_model_pvs(devices)
pv_data = live_data.get_pv_data(pvs)

remove_list = ['QUAD:IN20:121:BDES', 'QUAD:IN20:122:BDES', 'ACCL:LI21:1:L1S_ADES_DS0',
               'ACCL:LI21:1:L1S_PDES_DS0', 'ACCL:LI24:100:KLY_PDES:SETDATA_1',
               'ACCL:LI24:200:KLY_PDES:SETDATA_1', 'ACCL:LI24:300:KLY_PDES:SETDATA_1']
[pv_data.pop(k, None) for k in remove_list]

cu_hxr.set(pv_data)


import slac_model.bmad.model_plots as model_plots
model_plots.plot_twiss(cu_hxr)

#Energy Scaling
cu_energy_match(cu_hxr)

#It will be: 
from slac_model.models import cu_matching_functions
cu_matching_functions.cu_energy_match(cu_hxr)




def tc(cmd):
    [print(l) for l in cu_hxr.tao.cmd(cmd)]
