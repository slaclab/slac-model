import epics
import slac_db.device as dev
#from virtual_accelerator.models.cu_hxr import get_cu_hxr_bmad_model
from slac_model.models.cu import get_cu_hxr_bmad_va
from slac_model.model_tools.bmad import cu_energy_match, cu_energy_fudge

def tc(cmd):
    [print(l) for l in cu_hxr.tao.cmd(cmd)]

    
#cu_hxr = get_cu_hxr_bmad_model()  #VA
cu_hxr = get_cu_hxr_bmad_va()
tc('set ele * field_master = True')

RF_QUAD = ['LCAV','QUAD']
devices = dev.get_model_devices(beampath='CU_HXR', device_types = RF_QUAD)
pvs = dev.get_model_pvs(devices)


def get_pv_data(pvs):
    import epics
    values = epics.caget_many(pvs)
    failed = [pv for pv, val in zip(pvs, values) if val is None]
    if failed:
        raise ValueError(
            f"Could not read {len(failed)} PV(s): {failed}"
        )
    return dict(zip(pvs, values))

pv_data = get_pv_data(pvs)
#cu_hxr.set(pv_data)

#TODO  clean up device list based on model 
remove_list = ['QUAD:IN20:121:BDES', 'QUAD:IN20:122:BDES', 'ACCL:LI21:1:L1S_ADES_DS0',
               'ACCL:LI21:1:L1S_PDES_DS0', 
               'QUAD:IN20:361:BDES', 'QUAD:IN20:371:BDES', 'QUAD:IN20:425:BDES',
               'QUAD:IN20:441:BDES', 'QUAD:IN20:511:BDES', 'QUAD:IN20:525:BDES']

[pv_data.pop(k, None) for k in remove_list]

cu_hxr.set(pv_data)


def cu_get_edes(line):
    L1_edes = epics.caget('BEND:LI21:231:EDES')
    L2_edes = epics.caget('BEND:LI24:790:EDES')
    L3_edes = None
    if line == 'HXR':
        L3_edes = epics.caget('BEND:DMPH:400:EDES')
    if line == 'SXR':
        L3_edes = epics.caget('BEND:DMPS:400:EDES')
    return {'L1': L1_edes, 'L2': L2_edes, 'L3': L3_edes}

hxr_edes = cu_get_edes('HXR')

cu_energy_fudge(hxr_edes, cu_hxr.tao)
cu_hxr.update_state()

print(cu_hxr.tao.ele_gen_attribs('ENDL2')['E_TOT']/1E9)
print(cu_hxr.tao.ele_gen_attribs('ENDL3')['E_TOT']/1E9)
