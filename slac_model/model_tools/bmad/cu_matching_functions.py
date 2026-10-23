import epics
import slac_db.device as device

def cu_get_edes(line):
    L1_edes = epics.caget('BEND:LI21:231:EDES')
    L2_edes= epics.caget('BEND:LI24:790:EDES')
    if line == 'HXR':
        L3_edes =  epics.caget(' BEND:DMPH:400:EDES')
    if line == 'SXR':
        L3_edes =  epics.caget(' BEND:DMPS:400:EDES')
    return {'L1': L1_edes, 'L2': L2_edes, 'L3': L3_edes}


def cu_energy_match(model_instance):
    edes = cu_get_edes('HXR')
    bc1_engy_model = model_instance.tao.ele_gen_attribs(f"ENDBC1")["E_TOT"] / 1e9
    bc2_engy_model = model_instance.tao.ele_gen_attribs(f"ENDBC2")["E_TOT"] / 1e9
    L3_engy_model = model_instance.tao.ele_gen_attribs(f"ENDL3")["E_TOT"] / 1e9
    print(f'L1 Energy (meas, model): ({edes["L1"]},  {bc1_engy_model:0.4})')    
    print(f'L2 Energy (meas, model): ({edes["L2"]}, {bc2_engy_model:0.4})')
    print(f'L3 Energy (meas, model): ( {edes["L3"]}, {L3_engy_model})')

    cmds = [
        'veto var *',
        'veto dat *',
        f'set dat BC1.energy[2]|meas = {edes["L1"]}E9',
        'use dat BC1.energy[2]',
        'use var linac_fudge[1]'
    ]
    out  = model_instance.tao.cmds(cmds)
    model_instance.tao.cmd('run')
    bc1_engy_model = model_instance.tao.ele_gen_attribs(f"ENDBC1")["E_TOT"] / 1e9
    print(f'L1 Energy (meas, model): ({edes["L1"]},  {bc1_engy_model:0.4})')
    cmds = [
        'veto var *',
        'veto dat *',
        f'set dat BC2.energy[2]|meas = {edes["L2"]}E9',
        'use dat BC2.energy[2]',
        'use var linac_fudge[2]'
    ]
    out  = model_instance.tao.cmds(cmds)
    model_instance.tao.cmd('run')
    bc2_engy_model = model_instance.tao.ele_gen_attribs(f"ENDBC2")["E_TOT"] / 1e9
    print(f'L2 Energy (meas, model): ({edes["L2"]}, {bc2_engy_model:0.4})')
    cmds = [
        'veto var *',
        'veto dat *',
        f'set dat L3.energy[2]|meas = {edes["L3"]}E9',
        'use dat L3.energy[2]',
        'use var linac_fudge[3]'
    ]
    out  = model_instance.tao.cmds(cmds)
    model_instance.tao.cmd('run')
    L3_engy_model = model_instance.tao.ele_gen_attribs(f"ENDL3")["E_TOT"] / 1e9
    print(f'L3 Energy (meas, model): ( {edes["L3"]}, {L3_engy_model})')
    [print(l) for l in model_instance.tao.cmd('show var linac_fudge')]

