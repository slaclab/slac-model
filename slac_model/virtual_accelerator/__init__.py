import slac_model.virtual_accelerator.bmad_actions as va_actions

# Mapping of device types and accessors to LUME variables
ACCESSOR_MAPPING = {
    "QUAD": {
        "bact": "QuadrupoleBACTVariable",
        "bctrl": "QuadrupoleBCTRLVariable",
        "bdes": "QuadrupoleBCTRLVariable",
        "bcon": "QuadrupoleBCTRLVariable",
        "bmax": "BmaxVariable",
        "bmin": "BminVariable",
        "ctrl": "ControlStateVariable"
    },
    "BPM": {
        "tmit": "BPMTMITDummyVariable",
        "x": "BPMXVariable",
        "y": "BPMYVariable"
    },
    "SOLE": {
        "bact": "SolenoidBACTVariable",
        "bctrl": "SolenoidBCTRLVariable",
        "bdes": "SolenoidBCTRLVariable",
        "bcon": "SolenoidBCTRLVariable",
        "bmax": "BmaxVariable",
        "bmin": "BminVariable",
        "ctrl": "ControlStateVariable"
    },
    "XCOR": {
        "bact": "HKickerBACTVariable",
        "bctrl": "HKickerBCTRLVariable",
        "bdes": "HKickerBCTRLVariable",
        "bcon": "HKickerBCTRLVariable",
        "bmax": "BmaxVariable",
        "bmin": "BminVariable",
        "ctrl": "ControlStateVariable"
    },
    "YCOR": {
        "bact": "VKickerBACTVariable",
        "bctrl": "VKickerBCTRLVariable",
        "bdes": "VKickerBCTRLVariable",
        "bcon": "VKickerBCTRLVariable",
        "bmax": "BmaxVariable",
        "bmin": "BminVariable",
        "ctrl": "ControlStateVariable"
    },
    "TCAV": {
        "amplitude_act_beamcode1": "CavityAREQReadbackVariable",
        "amplitude_act_beamcode2": "CavityAREQReadbackVariable",
        "amplitude_des_beamcode1": "CavityAREQVariable",
        "amplitude_des_beamcode2": "CavityAREQVariable",
        "amplitude_fbenb": "DummyEnumVariable",
        "amplitude_fbst": "DummyEnumVariable",
        "amplitude_wocho": "CavityAREQVariable",
        "phase_act_beamcode1": "CavityPREQReadbackVariable",
        "phase_act_beamcode2": "CavityPREQReadbackVariable",
        "phase_des_beamcode1": "CavityPREQVariable",
        "phase_des_beamcode2": "CavityPREQVariable",
        "phase_fbenb": "DummyEnumVariable",
        "phase_fbst": "DummyEnumVariable",
        "rf_enable_beamcode1": "CavityMODECFGVariable",
        "rf_enable_beamcode2": "CavityMODECFGVariable"
    }
}
