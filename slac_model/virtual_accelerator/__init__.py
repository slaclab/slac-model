import slac_model.virtual_accelerator.actions as va_actions

# Mapping of device types and accessors to LUME variables
ACCESSOR_MAPPING = {
    "QUAD": {
        "bact": "QuadrupoleBACTVariable",
        "bctrl": "QuadrupoleBCTRLVariable",
        "bdes": "QuadrupoleBCTRLVariable",
        "bcon": "QuadrupoleBCTRLVariable",
        "bmax": "BMAXVariable",
        "bmin": "BMINVariable",
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
        "bmax": "BMAXVariable",
        "bmin": "BMINVariable",
        "ctrl": "ControlStateVariable"
    },
    "XCOR": {
        "bact": "HKickerBACTVariable",
        "bctrl": "HKickerBCTRLVariable",
        "bdes": "HKickerBCTRLVariable",
        "bcon": "HKickerBCTRLVariable",
        "bmax": "BMAXVariable",
        "bmin": "BMINVariable",
        "ctrl": "ControlStateVariable"
    },
    "VCOR": {
        "bact": "VKickerBACTVariable",
        "bctrl": "VKickerBCTRLVariable",
        "bdes": "VKickerBCTRLVariable",
        "bcon": "VKickerBCTRLVariable",
        "bmax": "BMAXVariable",
        "bmin": "BMINVariable",
        "ctrl": "ControlStateVariable"
    },
}
