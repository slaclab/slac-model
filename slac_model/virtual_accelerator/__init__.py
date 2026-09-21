import slac_model.virtual_accelerator.actions as va_actions


ELEMENT_ATTR_MAPPING= {
    "BPM":
        {
            "X": va_actions.BPMXVariable,
            "Y": va_actions.BPMYVariable,
            "TMIT": va_actions.BPMTMITDummyVariable,
        },
    "Quadrupole":
            {
                "BCTRL": va_actions.QuadrupoleBCTRLVariable,
                "BCTRL.DRVL": va_actions.BminVariable,
                "BCTRL.DRVH": va_actions.BmaxVariable,
                "BACT": va_actions.QuadrupoleBACTVariable,
                "BDES": va_actions.QuadrupoleBCTRLVariable,
                "BMAX": va_actions.BmaxVariable,
                "BMIN": va_actions.BminVariable,
                "STATCTRLSUB.T": va_actions.StatusVariable,
                "CTRL": va_actions.ControlStateVariable,
            },


    "Solenoid":
        {
            "BCTRL": va_actions.SolenoidBCTRLVariable,
            "BCTRL.DRVL": va_actions.BminVariable,
            "BCTRL.DRVH": va_actions.BmaxVariable,
            "BACT": va_actions.SolenoidBACTVariable,
            "BDES": va_actions.SolenoidBCTRLVariable,
            "BMIN": va_actions.BminVariable,
            "BMAX": va_actions.BmaxVariable,
            "STATCTRLSUB.T": va_actions.StatusVariable,
            "CTRL": va_actions.ControlStateVariable,
        },
"SBend": {
  "BCTRL": va_actions.SBendBCTRLVariable,
  "BCTRL.DRVL": va_actions.BminVariable,
  "BCTRL.DRVH": va_actions.BmaxVariable,
  "BACT": va_actions.SBendBACTVariable,
  "BDES": va_actions.SBendBCTRLVariable,
  "BMIN": va_actions.BminVariable,
  "BMAX": va_actions.BmaxVariable,
  "STATCTRLSUB.T": va_actions.StatusVariable,
  "CTRL": va_actions.ControlStateVariable
},
    "HorizontalCorrector":
        {
            "BCTRL": va_actions.KickerBCTRLVariable,
            "BCTRL.DRVL": va_actions.BminVariable,
            "BCTRL.DRVH": va_actions.BmaxVariable,
            "BACT": va_actions.KickerBACTVariable,
            "BDES": va_actions.KickerBCTRLVariable,
            "BMIN": va_actions.BminVariable,
            "BMAX": va_actions.BmaxVariable,
            "STATCTRLSUB.T": va_actions.StatusVariable,
            "CTRL": va_actions.ControlStateVariable,
        },

    "VerticalCorrector":
        {
            "BCTRL": va_actions.KickerBCTRLVariable,
            "BCTRL.DRVL": va_actions.BminVariable,
            "BCTRL.DRVH": va_actions.BmaxVariable,
            "BACT": va_actions.KickerBACTVariable,
            "BDES": va_actions.KickerBCTRLVariable,
            "BMIN": va_actions.BminVariable,
            "BMAX": va_actions.BmaxVariable,
            "STATCTRLSUB.T": va_actions.StatusVariable,
            "CTRL": va_actions.ControlStateVariable,
        },

    "Klystron": {
        "ENLD": va_actions.KlystronENLDVariable,
        "PACT": va_actions.KlystronPACTVariable,
        "PDES": va_actions.KlystronPDESVariable,
        "BEAMCODE1_STAT": va_actions.KlystronStatVariable,
        "BEAMCODE2_STAT": va_actions.KlystronStatVariable,

    },

    "Crab_Cavity": {
        "AREQ": va_actions.CavityAREQVariable,
        "ADES": va_actions.CavityAREQVariable,
        "AFBENB": va_actions.DummyEnumVariable,
        "AFBST": va_actions.DummyEnumVariable,
        "AMPL_W0CH0": va_actions.CavityAREQReadbackVariable,
        "MODECFG": va_actions.CavityMODECFGVariable,
        "PREQ": va_actions.CavityPREQVariable,
        "PACT_AVGNT": va_actions.CavityPREQReadbackVariable,
        "PDES": va_actions.CavityPREQVariable,
        "PFBENB": va_actions.DummyEnumVariable,
        "PFBST": va_actions.DummyEnumVariable,
        "RF_ENABLE": va_actions.DummyEnumVariable,
        "REFPOC": va_actions.CavityPREQVariable,
    },
}