"""Test the virtual accelerator factory against the Cheetah backend."""

import re

import pytest

from slac_db import device

pytest.importorskip("lume_cheetah")
pytest.importorskip("lume_torch")

from slac_model.virtual_accelerator import ACCESSOR_MAPPING, factory
from slac_model.virtual_accelerator import torch_actions as actions

# Skip entire module if Cheetah is not available
from slac_model.tests.virtual_accelerator.conftest import HAS_CHEETAH
pytestmark = pytest.mark.skipif(not HAS_CHEETAH, reason="Cheetah not available")

try:
    from lume_cheetah.model import LUMECheetahModel
except ImportError:
    pass

@pytest.fixture(scope="module")
def simulator(cheetah_va):
    """Use the accelerator's simulator so PV expectations match its lattice."""
    return cheetah_va.simulator


@pytest.fixture(scope="module")
def database_records(simulator):
    """Return database types and accessor-to-PV mappings for lattice devices.
    
    Only includes device types that are supported by Cheetah backend. LCAV devices
    (linac cavities/klystrons) are excluded as Cheetah doesn't have control functions
    for them yet (get_klystron_* functions are not implemented).
    """
    lattice_names = {
        re.sub(r"^(K\d+_\d+)[A-Z]$", r"\1", name.split("#")[0])
        for name in factory.get_simulator_device_names(simulator)
    }
    
    # Supported device types for Cheetah backend
    supported_types = {k for k in ACCESSOR_MAPPING.keys() if k != "LCAV"}
    
    return {
        name: (device_type, device.get_all_accessors(name))
        for name in device.get_devices(include_inactive=True, beampath="CU_HXR")
        if name in lattice_names
        and (device_type := device.get_attribute(name, "device_type")) in supported_types
    }


def test_get_model_and_variable_classes(simulator):
    """Verify that Cheetah selects the correct model class and variable classes."""
    from slac_model.virtual_accelerator.backend_utils import get_model_class
    
    assert get_model_class(simulator) is LUMECheetahModel
    
    # Test that we can instantiate static variable classes directly.
    # The static classes automatically load their conversion module via model_post_init().
    QuadBCTRLVariable = actions.QuadrupoleBCTRLVariable
    
    # Verify that instances have the conversion module bound
    var = QuadBCTRLVariable(name="Q1", element_name="Q1")
    assert var.get_conversion_module() is not None


@pytest.mark.parametrize(
    "class_name",
    sorted({name for mapping in ACCESSOR_MAPPING.values() for name in mapping.values()}),
)
def test_accessor_mapping_classes_exist(class_name):
    """Ensure every configured action class is defined in actions.py."""
    assert hasattr(actions, class_name), f"Missing action class: {class_name}"


def test_cheetah_device_names(simulator):
    """Verify that Cheetah simulator returns expected device names."""
    from slac_model.virtual_accelerator.backend_utils import get_simulator_device_names
    
    names = get_simulator_device_names(simulator)
    
    # Verify we got device names from Cheetah
    assert len(names) > 0, "No devices found in Cheetah simulator"
    
    # Names should be strings
    assert all(isinstance(name, str) for name in names)


def test_dummy_enum_variable_retains_instance_state(simulator):
    # Instantiate static variable classes directly
    variable = actions.DummyEnumVariable(name="TCAV:TEST:FB", element_name="TCAV0")
    other = actions.DummyEnumVariable(name="TCAV:TEST:OTHER_FB", element_name="TCAV0")

    assert variable._get(simulator) == "0"
    variable._set(simulator, "1")
    assert variable._get(simulator) == "1"
    assert other._get(simulator) == "0"


def test_create_virtual_accelerator_matches_database(simulator, database_records, cheetah_va, caplog):
    """Compare generated PV variables with Cheetah-supported CU_HXR records.

    Expected PV names come from slac-db (excluding klystrons), while supported
    accessors and action classes follow ACCESSOR_MAPPING. Cavity mode controls
    are excluded because Cheetah has no setter. Accessor aliases may share a PV
    only when they refer to the same element and variable class.
    """
    expected = {}
    covered_types = set()
    missing = {}
    for name, (device_type, accessors) in database_records.items():
        mapping = ACCESSOR_MAPPING.get(device_type, {})
        for accessor, class_name in mapping.items():
            if class_name == "CavityMODECFGVariable":
                continue
            if accessor in accessors:
                pv = accessors[accessor]
                assert pv, f"Empty PV for {name}.{accessor}"
                expected_variable = (name, getattr(actions, class_name))
                if pv in expected:
                    assert expected[pv] == expected_variable, (
                        f"Conflicting accessor aliases for PV: {pv}"
                    )
                expected[pv] = expected_variable
                covered_types.add(device_type)
            else:
                missing.setdefault(name, set()).add(accessor)

    # Prevent an empty or incomplete database match from passing
    assert {"QUAD", "BPM", "XCOR", "YCOR", "SOLE", "TCAV"} <= covered_types
    assert expected, "No supported CU_HXR PVs found in slac-db for Cheetah"

    accelerator = cheetah_va

    assert isinstance(accelerator, LUMECheetahModel)
    actual = {
        pv: variable
        for pv, variable in accelerator.supported_variables.items()
        if isinstance(variable, (actions.ScalarVariable, actions.EnumVariable))
    }
    assert actual.keys() == expected.keys(), (
        f"Expected PVs mismatch:\n"
        f"  Missing: {expected.keys() - actual.keys()}\n"
        f"  Extra: {actual.keys() - expected.keys()}"
    )
    for pv, (element_name, variable_class) in expected.items():
        variable = actual[pv]
        assert variable.name == pv
        assert variable.element_name == element_name
        assert isinstance(variable, variable_class)

    readings = accelerator.get(list(expected))
    assert readings.keys() == expected.keys()


def test_no_klystron_variables_in_cheetah(simulator, cheetah_va):
    """Verify that klystron PVs are not created in Cheetah backend."""
    accelerator = cheetah_va
    
    klystron_pvs = [
        "KLYS:LI21:11:ENLD",
        "KLYS:LI21:11:BEAMCODE1_STAT",
        "KLYS:LI21:11:BEAMCODE2_STAT",
        "KLYS:LI21:11:PHAS",
        "KLYS:LI21:11:PDES",
    ]
    
    for pv in klystron_pvs:
        assert pv not in accelerator.supported_variables, (
            f"Klystron PV {pv} should not be in Cheetah backend"
        )


def test_cheetah_variables_return_torch_tensors(simulator, cheetah_va):
    """Verify that Cheetah variables return torch.Tensor, not floats.
    
    This is a key difference between Cheetah (torch) and Bmad (numpy) backends.
    TorchScalarVariable should ensure that get() operations return torch tensors,
    and set() operations accept torch tensors or floats.
    """
    import torch
    
    accelerator = cheetah_va
    
    # Get some scalar variables from the accelerator
    scalar_vars = {
        pv: var
        for pv, var in accelerator.supported_variables.items()
        if isinstance(var, actions.ScalarVariable)
    }
    
    assert len(scalar_vars) > 0, "No scalar variables found in Cheetah accelerator"
    
    # Test the first few scalar variables
    test_pvs = list(scalar_vars.keys())[:3]
    
    for pv in test_pvs:
        var = scalar_vars[pv]
        
        # Test get() returns a torch.Tensor
        try:
            value = var._get(simulator)
            assert isinstance(
                value, torch.Tensor
            ), f"Variable {pv} returned {type(value).__name__} instead of torch.Tensor"
            assert value.dtype in [
                torch.float32,
                torch.float64,
            ], f"Variable {pv} has dtype {value.dtype}, expected float32 or float64"
        except Exception as e:
            # Some variables might fail due to missing conversion functions
            # This is expected for unimplemented variables
            if "not found in conversion module" not in str(e):
                raise


def test_cheetah_torchscalarvariable_class_used(simulator):
    """Verify that Cheetah backend uses TorchScalarVariable, not ScalarVariable."""
    from lume_torch.variables import TorchScalarVariable
    from lume.variables import ScalarVariable
    
    # Instantiate a static quadrupole control variable from torch_actions
    var = actions.QuadrupoleBCTRLVariable(name="Q1", element_name="Q1")
    
    # Should be instance of TorchScalarVariable, not the regular ScalarVariable
    assert isinstance(var, TorchScalarVariable), (
        f"Cheetah variable should be TorchScalarVariable, got {type(var).__name__}"
    )
    
    # But should also have our AgnosticSimulatorMixin
    from slac_model.virtual_accelerator._actions_base import AgnosticSimulatorMixin
    assert isinstance(var, AgnosticSimulatorMixin), (
        "Cheetah variable should have AgnosticSimulatorMixin for conversion module binding"
    )
