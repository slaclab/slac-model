"""Validate the virtual accelerator factory against the CU_HXR lattice and slac-db."""

import os
from pathlib import Path
import re

import pytest

pytest.importorskip("pytao")
pytest.importorskip("lume_bmad")

from lume_bmad.model import LUMEBmadModel
from slac_db import device

from slac_model.models.cu import get_cu_hxr_bmad_model
from slac_model.virtual_accelerator import ACCESSOR_MAPPING, factory, actions


@pytest.fixture(scope="module")
def simulator():
    """Load the Cu helper's OTR2:END lattice slice once for integration tests.

    An unset lattice root skips these tests; a configured but invalid root fails
    so configuration errors are not mistaken for unavailable optional data.
    """
    lattice_root = os.environ.get("LCLS_LATTICE")
    if not lattice_root:
        pytest.skip("Set LCLS_LATTICE to run the CU_HXR integration tests")
    init_file = Path(lattice_root) / "bmad" / "models" / "cu_hxr" / "tao.init"
    assert init_file.is_file(), f"CU_HXR lattice initialization file not found: {init_file}"
    return get_cu_hxr_bmad_model(start_element="OTR2", end_element="END")


@pytest.fixture(scope="module")
def database_records(simulator):
    """Return database types and accessor-to-PV mappings for lattice devices."""
    # Normalize split elements and klystron segments independently of the
    # factory helper so its output is not used to construct its own reference.
    # Drop deferred elements (Tao type starts with "@") to match the factory,
    # which does not create variables for hardware that is not installed.
    indices = simulator.lat_list("*", "ele.ix_ele")
    lattice_names = set()
    for name, ix in zip(simulator.lat_list("*", "ele.name"), indices):
        if simulator.ele_head(int(ix))["type"].startswith("@"):
            continue
        lattice_names.add(re.sub(r"^(K\d+_\d+)[A-Z]$", r"\1", name.split("#")[0]))
    return {
        name: (device.get_attribute(name, "device_type"), device.get_all_accessors(name))
        for name in device.get_devices(include_inactive=True, beampath="CU_HXR")
        if name in lattice_names
    }


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [("QM01", "QUAD"), ("BPM10", "BPM"), ("XC08", "XCOR"), ("YC08", "YCOR")],
)
def test_get_device_type_matches_database(name, expected_type):
    """Check representative device types against the database's device table."""
    assert device.get_attribute(name, "device_type") == expected_type
    assert factory.get_device_type(name) == expected_type


def test_get_simulator_device_names(simulator, database_records):
    """Check normalized, unique device names and inclusion of database matches."""
    names = factory.get_simulator_device_names(simulator)

    assert names[0] == "OTR2"
    assert {"QM01", "BPM10", "XC08", "YC08", "K21_1"} <= set(names)
    assert len(names) == len(set(names))
    assert not {"BEGINNING", "END"} & set(names)
    assert all("#" not in name for name in names)
    assert set(database_records) <= set(names)


def test_get_model_and_variable_classes(simulator):
    """Verify that Tao selects the correct model class and variable classes."""
    from slac_model.virtual_accelerator.backend_utils import (
        get_model_class,
        get_variable_class,
    )
    
    assert get_model_class(simulator) is LUMEBmadModel
    
    # Test that we can get variable classes configured for the simulator.
    # get_variable_class returns a subclass with the conversion module bound per instance.
    QuadBCTRLSubclass = get_variable_class(simulator, "QuadrupoleBCTRLVariable")
    assert issubclass(QuadBCTRLSubclass, actions.QuadrupoleBCTRLVariable)
    
    # Verify that instances have the conversion module bound
    var = QuadBCTRLSubclass(name="Q1", element_name="Q1")
    assert var.get_conversion_module() is not None


def test_unsupported_simulator_raises_device_names():
    """Require device name getter to reject unknown backends."""
    with pytest.raises(ValueError, match="Unsupported simulator type"):
        factory.get_simulator_device_names(object())


def test_unsupported_simulator_raises_virtual_accelerator():
    """Require virtual accelerator factory to reject unknown backends."""
    with pytest.raises(ValueError, match="Unsupported simulator type"):
        factory.create_virtual_accelerator(object(), "CU_HXR")


@pytest.mark.parametrize(
    "class_name",
    sorted({name for mapping in ACCESSOR_MAPPING.values() for name in mapping.values()}),
)
def test_accessor_mapping_classes_exist(class_name):
    """Ensure every configured action class is defined in actions.py."""
    assert hasattr(actions, class_name), f"Missing action class: {class_name}"


def test_create_virtual_accelerator_matches_database(simulator, database_records, caplog):
    """Compare generated PV variables and readbacks with real CU_HXR records.

    Expected PV names come from slac-db, while supported accessors and action
    classes follow ACCESSOR_MAPPING. Also require warnings for mapped accessors
    absent from the database.
    """
    expected = {}
    covered_types = set()
    missing = {}
    for name, (device_type, accessors) in database_records.items():
        mapping = ACCESSOR_MAPPING.get(device_type, {})
        for accessor, class_name in mapping.items():
            if accessor in accessors:
                pv = accessors[accessor]
                assert pv, f"Empty PV for {name}.{accessor}"
                assert pv not in expected, f"Duplicate PV in expected variables: {pv}"
                expected[pv] = (name, getattr(actions, class_name))
                covered_types.add(device_type)
            else:
                missing.setdefault(name, set()).add(accessor)

    # Prevent an empty or incomplete database match from passing by comparing
    # two equally incomplete variable sets.
    assert {"QUAD", "BPM", "XCOR", "YCOR"} <= covered_types
    assert expected, "No supported CU_HXR PVs found in slac-db"

    accelerator = factory.create_virtual_accelerator(simulator, "CU_HXR")

    assert isinstance(accelerator, LUMEBmadModel)
    assert accelerator.simulator is simulator
    # LUMEBmadModel also registers model-level outputs and tracking controls;
    # compare only device-facing variables created by the factory.
    actual = {
        pv: variable
        for pv, variable in accelerator.supported_variables.items()
        if isinstance(variable, (actions.ScalarVariable, actions.EnumVariable))
    }
    assert actual.keys() == expected.keys()
    for pv, (element_name, variable_class) in expected.items():
        variable = actual[pv]
        assert variable.name == pv
        assert variable.element_name == element_name
        # get_variable_class returns backend-specific subclasses for instance isolation,
        # so check isinstance instead of exact type match
        assert isinstance(variable, variable_class)

    readings = accelerator.get(list(expected))
    assert readings.keys() == expected.keys()
    # The quadrupole readback and control aliases share the same simulated field.
    quad_accessors = database_records["QM01"][1]
    for accessor in ("bact", "bctrl", "bdes", "bcon"):
        pv = quad_accessors[accessor]
        assert actual[pv].read_only is (accessor == "bact")
        assert readings[pv] == pytest.approx(readings[quad_accessors["bctrl"]])
    # Tao reports orbit coordinates in meters; BPM PVs expose millimeters.
    bpm_accessors = database_records["BPM10"][1]
    orbit = simulator.ele("BPM10").orbit
    assert readings[bpm_accessors["x"]] == pytest.approx(orbit.x * 1e3)
    assert readings[bpm_accessors["y"]] == pytest.approx(orbit.y * 1e3)

    for name, accessors in missing.items():
        assert any(
            record.name == factory.__name__
            and name in record.getMessage()
            and all(accessor in record.getMessage() for accessor in accessors)
            for record in caplog.records
        ), f"Missing-accessor warning not emitted for {name}"


LCAV_EXPECTED_FOR_K21_1 = {
    "KLYS:LI21:11:ENLD":           actions.KlystronENLDVariable,
    "KLYS:LI21:11:BEAMCODE1_STAT": actions.KlystronStatVariable,
    "KLYS:LI21:11:BEAMCODE2_STAT": actions.KlystronStatVariable,
    "KLYS:LI21:11:PHAS":           actions.KlystronPACTVariable,
    "KLYS:LI21:11:PDES":           actions.KlystronPDESVariable,
    "ACCL:LI21:1:L1S_PDES_DS0":    actions.KlystronPDESVariable,
    "ACCL:LI21:1:L1S_PDES_DS1":    actions.KlystronPDESVariable,
}


def test_lcav_variables_registered_for_k21_1(simulator):
    """K21_1 (an L1 LCAV station) exposes all 7 mapped PVs with correct classes."""
    accelerator = factory.create_virtual_accelerator(simulator, "CU_HXR")

    for pv, expected_class in LCAV_EXPECTED_FOR_K21_1.items():
        assert pv in accelerator.supported_variables, (
            f"LCAV PV missing from supported_variables: {pv}"
        )
        variable = accelerator.supported_variables[pv]
        # get_variable_class returns backend-specific subclasses for instance isolation
        assert isinstance(variable, expected_class), (
            f"{pv}: got {type(variable).__name__}, expected {expected_class.__name__}"
        )
        assert variable.name == pv
        assert variable.element_name == "K21_1"