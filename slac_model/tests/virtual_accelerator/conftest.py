"""Shared pytest configuration and fixtures for virtual accelerator tests."""

import os
from pathlib import Path

import pytest

# Try to import Bmad backend (optional)
try:
    from lume_bmad.model import LUMEBmadModel
    from slac_model.models.cu import get_cu_hxr_bmad_model, get_cu_hxr_bmad_va
    HAS_BMAD = True
except ImportError:
    HAS_BMAD = False

# Try to import Cheetah backend (optional)
try:
    from lume_cheetah.model import LUMECheetahModel
    from slac_model.models.cu import get_cu_hxr_cheetah_model, get_cu_hxr_cheetah_va
    HAS_CHEETAH = True
except ImportError:
    HAS_CHEETAH = False


@pytest.fixture(scope="module")
def bmad_simulator():
    """Load the Cu helper's OTR2:END lattice slice.

    An unset lattice root skips these tests; a configured but invalid root fails
    so configuration errors are not mistaken for unavailable optional data.
    """
    if not HAS_BMAD:
        pytest.skip("Bmad not available")
    
    lattice_root = os.environ.get("LCLS_LATTICE")
    if not lattice_root:
        pytest.skip("Set LCLS_LATTICE to run the CU_HXR integration tests")
    init_file = Path(lattice_root) / "bmad" / "models" / "cu_hxr" / "tao.init"
    assert init_file.is_file(), f"CU_HXR lattice initialization file not found: {init_file}"
    return get_cu_hxr_bmad_model(start_element="OTR2", end_element="END")


@pytest.fixture(scope="module")
def cheetah_simulator():
    """Load the Cheetah Cu_HXR model.
    
    Skip if Cheetah is not available.
    """
    if not HAS_CHEETAH:
        pytest.skip("Cheetah not available")
    
    lattice_root = os.environ.get("LCLS_LATTICE")
    if not lattice_root:
        pytest.skip("Set LCLS_LATTICE to run the CU_HXR integration tests")
    
    return get_cu_hxr_cheetah_model(start_element="otr2", end_element="end")


@pytest.fixture(scope="module")
def cheetah_va():
    """Load the Cheetah virtual accelerator for CU_HXR.
    
    Shared across multiple tests to avoid expensive VA instantiation.
    This fixture is module-scoped to reuse the same VA instance across tests
    that need to verify VA behavior (e.g., checking variable types, PV mappings).
    
    Skip if Cheetah is not available.
    """
    if not HAS_CHEETAH:
        pytest.skip("Cheetah not available")
    
    lattice_root = os.environ.get("LCLS_LATTICE")
    if not lattice_root:
        pytest.skip("Set LCLS_LATTICE to run the CU_HXR integration tests")
    
    return get_cu_hxr_cheetah_va()


@pytest.fixture(scope="module")
def bmad_va():
    """Load the Bmad virtual accelerator for CU_HXR.
    
    Shared across multiple tests to avoid expensive VA instantiation.
    This fixture is module-scoped to reuse the same VA instance across tests
    that need to verify VA behavior (e.g., checking variable types, PV mappings,
    LCAV device support).
    
    """
    return get_cu_hxr_bmad_va()
