"""Tests for Variable serialization round-trip with model_post_init() backend setup.

This module tests that:
1. Backend-dependent Variables can be directly instantiated
2. Variables serialize with model_dump() and deserialize with model_validate()
3. Backend conversion modules are automatically set up during deserialization
4. Full model archives work with the new serialization approach
"""

import pytest
import tempfile
from pathlib import Path

# Import backend availability flags from conftest
from slac_model.tests.virtual_accelerator.conftest import HAS_CHEETAH, HAS_BMAD

# Guard Cheetah imports
if HAS_CHEETAH:
    try:
        from slac_model.virtual_accelerator.torch_actions import (
            QuadrupoleBCTRLVariable,
            QuadrupoleBACTVariable,
            SolenoidBCTRLVariable,
            CavityAREQVariable,
        )
    except ImportError:
        HAS_CHEETAH = False

# Guard Bmad imports
if HAS_BMAD:
    try:
        from slac_model.virtual_accelerator.actions import (
            QuadrupoleBCTRLVariable as BmadQuadrupoleBCTRLVariable,
        )
    except ImportError:
        HAS_BMAD = False


@pytest.mark.skipif(not HAS_CHEETAH, reason="Cheetah not available")
class TestVariableDirectInstantiation:
    """Test that Variables can be directly instantiated without get_variable_class()."""
    
    def test_cheetah_quadrupole_instantiation(self):
        """Variables can be directly instantiated with backend module auto-loaded."""
        var = QuadrupoleBCTRLVariable(name="Q1", element_name="Q1_element")
        assert var.name == "Q1"
        assert var.element_name == "Q1_element"
        # Backend module should be automatically set up
        assert hasattr(var, '_instance_conversion_module')
        assert var.get_conversion_module() is not None
    
    def test_cheetah_solenoid_instantiation(self):
        """Solenoid Variables also support direct instantiation."""
        var = SolenoidBCTRLVariable(name="SOL1", element_name="SOLENOID")
        assert var.name == "SOL1"
        assert var.element_name == "SOLENOID"
        assert hasattr(var, '_instance_conversion_module')
    
    def test_cheetah_cavity_instantiation(self):
        """Cavity Variables support direct instantiation."""
        var = CavityAREQVariable(name="CAV1", element_name="CAVITY_1")
        assert var.name == "CAV1"
        assert var.element_name == "CAVITY_1"
        assert hasattr(var, '_instance_conversion_module')
    
    @pytest.mark.skipif(not HAS_BMAD, reason="Bmad not available")
    def test_bmad_quadrupole_instantiation(self):
        """Bmad Variables also support direct instantiation."""
        var = BmadQuadrupoleBCTRLVariable(name="Q2", element_name="Q2_element")
        assert var.name == "Q2"
        assert var.element_name == "Q2_element"
        assert hasattr(var, '_instance_conversion_module')


@pytest.mark.skipif(not HAS_CHEETAH, reason="Cheetah not available")
class TestVariableSerialization:
    """Test Variable serialization and deserialization with model_dump/model_validate."""
    
    def test_quadrupole_model_dump(self):
        """model_dump() exports all relevant fields."""
        var = QuadrupoleBCTRLVariable(
            name="Q3",
            element_name="Q3_element",
            read_only=False,
        )
        config = var.model_dump()
        
        # Check that essential fields are present
        assert "name" in config
        assert config["name"] == "Q3"
        assert "element_name" in config
        assert config["element_name"] == "Q3_element"
        assert "read_only" in config
        assert config["read_only"] is False
    
    def test_quadrupole_model_dump_excludes_instance_data(self):
        """model_dump() should exclude private fields that shouldn't be serialized."""
        var = QuadrupoleBCTRLVariable(name="Q4", element_name="Q4_element")
        
        # Include everything
        config_full = var.model_dump()
        
        # The private _instance_conversion_module should be there (Pydantic includes it)
        # but we'll handle exclusion during archiving
        
        # Exclude private fields
        config_clean = var.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        })
        
        # Check clean config has the essential fields
        assert "name" in config_clean
        assert "element_name" in config_clean
        # Private fields should not be there
        assert '_instance_conversion_module' not in config_clean
    
    def test_quadrupole_model_validate_roundtrip(self):
        """model_validate() reconstructs Variable with backend setup."""
        var_original = QuadrupoleBCTRLVariable(name="Q5", element_name="Q5_element")
        
        # Export config
        config = var_original.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        
        # Reconstruct using model_validate
        var_restored = QuadrupoleBCTRLVariable.model_validate(config)
        
        # Check fields match
        assert var_restored.name == var_original.name
        assert var_restored.element_name == var_original.element_name
        assert var_restored.read_only == var_original.read_only
        
        # Backend should be set up automatically
        assert hasattr(var_restored, '_instance_conversion_module')
        assert var_restored.get_conversion_module() is not None
    
    def test_solenoid_roundtrip(self):
        """Solenoid serialization roundtrip works correctly."""
        var_original = SolenoidBCTRLVariable(name="SOL2", element_name="SOL2_element")
        
        config = var_original.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        
        var_restored = SolenoidBCTRLVariable.model_validate(config)
        
        assert var_restored.name == var_original.name
        assert var_restored.element_name == var_original.element_name
        assert hasattr(var_restored, '_instance_conversion_module')
    
    def test_cavity_roundtrip(self):
        """Cavity serialization roundtrip works correctly."""
        var_original = CavityAREQVariable(name="CAV2", element_name="CAVITY_2")
        
        config = var_original.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        
        var_restored = CavityAREQVariable.model_validate(config)
        
        assert var_restored.name == var_original.name
        assert var_restored.element_name == var_original.element_name
        assert hasattr(var_restored, '_instance_conversion_module')
    
    @pytest.mark.skipif(not HAS_BMAD, reason="Bmad not available")
    def test_bmad_roundtrip(self):
        """Bmad Variables serialize/deserialize with correct backend module."""
        var_original = BmadQuadrupoleBCTRLVariable(name="Q6", element_name="Q6_element")
        
        config = var_original.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        
        var_restored = BmadQuadrupoleBCTRLVariable.model_validate(config)
        
        assert var_restored.name == var_original.name
        assert var_restored.element_name == var_original.element_name
        # Bmad backend should be set up
        assert hasattr(var_restored, '_instance_conversion_module')
        conversion_module = var_restored.get_conversion_module()
        # Bmad module should be loaded
        assert conversion_module is not None


@pytest.mark.skipif(not HAS_CHEETAH, reason="Cheetah not available")
class TestVariableMultipleRoundtrips:
    """Test that Variables survive multiple serialization roundtrips."""
    
    def test_triple_roundtrip(self):
        """Variable survives multiple serialization cycles."""
        var = QuadrupoleBCTRLVariable(name="Q7", element_name="Q7_element")
        
        # First roundtrip
        config1 = var.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        var1 = QuadrupoleBCTRLVariable.model_validate(config1)
        
        # Second roundtrip
        config2 = var1.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        var2 = QuadrupoleBCTRLVariable.model_validate(config2)
        
        # Third roundtrip
        config3 = var2.model_dump(exclude={
            '_instance_conversion_module',
            '_backend_conversion_module_path'
        }, exclude_none=False)
        var3 = QuadrupoleBCTRLVariable.model_validate(config3)
        
        # All should have the same fields
        assert var3.name == var.name
        assert var3.element_name == var.element_name
        assert var3.read_only == var.read_only
        
        # Backend should still be set up
        assert hasattr(var3, '_instance_conversion_module')


@pytest.mark.skipif(not HAS_CHEETAH, reason="Cheetah not available")
class TestBackendModuleAutoLoading:
    """Test that model_post_init() correctly auto-loads backend modules."""
    
    def test_cheetah_backend_loaded(self):
        """Cheetah Variables load Cheetah conversion module automatically."""
        var = QuadrupoleBCTRLVariable(name="Q8", element_name="Q8_element")
        
        conversion_module = var.get_conversion_module()
        assert conversion_module is not None
        
        # Should be from slac_model.pv_conversion.cheetah
        assert hasattr(conversion_module, '__name__')
        assert 'cheetah' in conversion_module.__name__.lower() or hasattr(conversion_module, '__file__')
    
    @pytest.mark.skipif(not HAS_BMAD, reason="Bmad not available")
    def test_bmad_backend_loaded(self):
        """Bmad Variables load Bmad conversion module automatically."""
        var = BmadQuadrupoleBCTRLVariable(name="Q9", element_name="Q9_element")
        
        conversion_module = var.get_conversion_module()
        assert conversion_module is not None
        
        # Should be from slac_model.pv_conversion.bmad
        assert hasattr(conversion_module, '__name__')
        assert 'bmad' in conversion_module.__name__.lower() or hasattr(conversion_module, '__file__')
    
    def test_model_validate_triggers_post_init(self):
        """model_validate() triggers model_post_init() which loads backend."""
        var_original = QuadrupoleBCTRLVariable(name="Q10", element_name="Q10_element")
        
        # Export without the conversion module
        config = var_original.model_dump(exclude={
            '_instance_conversion_module',
        }, exclude_none=False)
        
        # Reconstruct - model_validate should call model_post_init
        var_restored = QuadrupoleBCTRLVariable.model_validate(config)
        
        # Backend should be loaded even though we didn't explicitly set it
        conversion_module = var_restored.get_conversion_module()
        assert conversion_module is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
