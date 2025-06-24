# tests/common/domain/test_base_value_object.py
import pytest
from common.domain.base_value_object import BaseValueObject
from dataclasses import dataclass

@dataclass(frozen=True)
class TestValueObject(BaseValueObject):
    """A test value object class."""
    name: str
    value: int

class TestBaseValueObject:
    """Tests for the BaseValueObject class."""
    
    def test_init(self):
        """Test initialization."""
        vo = TestValueObject(name="Test", value=42)
        
        assert vo.name == "Test"
        assert vo.value == 42
    
    def test_equality(self):
        """Test equality comparison."""
        vo1 = TestValueObject(name="Test", value=42)
        vo2 = TestValueObject(name="Test", value=42)
        vo3 = TestValueObject(name="Different", value=42)
        
        assert vo1 == vo2
        assert vo1 != vo3
    
    def test_immutability(self):
        """Test that the value object is immutable."""
        vo = TestValueObject(name="Test", value=42)
        
        with pytest.raises(Exception):
            vo.name = "Changed"
    
    def test_to_dict(self):
        """Test the to_dict method."""
        vo = TestValueObject(name="Test", value=42)
        vo_dict = vo.to_dict()
        
        assert vo_dict == {'name': 'Test', 'value': 42}
    
    def test_from_dict(self):
        """Test the from_dict method."""
        data = {'name': 'Test', 'value': 42}
        vo = TestValueObject.from_dict(data)
        
        assert vo.name == "Test"
        assert vo.value == 42