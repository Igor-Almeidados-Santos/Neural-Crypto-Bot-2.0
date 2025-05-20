# tests/common/domain/test_base_entity.py
import pytest
from datetime import datetime, timezone
import uuid
from src.common.domain.base_entity import BaseEntity

class TestEntity(BaseEntity):
    """A test entity class."""
    def __init__(self, id=None, created_at=None, updated_at=None, name=None):
        super().__init__(id=id, created_at=created_at, updated_at=updated_at)
        self.name = name
        
    def to_dict(self):
        data = super().to_dict()
        data['name'] = self.name
        return data
    
    @classmethod
    def from_dict(cls, data):
        entity = super().from_dict(data)
        entity.name = data.get('name')
        return entity

class TestBaseEntity:
    """Tests for the BaseEntity class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        entity = TestEntity()
        
        assert entity.id is not None
        assert isinstance(entity.id, str)
        assert uuid.UUID(entity.id, version=4)  # Validate UUID format
        
        assert entity.created_at is not None
        assert isinstance(entity.created_at, datetime)
        
        assert entity.updated_at is not None
        assert isinstance(entity.updated_at, datetime)
        
        # created_at and updated_at should be the same initially
        assert entity.created_at == entity.updated_at
    
    def test_init_with_values(self):
        """Test initialization with provided values."""
        entity_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        name = "Test Entity"
        
        entity = TestEntity(
            id=entity_id,
            created_at=created_at,
            updated_at=updated_at,
            name=name
        )
        
        assert entity.id == entity_id
        assert entity.created_at == created_at
        assert entity.updated_at == updated_at
        assert entity.name == name
    
    def test_update(self):
        """Test the update method."""
        entity = TestEntity(name="Original Name")
        original_updated_at = entity.updated_at
        
        # Wait a moment to ensure the timestamp changes
        import time
        time.sleep(0.001)
        
        entity.update()
        
        assert entity.updated_at > original_updated_at
    
    def test_to_dict(self):
        """Test the to_dict method."""
        entity_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        name = "Test Entity"
        
        entity = TestEntity(
            id=entity_id,
            created_at=created_at,
            updated_at=updated_at,
            name=name
        )
        
        entity_dict = entity.to_dict()
        
        assert entity_dict['id'] == entity_id
        assert entity_dict['created_at'] == created_at.isoformat()
        assert entity_dict['updated_at'] == updated_at.isoformat()
        assert entity_dict['name'] == name
    
    def test_from_dict(self):
        """Test the from_dict method."""
        entity_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc)
        updated_at = datetime.now(timezone.utc)
        name = "Test Entity"
        
        data = {
            'id': entity_id,
            'created_at': created_at.isoformat(),
            'updated_at': updated_at.isoformat(),
            'name': name
        }
        
        entity = TestEntity.from_dict(data)
        
        assert entity.id == entity_id
        assert entity.created_at.isoformat() == created_at.isoformat()
        assert entity.updated_at.isoformat() == updated_at.isoformat()
        assert entity.name == name