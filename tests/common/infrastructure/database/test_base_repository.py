# tests/common/infrastructure/database/test_base_repository.py
# tests/common/infrastructure/database/test_base_repository.py
import pytest
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timezone
import uuid
from src.common.domain.base_entity import BaseEntity
from src.common.infrastructure.database.base_repository import BaseRepository

class TestEntity(BaseEntity):
    """A test entity class."""
    def __init__(self, id=None, created_at=None, updated_at=None, name=None, value=None):
        super().__init__(id, created_at, updated_at)
        self.name = name
        self.value = value

class TestRepository(BaseRepository[TestEntity]):
    """A test repository implementation."""
    
    def __init__(self):
        self.entities: Dict[str, TestEntity] = {}
    
    async def find_by_id(self, entity_id: str) -> Optional[TestEntity]:
        """Find an entity by ID."""
        return self.entities.get(entity_id)
    
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[TestEntity]:
        """Find all entities."""
        entities = list(self.entities.values())
        return entities[offset:offset+limit]
    
    async def find_by_criteria(self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[TestEntity]:
        """Find entities by criteria."""
        result = []
        
        for entity in self.entities.values():
            match = True
            
            for key, value in criteria.items():
                if not hasattr(entity, key) or getattr(entity, key) != value:
                    match = False
                    break
            
            if match:
                result.append(entity)
                
                if len(result) >= limit:
                    break
        
        return result[offset:offset+limit]
    
    async def save(self, entity: TestEntity) -> TestEntity:
        """Save an entity."""
        entity = self._prepare_entity_for_save(entity)
        self.entities[entity.id] = entity
        return entity
    
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False
    
    async def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count entities by criteria."""
        if criteria is None:
            return len(self.entities)
        
        count = 0
        
        for entity in self.entities.values():
            match = True
            
            for key, value in criteria.items():
                if not hasattr(entity, key) or getattr(entity, key) != value:
                    match = False
                    break
            
            if match:
                count += 1
        
        return count

class TestBaseRepository:
    """Tests for the BaseRepository class."""
    
    @pytest.fixture
    def repository(self):
        """Create a test repository."""
        return TestRepository()
    
    async def _create_populated_repository(self, repository):
        """Helper method to populate a repository."""
        for i in range(10):
            entity = TestEntity(
                name=f"Entity {i}",
                value=i
            )
            await repository.save(entity)
        
        return repository
    
    @pytest.mark.asyncio
    async def test_find_by_id(self, repository):
        """Test finding an entity by ID."""
        # Populate the repository
        populated_repo = await self._create_populated_repository(repository)
        
        # Get all entities
        entities = await populated_repo.find_all()
        entity_id = entities[0].id
        
        # Find the entity by ID
        found_entity = await populated_repo.find_by_id(entity_id)
        
        assert found_entity is not None
        assert found_entity.id == entity_id
    
    @pytest.mark.asyncio
    async def test_find_all(self, repository):
        """Test finding all entities."""
        # Populate the repository
        populated_repo = await self._create_populated_repository(repository)
        
        entities = await populated_repo.find_all()
        
        assert len(entities) == 10
        
        # Test with limit
        limited_entities = await populated_repo.find_all(limit=5)
        assert len(limited_entities) == 5
        
        # Test with offset
        offset_entities = await populated_repo.find_all(offset=5)
        assert len(offset_entities) == 5
        
        # Test with limit and offset
        paginated_entities = await populated_repo.find_all(limit=3, offset=5)
        assert len(paginated_entities) == 3
    
    @pytest.mark.asyncio
    async def test_find_by_criteria(self, repository):
        """Test finding entities by criteria."""
        # Populate the repository
        populated_repo = await self._create_populated_repository(repository)
        
        # Find entities with even values
        even_entities = await populated_repo.find_by_criteria({"value": 2})
        
        assert len(even_entities) == 1
        assert even_entities[0].value == 2
        
        # Find entities with a specific name
        named_entities = await populated_repo.find_by_criteria({"name": "Entity 5"})
        
        assert len(named_entities) == 1
        assert named_entities[0].name == "Entity 5"
        assert named_entities[0].value == 5
    
    @pytest.mark.asyncio
    async def test_save(self, repository):
        """Test saving an entity."""
        entity = TestEntity(name="New Entity", value=42)
        
        # Save the entity
        saved_entity = await repository.save(entity)
        
        assert saved_entity.id is not None
        assert saved_entity.created_at is not None
        assert saved_entity.updated_at is not None
        assert saved_entity.name == "New Entity"
        assert saved_entity.value == 42
        
        # Find the entity by ID
        found_entity = await repository.find_by_id(saved_entity.id)
        
        assert found_entity is not None
        assert found_entity.id == saved_entity.id
    
    @pytest.mark.asyncio
    async def test_update(self, repository):
        """Test updating an entity."""
        # Create an entity
        entity = TestEntity(name="Original Name", value=1)
        saved_entity = await repository.save(entity)
        
        # Update the entity
        saved_entity.name = "Updated Name"
        saved_entity.value = 2
        updated_entity = await repository.save(saved_entity)
        
        assert updated_entity.id == saved_entity.id
        assert updated_entity.name == "Updated Name"
        assert updated_entity.value == 2
        
        # Find the entity by ID
        found_entity = await repository.find_by_id(updated_entity.id)
        
        assert found_entity is not None
        assert found_entity.name == "Updated Name"
        assert found_entity.value == 2
    
    @pytest.mark.asyncio
    async def test_delete(self, repository):
        """Test deleting an entity."""
        # Populate the repository
        populated_repo = await self._create_populated_repository(repository)
        
        # Get all entities
        entities = await populated_repo.find_all()
        entity_id = entities[0].id
        
        # Delete the entity
        result = await populated_repo.delete(entity_id)
        
        assert result is True
        
        # Try to find the deleted entity
        found_entity = await populated_repo.find_by_id(entity_id)
        
        assert found_entity is None
        
        # Try to delete a nonexistent entity
        result = await populated_repo.delete("nonexistent_id")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_count(self, repository):
        """Test counting entities."""
        # Populate the repository
        populated_repo = await self._create_populated_repository(repository)
        
        # Count all entities
        count = await populated_repo.count()
        
        assert count == 10
        
        # Count entities with even values
        even_count = await populated_repo.count({"value": 2})
        
        assert even_count == 1
        
        # Count entities with a specific name
        named_count = await populated_repo.count({"name": "Entity 5"})
        
        assert named_count == 1