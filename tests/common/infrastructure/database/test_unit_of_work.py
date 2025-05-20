# tests/common/infrastructure/database/test_unit_of_work.py
import pytest
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime, timezone
import uuid
from src.common.domain.base_entity import BaseEntity
from src.common.infrastructure.database.unit_of_work import UnitOfWork

class TestEntity(BaseEntity):
    """A test entity class."""
    def __init__(self, id=None, created_at=None, updated_at=None, name=None, value=None):
        super().__init__(id, created_at, updated_at)
        self.name = name
        self.value = value

class TestRepository:
    """A test repository class."""
    
    def __init__(self):
        self.entities = {}
        self.committed = False
        self.rolled_back = False
    
    async def save(self, entity):
        """Save an entity."""
        self.entities[entity.id] = entity
        return entity
    
    async def find_by_id(self, entity_id):
        """Find an entity by ID."""
        return self.entities.get(entity_id)

class TestUnitOfWork(UnitOfWork):
    """A test unit of work implementation."""
    
    def __init__(self):
        super().__init__()
        self.test_repository = TestRepository()
        self.repositories['test'] = self.test_repository
        self.begun = False
        self.committed = False
        self.rolled_back = False
    
    async def begin(self):
        """Begin a transaction."""
        self.begun = True
    
    async def commit(self):
        """Commit the transaction."""
        self.committed = True
        self.test_repository.committed = True
    
    async def rollback(self):
        """Rollback the transaction."""
        self.rolled_back = True
        self.test_repository.rolled_back = True
    
    def register_repository(self, name, repository):
        """Register a repository with this unit of work."""
        self.repositories[name] = repository

class TestUnitOfWorkClass:
    """Tests for the UnitOfWork class."""
    
    @pytest.fixture
    def unit_of_work(self):
        """Create a test unit of work."""
        return TestUnitOfWork()
    
    @pytest.mark.asyncio
    async def test_context_manager_success(self, unit_of_work):
        """Test the context manager with successful execution."""
        async with unit_of_work:
            # Perform some operations
            entity = TestEntity(name="Test", value=42)
            await unit_of_work.test_repository.save(entity)
        
        assert unit_of_work.begun is True
        assert unit_of_work.committed is True
        assert unit_of_work.rolled_back is False
        assert unit_of_work.test_repository.committed is True
        assert unit_of_work.test_repository.rolled_back is False
    
    @pytest.mark.asyncio
    async def test_context_manager_error(self, unit_of_work):
        """Test the context manager with an error."""
        with pytest.raises(ValueError):
            async with unit_of_work:
                # Perform some operations
                entity = TestEntity(name="Test", value=42)
                await unit_of_work.test_repository.save(entity)
                
                # Raise an error
                raise ValueError("Test error")
        
        assert unit_of_work.begun is True
        assert unit_of_work.committed is False
        assert unit_of_work.rolled_back is True
        assert unit_of_work.test_repository.committed is False
        assert unit_of_work.test_repository.rolled_back is True
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction_success(self, unit_of_work):
        """Test executing a function in a transaction with success."""
        async def test_function():
            entity = TestEntity(name="Test", value=42)
            await unit_of_work.test_repository.save(entity)
            return entity
        
        entity = await unit_of_work.execute_in_transaction(test_function)
        
        assert entity.name == "Test"
        assert entity.value == 42
        assert unit_of_work.begun is True
        assert unit_of_work.committed is True
        assert unit_of_work.rolled_back is False
    
    @pytest.mark.asyncio
    async def test_execute_in_transaction_error(self, unit_of_work):
        """Test executing a function in a transaction with an error."""
        async def test_function():
            entity = TestEntity(name="Test", value=42)
            await unit_of_work.test_repository.save(entity)
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await unit_of_work.execute_in_transaction(test_function)
        
        assert unit_of_work.begun is True
        assert unit_of_work.committed is False
        assert unit_of_work.rolled_back is True