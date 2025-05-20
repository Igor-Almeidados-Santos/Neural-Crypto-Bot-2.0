# src/common/infrastructure/database/base_repository.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Dict, Any, Type
import uuid
from datetime import datetime

from ...domain.base_entity import BaseEntity

T = TypeVar('T', bound=BaseEntity)

class BaseRepository(Generic[T], ABC):
    """Base class for all repositories.
    
    Repositories mediate between the domain and data mapping layers.
    They provide methods to retrieve domain objects by ID or other criteria,
    and to persist domain objects.
    
    Attributes:
        None
    """
    
    @abstractmethod
    async def find_by_id(self, entity_id: str) -> Optional[T]:
        """Find an entity by its ID.
        
        Args:
            entity_id: The ID of the entity to find.
            
        Returns:
            The entity with the given ID, or None if not found.
        """
        pass
    
    @abstractmethod
    async def find_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Find all entities.
        
        Args:
            limit: The maximum number of entities to return.
            offset: The number of entities to skip.
            
        Returns:
            A list of entities.
        """
        pass
    
    @abstractmethod
    async def find_by_criteria(self, criteria: Dict[str, Any], limit: int = 100, offset: int = 0) -> List[T]:
        """Find entities matching the given criteria.
        
        Args:
            criteria: The criteria to match.
            limit: The maximum number of entities to return.
            offset: The number of entities to skip.
            
        Returns:
            A list of entities matching the criteria.
        """
        pass
    
    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save an entity.
        
        Args:
            entity: The entity to save.
            
        Returns:
            The saved entity.
        """
        pass
    
    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """Delete an entity.
        
        Args:
            entity_id: The ID of the entity to delete.
            
        Returns:
            True if the entity was deleted, False otherwise.
        """
        pass
    
    @abstractmethod
    async def count(self, criteria: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching the given criteria.
        
        Args:
            criteria: The criteria to match.
            
        Returns:
            The number of entities matching the criteria.
        """
        pass
    
    def _prepare_entity_for_save(self, entity: T) -> T:
        """Prepare an entity for saving.
        
        This method ensures that the entity has an ID and timestamps.
        
        Args:
            entity: The entity to prepare.
            
        Returns:
            The prepared entity.
        """
        # Ensure entity has an ID
        if not entity.id:
            entity.id = str(uuid.uuid4())
            
        # Set timestamps
        now = datetime.utcnow()
        
        if not entity.created_at:
            entity.created_at = now
            
        entity.updated_at = now
        
        return entity