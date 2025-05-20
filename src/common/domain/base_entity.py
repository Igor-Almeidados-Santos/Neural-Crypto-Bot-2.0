"""
BaseEntity é a classe base para todas as entidades de domínio no sistema.

Fornece funcionalidades comuns como id, timestamps e métodos para conversão.
"""
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, ClassVar, Dict, Any
import uuid


@dataclass
class BaseEntity(ABC):
    """Base class for all domain entities."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update(self):
        """Update the entity's updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary."""
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEntity':
        """Create entity from dictionary."""
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data.get('created_at'))
            
        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data.get('updated_at'))
            
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow()
        )