"""
BaseEvent é a classe base para todos os eventos de domínio no sistema.

Eventos de domínio representam ocorrências importantes no sistema que podem
ser usadas para comunicação entre serviços e para manter trilhas de auditoria.
"""
# tests/common/domain/test_base_entity.py
import pytest
from datetime import datetime, timezone
import uuid
from src.common.domain.base_entity import BaseEntity

class TestEntity(BaseEntity):
    """A test entity class."""
    def __init__(self, id=None, created_at=None, updated_at=None, name=None):
        super().__init__(
            id=id or str(uuid.uuid4()),
            created_at=created_at or datetime.utcnow(),
            updated_at=updated_at or datetime.utcnow()
        )
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