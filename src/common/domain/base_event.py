"""
BaseEvent é a classe base para todos os eventos de domínio no sistema.

Eventos de domínio representam ocorrências importantes no sistema que podem
ser usadas para comunicação entre serviços e para manter trilhas de auditoria.
"""
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional
import uuid


@dataclass
class BaseEvent(ABC):
    """Base class for all domain events."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_type: str = field(init=False)
    aggregate_id: Optional[str] = None
    version: int = 1  # Adicionado campo version

    def __post_init__(self):
        self.event_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'aggregate_id': self.aggregate_id,
            'version': self.version
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Create event from dictionary."""
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data.get('timestamp'))
            
        # Obtém os parâmetros básicos
        params = {
            'id': data.get('id'),
            'aggregate_id': data.get('aggregate_id'),
        }
        
        # Adiciona timestamp se disponível
        if timestamp:
            params['timestamp'] = timestamp
            
        # Adiciona version se disponível
        if 'version' in data:
            params['version'] = data.get('version')
            
        # Cria o evento
        return cls(**params)