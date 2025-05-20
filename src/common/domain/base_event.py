"""
BaseEvent é a classe base para todos os eventos de domínio no sistema.

Eventos de domínio representam ocorrências importantes no sistema que podem
ser usadas para comunicação entre serviços e para manter trilhas de auditoria.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, ClassVar, Dict, Any
import uuid


@dataclass
class BaseEvent:
    """Base class for all domain events."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    aggregate_id: Optional[str] = None
    event_type: str = field(init=False)
    version: int = 1
    
    def __post_init__(self):
        """Set the event type based on the class name."""
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
            
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            timestamp=timestamp or datetime.utcnow(),
            aggregate_id=data.get('aggregate_id'),
            version=data.get('version', 1)
        )