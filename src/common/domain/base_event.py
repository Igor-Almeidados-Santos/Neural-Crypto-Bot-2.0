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

    def __post_init__(self):
        self.event_type = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'aggregate_id': self.aggregate_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvent':
        """Create event from dictionary."""
        event = cls(
            id=data.get('id'),
            aggregate_id=data.get('aggregate_id')
        )
        event.timestamp = datetime.fromisoformat(data.get('timestamp')) if data.get('timestamp') else event.timestamp
        return event
