"""
BaseAggregateRoot é a classe base para todas as raízes de agregação no sistema.

Raízes de agregação são entidades que encapsulam outras entidades e objetos de valor,
mantendo a consistência do agregado como um todo.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

from src.common.domain.base_entity import BaseEntity
from src.common.domain.base_event import BaseEvent


@dataclass
class BaseAggregateRoot(BaseEntity):
    """Base class for all aggregate roots."""
    _events: List[BaseEvent] = field(default_factory=list, init=False, repr=False)
    _version: int = field(default=0, init=False)

    def add_event(self, event: BaseEvent) -> None:
        """Add a domain event to the aggregate's list of pending events."""
        self._events.append(event)

    def clear_events(self) -> List[BaseEvent]:
        """Clear all pending events and return them."""
        events = self._events.copy()
        self._events.clear()
        return events

    def get_uncommitted_events(self) -> List[BaseEvent]:
        """Get all uncommitted events."""
        return self._events.copy()
    
    def apply_event(self, event: BaseEvent) -> None:
        """Apply an event to the aggregate."""
        method_name = f"apply_{event.event_type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            method(event)
        
        self._version += 1
        self.update()  # Update the updated_at timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert aggregate root to dictionary including version."""
        base_dict = super().to_dict()
        base_dict['_version'] = self._version
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseAggregateRoot':
        """Create aggregate root from dictionary."""
        aggregate = super().from_dict(data)
        if isinstance(aggregate, BaseAggregateRoot):
            aggregate._version = data.get('_version', 0)
        return aggregate
    
    @property
    def version(self) -> int:
        """Get the current version of the aggregate."""
        return self._version