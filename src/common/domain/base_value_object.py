"""
BaseValueObject é a classe base para todos os objetos de valor no sistema.

Objetos de valor são imutáveis e não possuem identidade própria.
"""
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, TypeVar

T = TypeVar('T', bound='BaseValueObject')

@dataclass(frozen=True)
class BaseValueObject(ABC):
    """Base class for all value objects."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert value object to dictionary."""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> T:
        """Create value object from dictionary."""
        return cls(**data)
