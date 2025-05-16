#!/bin/bash
# setup_base_domain.sh

echo "=== Configurando arquivos de domínio base para o Trading Bot ==="

# Garante que o diretório existe
mkdir -p src/common/domain

# Cria base_entity.py
echo "Criando src/common/domain/base_entity.py..."
cat > src/common/domain/base_entity.py << EOF
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
        return cls(
            id=data.get('id'),
            created_at=datetime.fromisoformat(data.get('created_at')) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data.get('updated_at')) if data.get('updated_at') else None
        )
EOF

# Cria base_value_object.py
echo "Criando src/common/domain/base_value_object.py..."
cat > src/common/domain/base_value_object.py << EOF
from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any, TypeVar

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
EOF

# Cria base_event.py
echo "Criando src/common/domain/base_event.py..."
cat > src/common/domain/base_event.py << EOF
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
EOF

echo "✅ Arquivos de domínio base criados com sucesso!"
