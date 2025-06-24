"""
FeatureSet Entity Module

Defines the FeatureSet entity which represents a collection of related features.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

from feature_engineering.domain.entities.feature import Feature


class FeatureSetType(Enum):
    """Types of feature sets based on their intended use."""
    TRAINING = auto()
    INFERENCE = auto()
    BACKTESTING = auto()
    VALIDATION = auto()
    PRODUCTION = auto()
    RESEARCH = auto()


class FeatureSetStatus(Enum):
    """Status of the feature set in its lifecycle."""
    DRAFT = auto()
    ACTIVE = auto()
    DEPRECATED = auto()
    ARCHIVED = auto()


@dataclass(frozen=True)
class FeatureSetMetadata:
    """Metadata for feature sets."""
    description: str
    version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    author: Optional[str] = None


@dataclass(frozen=True)
class FeatureSet:
    """
    A collection of related features that are processed together.
    
    FeatureSets represent logical groupings of features used for specific
    purposes in the trading system.
    """
    id: UUID
    name: str
    type: FeatureSetType
    status: FeatureSetStatus
    features: List[Feature]
    symbols: Set[str]
    metadata: FeatureSetMetadata
    start_timestamp: datetime
    end_timestamp: datetime
    
    @classmethod
    def create(cls,
               name: str,
               type: FeatureSetType,
               features: List[Feature],
               metadata: FeatureSetMetadata,
               status: FeatureSetStatus = FeatureSetStatus.ACTIVE,
               start_timestamp: Optional[datetime] = None,
               end_timestamp: Optional[datetime] = None,
               id: Optional[UUID] = None) -> 'FeatureSet':
        """Factory method to create a new FeatureSet instance."""
        # Extract unique symbols from features
        symbols = {feature.symbol for feature in features}
        
        # Determine timestamps from features if not provided
        if features and not start_timestamp:
            start_timestamp = min(feature.timestamp for feature in features)
        if features and not end_timestamp:
            end_timestamp = max(feature.timestamp for feature in features)
        
        # Use current time as default if no features or timestamps provided
        now = datetime.utcnow()
        if not start_timestamp:
            start_timestamp = now
        if not end_timestamp:
            end_timestamp = now
            
        return cls(
            id=id or uuid4(),
            name=name,
            type=type,
            status=status,
            features=features,
            symbols=symbols,
            metadata=metadata,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
    
    def add_feature(self, feature: Feature) -> 'FeatureSet':
        """Create a new FeatureSet with an additional feature."""
        new_features = list(self.features)
        new_features.append(feature)
        
        new_symbols = set(self.symbols)
        new_symbols.add(feature.symbol)
        
        # Update timestamps if needed
        start_timestamp = min(self.start_timestamp, feature.timestamp)
        end_timestamp = max(self.end_timestamp, feature.timestamp)
        
        # Create updated metadata with new timestamp
        updated_metadata = FeatureSetMetadata(
            description=self.metadata.description,
            version=self.metadata.version,
            created_at=self.metadata.created_at,
            updated_at=datetime.utcnow(),
            tags=self.metadata.tags,
            properties=self.metadata.properties,
            author=self.metadata.author
        )
        
        return FeatureSet(
            id=self.id,
            name=self.name,
            type=self.type,
            status=self.status,
            features=new_features,
            symbols=new_symbols,
            metadata=updated_metadata,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
    
    def remove_feature(self, feature_id: UUID) -> 'FeatureSet':
        """Create a new FeatureSet with a feature removed."""
        new_features = [f for f in self.features if f.id != feature_id]
        
        # Recalculate symbols set
        new_symbols = {f.symbol for f in new_features}
        
        # Recalculate timestamps if needed
        start_timestamp = min((f.timestamp for f in new_features), default=self.start_timestamp)
        end_timestamp = max((f.timestamp for f in new_features), default=self.end_timestamp)
        
        # Create updated metadata
        updated_metadata = FeatureSetMetadata(
            description=self.metadata.description,
            version=self.metadata.version,
            created_at=self.metadata.created_at,
            updated_at=datetime.utcnow(),
            tags=self.metadata.tags,
            properties=self.metadata.properties,
            author=self.metadata.author
        )
        
        return FeatureSet(
            id=self.id,
            name=self.name,
            type=self.type,
            status=self.status,
            features=new_features,
            symbols=new_symbols,
            metadata=updated_metadata,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp
        )
    
    def update_status(self, new_status: FeatureSetStatus) -> 'FeatureSet':
        """Create a new FeatureSet with updated status."""
        updated_metadata = FeatureSetMetadata(
            description=self.metadata.description,
            version=self.metadata.version,
            created_at=self.metadata.created_at,
            updated_at=datetime.utcnow(),
            tags=self.metadata.tags,
            properties=self.metadata.properties,
            author=self.metadata.author
        )
        
        return FeatureSet(
            id=self.id,
            name=self.name,
            type=self.type,
            status=new_status,
            features=self.features,
            symbols=self.symbols,
            metadata=updated_metadata,
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert FeatureSet to dictionary representation for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.type.name,
            "status": self.status.name,
            "features": [feature.to_dict() for feature in self.features],
            "symbols": list(self.symbols),
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "metadata": {
                "description": self.metadata.description,
                "version": self.metadata.version,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                "tags": self.metadata.tags,
                "properties": self.metadata.properties,
                "author": self.metadata.author
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSet':
        """Create FeatureSet instance from dictionary representation."""
        from feature_engineering.domain.entities.feature import Feature
        
        metadata = FeatureSetMetadata(
            description=data["metadata"]["description"],
            version=data["metadata"]["version"],
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            updated_at=datetime.fromisoformat(data["metadata"]["updated_at"]) 
                if data["metadata"]["updated_at"] else None,
            tags=data["metadata"]["tags"],
            properties=data["metadata"]["properties"],
            author=data["metadata"]["author"]
        )
        
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            type=FeatureSetType[data["type"]],
            status=FeatureSetStatus[data["status"]],
            features=[Feature.from_dict(f) for f in data["features"]],
            symbols=set(data["symbols"]),
            metadata=metadata,
            start_timestamp=datetime.fromisoformat(data["start_timestamp"]),
            end_timestamp=datetime.fromisoformat(data["end_timestamp"])
        )