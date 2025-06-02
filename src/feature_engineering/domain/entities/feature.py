"""
Feature Entity Module

Defines the core Feature entity representing a calculated feature in the trading system.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4


class FeatureType(Enum):
    """Enumeration of available feature types in the system."""
    TECHNICAL = auto()
    STATISTICAL = auto()
    SENTIMENT = auto()
    ORDERBOOK = auto()
    ONCHAIN = auto()
    MARKET = auto()
    CUSTOM = auto()


class FeatureCategory(Enum):
    """Categorization of features by their analytical purpose."""
    TREND = auto()
    MOMENTUM = auto()
    VOLATILITY = auto()
    VOLUME = auto()
    LIQUIDITY = auto()
    SENTIMENT = auto()
    CORRELATION = auto()
    PATTERN = auto()
    MARKET_REGIME = auto()
    FUNDAMENTAL = auto()
    ANOMALY = auto()
    OTHER = auto()


class FeatureScope(Enum):
    """Scope defining the context in which the feature operates."""
    SINGLE_ASSET = auto()
    MULTI_ASSET = auto()
    MARKET_WIDE = auto()
    EXCHANGE_SPECIFIC = auto()
    CROSS_EXCHANGE = auto()


class FeatureTimeframe(Enum):
    """Standard timeframes for feature calculation."""
    TICK = auto()
    SECOND_1 = auto()
    SECOND_5 = auto()
    SECOND_15 = auto()
    SECOND_30 = auto()
    MINUTE_1 = auto()
    MINUTE_3 = auto()
    MINUTE_5 = auto()
    MINUTE_15 = auto()
    MINUTE_30 = auto()
    HOUR_1 = auto()
    HOUR_2 = auto()
    HOUR_4 = auto()
    HOUR_6 = auto()
    HOUR_8 = auto()
    HOUR_12 = auto()
    DAY_1 = auto()
    DAY_3 = auto()
    WEEK_1 = auto()
    MONTH_1 = auto()


@dataclass(frozen=True)
class FeatureMetadata:
    """Metadata associated with a feature."""
    description: str
    category: FeatureCategory
    scope: FeatureScope
    timeframe: FeatureTimeframe
    is_experimental: bool = False
    created_at: datetime = datetime.utcnow()
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    tags: List[str] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        # Initialize default values for mutable fields
        if self.tags is None:
            object.__setattr__(self, 'tags', [])
        if self.properties is None:
            object.__setattr__(self, 'properties', {})


@dataclass(frozen=True)
class Feature:
    """
    Core domain entity representing a calculable feature for the trading system.
    
    A Feature is an immutable value object that contains all the information necessary
    to identify, calculate, and utilize a specific trading signal or indicator.
    """
    id: UUID
    name: str
    symbol: str
    type: FeatureType
    value: Union[float, int, bool, str]
    timestamp: datetime
    metadata: FeatureMetadata
    lookback_periods: int = 0
    dependencies: List[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        # Initialize default values for mutable fields
        if self.dependencies is None:
            object.__setattr__(self, 'dependencies', [])
    
    @classmethod
    def create(cls, 
               name: str,
               symbol: str,
               type: FeatureType,
               value: Union[float, int, bool, str],
               metadata: FeatureMetadata,
               lookback_periods: int = 0,
               dependencies: List[str] = None,
               confidence: Optional[float] = None,
               timestamp: Optional[datetime] = None,
               id: Optional[UUID] = None) -> 'Feature':
        """Factory method to create a new Feature instance."""
        return cls(
            id=id or uuid4(),
            name=name,
            symbol=symbol,
            type=type,
            value=value,
            timestamp=timestamp or datetime.utcnow(),
            metadata=metadata,
            lookback_periods=lookback_periods,
            dependencies=dependencies,
            confidence=confidence
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Feature to dictionary representation for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "symbol": self.symbol,
            "type": self.type.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "lookback_periods": self.lookback_periods,
            "dependencies": self.dependencies,
            "confidence": self.confidence,
            "metadata": {
                "description": self.metadata.description,
                "category": self.metadata.category.name,
                "scope": self.metadata.scope.name,
                "timeframe": self.metadata.timeframe.name,
                "is_experimental": self.metadata.is_experimental,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat() if self.metadata.updated_at else None,
                "version": self.metadata.version,
                "tags": self.metadata.tags,
                "properties": self.metadata.properties
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feature':
        """Create Feature instance from dictionary representation."""
        metadata = FeatureMetadata(
            description=data["metadata"]["description"],
            category=FeatureCategory[data["metadata"]["category"]],
            scope=FeatureScope[data["metadata"]["scope"]],
            timeframe=FeatureTimeframe[data["metadata"]["timeframe"]],
            is_experimental=data["metadata"]["is_experimental"],
            created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
            updated_at=datetime.fromisoformat(data["metadata"]["updated_at"]) 
                if data["metadata"]["updated_at"] else None,
            version=data["metadata"]["version"],
            tags=data["metadata"]["tags"],
            properties=data["metadata"]["properties"]
        )
        
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            symbol=data["symbol"],
            type=FeatureType[data["type"]],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            lookback_periods=data["lookback_periods"],
            dependencies=data["dependencies"],
            confidence=data["confidence"],
            metadata=metadata
        )