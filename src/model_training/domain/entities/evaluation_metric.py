"""
Evaluation metric entity for model performance assessment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


class MetricType(Enum):
    """Types of evaluation metrics."""
    # Regression metrics
    MSE = "mean_squared_error"
    RMSE = "root_mean_squared_error"
    MAE = "mean_absolute_error"
    MAPE = "mean_absolute_percentage_error"
    R2 = "r_squared"
    
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1_score"
    AUC = "area_under_curve"
    
    # Time series specific
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    
    # Trading specific
    PNL = "profit_and_loss"
    RETURN = "return"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    CALMAR_RATIO = "calmar_ratio"
    
    # Custom
    CUSTOM = "custom"


class MetricDirection(Enum):
    """Direction indicating whether higher or lower values are better."""
    HIGHER_BETTER = auto()
    LOWER_BETTER = auto()


@dataclass
class EvaluationMetric:
    """Domain entity representing a model evaluation metric."""
    
    # Core attributes
    name: str
    value: float
    type: MetricType
    direction: MetricDirection
    
    # Context
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None  # e.g., "train", "validation", "test"
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def is_better_than(self, other_metric: EvaluationMetric) -> bool:
        """
        Compare this metric with another to determine if this one is better.
        
        Args:
            other_metric: Another metric of the same type to compare with
            
        Returns:
            bool: True if this metric is better than the other one
            
        Raises:
            ValueError: If metrics are not comparable (different types)
        """
        if self.type != other_metric.type or self.name != other_metric.name:
            raise ValueError(f"Cannot compare metrics of different types: {self.type} vs {other_metric.type}")
        
        if self.direction == MetricDirection.HIGHER_BETTER:
            return self.value > other_metric.value
        else:
            return self.value < other_metric.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary representation."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.type.value,
            "direction": self.direction.name,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "timestamp": self.timestamp.isoformat(),
            "additional_info": self.additional_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvaluationMetric:
        """Create an EvaluationMetric instance from a dictionary."""
        # Parse timestamp
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow()
        
        return cls(
            name=data["name"],
            value=data["value"],
            type=MetricType(data["type"]),
            direction=MetricDirection[data["direction"]],
            dataset_name=data.get("dataset_name"),
            dataset_split=data.get("dataset_split"),
            timestamp=timestamp,
            additional_info=data.get("additional_info", {})
        )