"""
Model entity representing a trained machine learning model.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union
import uuid

from model_training.domain.entities.evaluation_metric import EvaluationMetric
from model_training.domain.entities.training_config import TrainingConfig


class ModelStatus(Enum):
    """Status of a model in its lifecycle."""
    TRAINING = auto()
    TRAINED = auto()
    EVALUATING = auto()
    EVALUATED = auto()
    DEPLOYED = auto()
    ARCHIVED = auto()
    FAILED = auto()


class ModelType(Enum):
    """Types of models supported by the system."""
    # Time series models
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    
    # Statistical models
    GARCH = "garch"
    HAWKES = "hawkes"
    
    # Graph models
    GNN = "gnn"
    
    # Reinforcement learning models
    DQN = "dqn"
    PPO = "ppo"
    
    # Ensemble models
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"


@dataclass
class Model:
    """Domain entity representing a machine learning model."""
    
    # Core attributes
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str
    type: ModelType
    status: ModelStatus = ModelStatus.TRAINING
    
    # Configuration and artifacts
    training_config: TrainingConfig
    hyperparameters: Dict[str, Any]
    model_artifacts_path: Optional[str] = None
    
    # Performance and evaluation
    metrics: List[EvaluationMetric] = field(default_factory=list)
    performance_summary: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    
    # Additional information
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_status(self, status: ModelStatus) -> None:
        """Update the model status and related timestamps."""
        previous_status = self.status
        self.status = status
        self.updated_at = datetime.utcnow()
        
        # Update specific timestamps based on status transition
        if status == ModelStatus.TRAINED and previous_status == ModelStatus.TRAINING:
            self.trained_at = datetime.utcnow()
        elif status == ModelStatus.EVALUATED and previous_status == ModelStatus.EVALUATING:
            self.evaluated_at = datetime.utcnow()
        elif status == ModelStatus.DEPLOYED:
            self.deployed_at = datetime.utcnow()
    
    def add_metric(self, metric: EvaluationMetric) -> None:
        """Add an evaluation metric to the model."""
        self.metrics.append(metric)
        self.performance_summary[metric.name] = metric.value
        self.updated_at = datetime.utcnow()
    
    def get_primary_metric(self) -> Optional[float]:
        """Get the primary evaluation metric value."""
        if self.training_config.primary_metric in self.performance_summary:
            return self.performance_summary[self.training_config.primary_metric]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "type": self.type.value,
            "status": self.status.name,
            "training_config": self.training_config.to_dict(),
            "hyperparameters": self.hyperparameters,
            "model_artifacts_path": self.model_artifacts_path,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "performance_summary": self.performance_summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "description": self.description,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Model:
        """Create a Model instance from a dictionary."""
        # Handle nested objects
        training_config = TrainingConfig.from_dict(data["training_config"])
        metrics = [EvaluationMetric.from_dict(m) for m in data.get("metrics", [])]
        
        # Parse datetime fields
        created_at = datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.utcnow()
        updated_at = datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.utcnow()
        trained_at = datetime.fromisoformat(data["trained_at"]) if data.get("trained_at") else None
        evaluated_at = datetime.fromisoformat(data["evaluated_at"]) if data.get("evaluated_at") else None
        deployed_at = datetime.fromisoformat(data["deployed_at"]) if data.get("deployed_at") else None
        
        # Create and return model instance
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data["name"],
            version=data["version"],
            type=ModelType(data["type"]),
            status=ModelStatus[data["status"]] if "status" in data else ModelStatus.TRAINING,
            training_config=training_config,
            hyperparameters=data.get("hyperparameters", {}),
            model_artifacts_path=data.get("model_artifacts_path"),
            metrics=metrics,
            performance_summary=data.get("performance_summary", {}),
            created_at=created_at,
            updated_at=updated_at,
            trained_at=trained_at,
            evaluated_at=evaluated_at,
            deployed_at=deployed_at,
            description=data.get("description"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )