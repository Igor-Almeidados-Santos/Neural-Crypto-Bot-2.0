"""
Training configuration entity for model training.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Union
import json


class DatasetSplit(Enum):
    """Types of dataset splits."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TrainingMode(Enum):
    """Training modes for models."""
    FULL = "full"  # Train on all available data
    INCREMENTAL = "incremental"  # Update existing model with new data
    TRANSFER = "transfer"  # Transfer learning from existing model
    ENSEMBLE = "ensemble"  # Train as part of an ensemble


@dataclass
class TrainingConfig:
    """Domain entity representing configuration for model training."""
    
    # Core training parameters
    dataset_id: str
    features: List[str]
    target: str
    train_split_ratio: float = 0.7
    validation_split_ratio: float = 0.15
    test_split_ratio: float = 0.15
    
    # Time series specific
    sequence_length: Optional[int] = None
    forecast_horizon: Optional[int] = None
    stride: int = 1
    
    # Training process
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "mse"
    use_gpu: bool = True
    distributed_training: bool = False
    mixed_precision: bool = False
    
    # Data preprocessing
    normalization: str = "standard"  # "standard", "minmax", "robust", "none"
    handle_missing_data: str = "interpolate"  # "drop", "interpolate", "fill_mean", "fill_zero"
    feature_engineering_pipeline: Optional[str] = None
    data_augmentation: bool = False
    
    # Cross-validation
    cross_validation_folds: int = 0  # 0 means no cross-validation
    cross_validation_strategy: str = "time_series_split"  # "time_series_split", "purged_kfold", "standard_kfold"
    
    # Evaluation
    primary_metric: str = "mse"
    evaluation_metrics: List[str] = field(default_factory=lambda: ["mse", "mae", "r2"])
    
    # Training mode
    mode: TrainingMode = TrainingMode.FULL
    base_model_id: Optional[str] = None  # For transfer learning or incremental training
    
    # Asset-specific settings
    assets: List[str] = field(default_factory=list)
    timeframe: str = "1h"  # Timeframe for the data: 1m, 5m, 15m, 1h, 4h, 1d
    
    # Advanced options
    random_seed: int = 42
    experiment_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert training config to dictionary representation."""
        result = {
            "dataset_id": self.dataset_id,
            "features": self.features,
            "target": self.target,
            "train_split_ratio": self.train_split_ratio,
            "validation_split_ratio": self.validation_split_ratio,
            "test_split_ratio": self.test_split_ratio,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "loss_function": self.loss_function,
            "use_gpu": self.use_gpu,
            "distributed_training": self.distributed_training,
            "mixed_precision": self.mixed_precision,
            "normalization": self.normalization,
            "handle_missing_data": self.handle_missing_data,
            "data_augmentation": self.data_augmentation,
            "cross_validation_folds": self.cross_validation_folds,
            "cross_validation_strategy": self.cross_validation_strategy,
            "primary_metric": self.primary_metric,
            "evaluation_metrics": self.evaluation_metrics,
            "mode": self.mode.value,
            "random_seed": self.random_seed,
            "assets": self.assets,
            "timeframe": self.timeframe,
            "tags": self.tags,
            "additional_params": self.additional_params
        }
        
        # Add optional fields if they are set
        if self.sequence_length is not None:
            result["sequence_length"] = self.sequence_length
        
        if self.forecast_horizon is not None:
            result["forecast_horizon"] = self.forecast_horizon
            
        if self.feature_engineering_pipeline is not None:
            result["feature_engineering_pipeline"] = self.feature_engineering_pipeline
            
        if self.base_model_id is not None:
            result["base_model_id"] = self.base_model_id
            
        if self.experiment_name is not None:
            result["experiment_name"] = self.experiment_name
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingConfig:
        """Create a TrainingConfig instance from a dictionary."""
        # Extract fields with defaults
        mode = TrainingMode(data.get("mode", "full"))
        
        # Create instance with provided values
        return cls(
            dataset_id=data["dataset_id"],
            features=data["features"],
            target=data["target"],
            train_split_ratio=data.get("train_split_ratio", 0.7),
            validation_split_ratio=data.get("validation_split_ratio", 0.15),
            test_split_ratio=data.get("test_split_ratio", 0.15),
            sequence_length=data.get("sequence_length"),
            forecast_horizon=data.get("forecast_horizon"),
            stride=data.get("stride", 1),
            batch_size=data.get("batch_size", 64),
            epochs=data.get("epochs", 100),
            early_stopping_patience=data.get("early_stopping_patience", 10),
            learning_rate=data.get("learning_rate", 0.001),
            optimizer=data.get("optimizer", "adam"),
            loss_function=data.get("loss_function", "mse"),
            use_gpu=data.get("use_gpu", True),
            distributed_training=data.get("distributed_training", False),
            mixed_precision=data.get("mixed_precision", False),
            normalization=data.get("normalization", "standard"),
            handle_missing_data=data.get("handle_missing_data", "interpolate"),
            feature_engineering_pipeline=data.get("feature_engineering_pipeline"),
            data_augmentation=data.get("data_augmentation", False),
            cross_validation_folds=data.get("cross_validation_folds", 0),
            cross_validation_strategy=data.get("cross_validation_strategy", "time_series_split"),
            primary_metric=data.get("primary_metric", "mse"),
            evaluation_metrics=data.get("evaluation_metrics", ["mse", "mae", "r2"]),
            mode=mode,
            base_model_id=data.get("base_model_id"),
            assets=data.get("assets", []),
            timeframe=data.get("timeframe", "1h"),
            random_seed=data.get("random_seed", 42),
            experiment_name=data.get("experiment_name"),
            tags=data.get("tags", []),
            additional_params=data.get("additional_params", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> TrainingConfig:
        """Create a TrainingConfig instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def to_json(self) -> str:
        """Convert training config to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate the training configuration.
        
        Returns:
            List[str]: List of validation errors, empty if valid
        """
        errors = []
        
        # Validate split ratios
        total_split = self.train_split_ratio + self.validation_split_ratio + self.test_split_ratio
        if not (0.99 <= total_split <= 1.01):  # Allow small floating point errors
            errors.append(f"Split ratios must sum to 1.0, got {total_split}")
        
        # Validate features and target
        if not self.features:
            errors.append("Features list cannot be empty")
        
        if not self.target:
            errors.append("Target must be specified")
            
        if self.target in self.features:
            errors.append(f"Target '{self.target}' should not be included in features")
        
        # Validate time series parameters
        if self.sequence_length is not None and self.sequence_length <= 0:
            errors.append(f"Sequence length must be positive, got {self.sequence_length}")
            
        if self.forecast_horizon is not None and self.forecast_horizon <= 0:
            errors.append(f"Forecast horizon must be positive, got {self.forecast_horizon}")
            
        # Validate training parameters
        if self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.batch_size}")
            
        if self.epochs <= 0:
            errors.append(f"Number of epochs must be positive, got {self.epochs}")
            
        if self.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {self.learning_rate}")
        
        # Validate transfer learning mode
        if self.mode == TrainingMode.TRANSFER and not self.base_model_id:
            errors.append("Base model ID is required for transfer learning mode")
            
        # Validate incremental learning mode
        if self.mode == TrainingMode.INCREMENTAL and not self.base_model_id:
            errors.append("Base model ID is required for incremental learning mode")
        
        return errors