"""
Use case for training machine learning models.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

from src.model_training.domain.entities.model import Model, ModelStatus, ModelType
from src.model_training.domain.entities.training_config import TrainingConfig
from src.model_training.application.services.hyperparameter_optimization_service import HyperparameterOptimizationService
from src.model_training.application.services.model_registry_service import ModelRegistryService
from src.model_training.infrastructure.experiment_tracker import ExperimentTracker
from src.model_training.infrastructure.model_repository import ModelRepository

# Import model implementations
from src.model_training.models.time_series.lstm_model import LSTMTimeSeriesForecaster
from src.model_training.models.time_series.gru_model import GRUTimeSeriesForecaster
from src.model_training.models.time_series.transformer_model import TransformerTimeSeriesForecaster
from src.model_training.models.statistical.garch_model import GARCHModel
from src.model_training.models.statistical.hawkes_process import HawkesProcess


logger = logging.getLogger(__name__)


class TrainModelUseCase:
    """Use case for training machine learning models."""
    
    def __init__(
        self,
        model_repository: ModelRepository,
        model_registry_service: ModelRegistryService,
        hyperparameter_optimization_service: HyperparameterOptimizationService,
        experiment_tracker: ExperimentTracker
    ):
        """
        Initialize TrainModelUseCase.
        
        Args:
            model_repository: Repository for model storage
            model_registry_service: Service for model registration
            hyperparameter_optimization_service: Service for hyperparameter optimization
            experiment_tracker: Tracker for experiment logging
        """
        self.model_repository = model_repository
        self.model_registry_service = model_registry_service
        self.hyperparameter_optimization_service = hyperparameter_optimization_service
        self.experiment_tracker = experiment_tracker
    
    def execute(
        self,
        training_config: TrainingConfig,
        data: pd.DataFrame,
        optimize_hyperparameters: bool = True,
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        register_model: bool = True
    ) -> Model:
        """
        Execute the use case to train a model.
        
        Args:
            training_config: Configuration for model training
            data: Input data for training
            optimize_hyperparameters: Whether to optimize hyperparameters
            n_trials: Number of hyperparameter optimization trials
            timeout: Timeout for hyperparameter optimization in seconds
            register_model: Whether to register the model in the registry
            
        Returns:
            Trained model entity
        """
        # Validate training configuration
        validation_errors = training_config.validate()
        if validation_errors:
            error_msg = f"Invalid training configuration: {', '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create model entity
        model_type = self._get_model_type(training_config)
        model_name = training_config.experiment_name or f"{model_type.value}_model"
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model = Model(
            name=model_name,
            version=model_version,
            type=model_type,
            status=ModelStatus.TRAINING,
            training_config=training_config,
            hyperparameters={},
            description=f"Auto-trained {model_type.value} model",
            tags=training_config.tags + ["auto-trained"]
        )
        
        # Log experiment start
        experiment_id = self.experiment_tracker.start_experiment(
            name=model_name,
            tags=model.tags,
            params=training_config.to_dict()
        )
        
        try:
            # Preprocess data
            preprocessed_data = self._preprocess_data(data, training_config)
            
            # Create model trainer
            model_trainer = self._create_model_trainer(model_type, training_config)
            
            # Optimize hyperparameters if requested
            if optimize_hyperparameters:
                logger.info(f"Starting hyperparameter optimization with {n_trials} trials...")
                best_hyperparameters = self._optimize_hyperparameters(
                    model_type=model_type,
                    training_config=training_config,
                    data=preprocessed_data,
                    n_trials=n_trials,
                    timeout=timeout
                )
                model.hyperparameters = best_hyperparameters
                self.experiment_tracker.log_params(best_hyperparameters)
            else:
                # Use default hyperparameters for the model type
                model.hyperparameters = self._get_default_hyperparameters(model_type)
            
            # Train model
            logger.info(f"Training {model_type.value} model...")
            metrics = self._train_model(
                model_trainer=model_trainer,
                data=preprocessed_data,
                hyperparameters=model.hyperparameters
            )
            
            # Update model status and metrics
            model.update_status(ModelStatus.TRAINED)
            for metric_name, metric_value in metrics.items():
                self.experiment_tracker.log_metric(metric_name, metric_value)
            
            # Save model artifacts
            artifacts_path = self.model_repository.save_model(
                model_id=model.id,
                model_trainer=model_trainer
            )
            model.model_artifacts_path = artifacts_path
            
            # Register model if requested
            if register_model:
                logger.info(f"Registering model {model.name} version {model.version}...")
                self.model_registry_service.register_model(model)
            
            # Log experiment end
            self.experiment_tracker.end_experiment(status="success")
            
            return model
            
        except Exception as e:
            # Handle failure
            model.update_status(ModelStatus.FAILED)
            model.metadata["error"] = str(e)
            
            # Log failure
            self.experiment_tracker.log_param("error", str(e))
            self.experiment_tracker.end_experiment(status="failed")
            
            logger.error(f"Model training failed: {str(e)}", exc_info=True)
            
            # Re-raise exception
            raise
    
    def _get_model_type(self, training_config: TrainingConfig) -> ModelType:
        """
        Determine model type from training configuration.
        
        Args:
            training_config: Training configuration
            
        Returns:
            ModelType enum value
        """
        # Extract model type from additional params if specified
        model_type_str = training_config.additional_params.get("model_type", "").lower()
        
        if model_type_str:
            try:
                return ModelType(model_type_str)
            except ValueError:
                logger.warning(f"Unknown model type: {model_type_str}, using default based on configuration")
        
        # Infer model type from configuration
        # Time series models are the default for cryptocurrency price prediction
        if training_config.sequence_length is not None:
            # Choose between different time series models based on configuration
            if "transformer" in training_config.tags:
                return ModelType.TRANSFORMER
            elif "gru" in training_config.tags:
                return ModelType.GRU
            else:
                # Default to LSTM
                return ModelType.LSTM
        
        # Statistical models
        if "volatility" in training_config.tags or "garch" in training_config.tags:
            return ModelType.GARCH
        
        if "point_process" in training_config.tags or "hawkes" in training_config.tags:
            return ModelType.HAWKES
        
        # Default to LSTM if no clear indicators
        return ModelType.LSTM
    
    def _create_model_trainer(
        self,
        model_type: ModelType,
        training_config: TrainingConfig
    ) -> Any:
        """
        Create model trainer based on model type.
        
        Args:
            model_type: Type of model to create
            training_config: Training configuration
            
        Returns:
            Model trainer instance
        """
        if model_type == ModelType.LSTM:
            return LSTMTimeSeriesForecaster(training_config)
        elif model_type == ModelType.GRU:
            return GRUTimeSeriesForecaster(training_config)
        elif model_type == ModelType.TRANSFORMER:
            return TransformerTimeSeriesForecaster(training_config)
        elif model_type == ModelType.GARCH:
            return GARCHModel(training_config)
        elif model_type == ModelType.HAWKES:
            return HawkesProcess(training_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_default_hyperparameters(self, model_type: ModelType) -> Dict[str, Any]:
        """
        Get default hyperparameters for model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of default hyperparameters
        """
        if model_type == ModelType.LSTM:
            return {
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False,
                "learning_rate": 0.001,
                "l1_regularization": 0.0,
                "l2_regularization": 0.0
            }
        elif model_type == ModelType.GRU:
            return {
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False,
                "learning_rate": 0.001,
                "l1_regularization": 0.0,
                "l2_regularization": 0.0
            }
        elif model_type == ModelType.TRANSFORMER:
            return {
                "d_model": 64,
                "nhead": 8,
                "num_encoder_layers": 3,
                "dim_feedforward": 256,
                "dropout": 0.1,
                "activation": "relu",
                "learning_rate": 0.001,
                "l1_regularization": 0.0,
                "l2_regularization": 0.0,
                "scheduler": "cosine"
            }
        elif model_type == ModelType.GARCH:
            return {
                "p": 1,
                "q": 1,
                "mean": "Zero",
                "vol": "GARCH",
                "dist": "normal"
            }
        elif model_type == ModelType.HAWKES:
            return {
                "mu": 0.1,
                "alpha": 0.5,
                "beta": 1.0,
                "kernel": "exponential"
            }
        else:
            return {}
    
    def _preprocess_data(
        self,
        data: pd.DataFrame,
        training_config: TrainingConfig
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Preprocess data for model training.
        
        Args:
            data: Input data
            training_config: Training configuration
            
        Returns:
            Preprocessed data
        """
        # Extract relevant features and target
        if not all(feature in data.columns for feature in training_config.features):
            missing_features = [f for f in training_config.features if f not in data.columns]
            raise ValueError(f"Missing features in data: {missing_features}")
        
        if training_config.target not in data.columns:
            raise ValueError(f"Target '{training_config.target}' not found in data columns")
        
        # Select features and target
        selected_columns = list(training_config.features) + [training_config.target]
        selected_data = data[selected_columns].copy()
        
        # Handle missing values
        if training_config.handle_missing_data == "drop":
            selected_data = selected_data.dropna()
        elif training_config.handle_missing_data == "interpolate":
            selected_data = selected_data.interpolate(method='linear', limit_direction='both')
        elif training_config.handle_missing_data == "fill_mean":
            selected_data = selected_data.fillna(selected_data.mean())
        elif training_config.handle_missing_data == "fill_zero":
            selected_data = selected_data.fillna(0)
        
        # Apply normalization
        if training_config.normalization == "standard":
            # Apply standard scaling (zero mean, unit variance)
            mean = selected_data.mean()
            std = selected_data.std()
            selected_data = (selected_data - mean) / std
            
            # Store normalization parameters in training config for later use
            training_config.additional_params["normalization_params"] = {
                "mean": mean.to_dict(),
                "std": std.to_dict()
            }
        elif training_config.normalization == "minmax":
            # Apply min-max scaling (0 to 1)
            min_vals = selected_data.min()
            max_vals = selected_data.max()
            selected_data = (selected_data - min_vals) / (max_vals - min_vals)
            
            # Store normalization parameters
            training_config.additional_params["normalization_params"] = {
                "min": min_vals.to_dict(),
                "max": max_vals.to_dict()
            }
        elif training_config.normalization == "robust":
            # Apply robust scaling (using quantiles)
            q1 = selected_data.quantile(0.25)
            q3 = selected_data.quantile(0.75)
            iqr = q3 - q1
            selected_data = (selected_data - q1) / iqr
            
            # Store normalization parameters
            training_config.additional_params["normalization_params"] = {
                "q1": q1.to_dict(),
                "q3": q3.to_dict(),
                "iqr": iqr.to_dict()
            }
        
        # Convert to numpy array for most models
        if training_config.additional_params.get("return_dataframe", False):
            return selected_data
        else:
            return selected_data.values
    
    def _optimize_hyperparameters(
        self,
        model_type: ModelType,
        training_config: TrainingConfig,
        data: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = 3600
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for the model.
        
        Args:
            model_type: Type of model
            training_config: Training configuration
            data: Preprocessed data
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization for {model_type.value} model")
        
        best_hyperparameters = self.hyperparameter_optimization_service.optimize(
            model_type=model_type,
            training_config=training_config,
            data=data,
            n_trials=n_trials,
            timeout=timeout
        )
        
        logger.info(f"Hyperparameter optimization completed, best parameters: {best_hyperparameters}")
        
        return best_hyperparameters
    
    def _train_model(
        self,
        model_trainer: Any,
        data: np.ndarray,
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Train model with given hyperparameters.
        
        Args:
            model_trainer: Model trainer instance
            data: Preprocessed data
            hyperparameters: Model hyperparameters
            
        Returns:
            Dictionary of training metrics
        """
        # The training process depends on the model type
        if isinstance(model_trainer, (LSTMTimeSeriesForecaster, GRUTimeSeriesForecaster, TransformerTimeSeriesForecaster)):
            # For neural network models, we need to create sequences and train/test split
            X, y = model_trainer.preprocess_data(data)
            
            # Get dimensions
            n_samples, seq_len, n_features = X.shape
            model_trainer.input_dim = n_features
            model_trainer.output_dim = y.shape[1]
            
            # Split data
            train_size = int(n_samples * model_trainer.config.train_split_ratio)
            val_size = int(n_samples * model_trainer.config.validation_split_ratio)
            test_size = n_samples - train_size - val_size
            
            # Train set
            X_train, y_train = X[:train_size], y[:train_size]
            
            # Validation set
            X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
            
            # Test set
            X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
            
            # Create data loaders
            train_loader, val_loader, test_loader = model_trainer.create_dataloaders(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            
            # Train model
            metrics = model_trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                hyperparameters=hyperparameters
            )
            
            return metrics
            
        elif isinstance(model_trainer, GARCHModel):
            # For GARCH model, training is simpler
            metrics = model_trainer.train(
                data=data,
                hyperparameters=hyperparameters
            )
            
            return metrics
            
        elif isinstance(model_trainer, HawkesProcess):
            # For Hawkes Process, we need to extract event times
            # Assume data is a time series of events (1 for event, 0 for no event)
            if len(data.shape) > 1:
                # Extract the target column (assuming it's the event indicator)
                target_idx = list(model_trainer.config.features).index(model_trainer.config.target)
                event_series = data[:, target_idx]
            else:
                event_series = data
            
            # Convert to event times (indices where event_series == 1)
            event_times = np.where(event_series == 1)[0]
            
            # Train model
            metrics = model_trainer.train(
                event_times=event_times,
                hyperparameters=hyperparameters
            )
            
            return metrics
            
        else:
            raise ValueError(f"Unsupported model trainer type: {type(model_trainer)}")
