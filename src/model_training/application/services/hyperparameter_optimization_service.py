"""
Service for hyperparameter optimization of machine learning models.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from datetime import datetime

from model_training.domain.entities.model import ModelType
from model_training.domain.entities.training_config import TrainingConfig

# Import model implementations for optimization
from model_training.models.time_series.lstm_model import LSTMTimeSeriesForecaster
from model_training.models.time_series.gru_model import GRUTimeSeriesForecaster
from model_training.models.time_series.transformer_model import TransformerTimeSeriesForecaster
from model_training.models.statistical.garch_model import GARCHModel
from model_training.models.statistical.hawkes_process import HawkesProcess


logger = logging.getLogger(__name__)


class HyperparameterOptimizationService:
    """Service for hyperparameter optimization of machine learning models."""
    
    def __init__(self, study_persistence_path: Optional[str] = None):
        """
        Initialize hyperparameter optimization service.
        
        Args:
            study_persistence_path: Optional path to persist Optuna studies
        """
        self.study_persistence_path = study_persistence_path
    
    def optimize(
        self,
        model_type: ModelType,
        training_config: TrainingConfig,
        data: Union[np.ndarray, pd.DataFrame],
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters for a given model type and training configuration.
        
        Args:
            model_type: Type of model to optimize
            training_config: Training configuration
            data: Input data for training
            n_trials: Number of optimization trials
            timeout: Timeout for optimization in seconds
            sampler: Optuna sampler to use (default: TPESampler)
            pruner: Optuna pruner to use (default: MedianPruner)
            
        Returns:
            Dictionary of best hyperparameters
        """
        logger.info(f"Starting hyperparameter optimization for {model_type.value} model with {n_trials} trials")
        
        # Create objective function based on model type
        objective = self._create_objective(model_type, training_config, data)
        
        # Set up sampler and pruner
        if sampler is None:
            sampler = TPESampler(seed=training_config.random_seed)
        
        if pruner is None:
            pruner = MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        
        # Create study name
        study_name = f"{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up storage URL if persistence path is provided
        storage = None
        if self.study_persistence_path:
            storage = f"sqlite:///{self.study_persistence_path}/{study_name}.db"
        
        # Create and run optimization study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",  # Minimize loss/error metrics
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best hyperparameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Hyperparameter optimization completed. Best value: {best_value:.6f}, Parameters: {best_params}")
        
        # Return best hyperparameters
        return best_params
    
    def _create_objective(
        self,
        model_type: ModelType,
        training_config: TrainingConfig,
        data: Union[np.ndarray, pd.DataFrame]
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        Create objective function for hyperparameter optimization.
        
        Args:
            model_type: Type of model to optimize
            training_config: Training configuration
            data: Input data for training
            
        Returns:
            Objective function for optimization
        """
        if model_type == ModelType.LSTM:
            return self._create_lstm_objective(training_config, data)
        elif model_type == ModelType.GRU:
            return self._create_gru_objective(training_config, data)
        elif model_type == ModelType.TRANSFORMER:
            return self._create_transformer_objective(training_config, data)
        elif model_type == ModelType.GARCH:
            return self._create_garch_objective(training_config, data)
        elif model_type == ModelType.HAWKES:
            return self._create_hawkes_objective(training_config, data)
        else:
            raise ValueError(f"Unsupported model type for hyperparameter optimization: {model_type}")
    
    def _create_lstm_objective(
        self,
        training_config: TrainingConfig,
        data: np.ndarray
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        Create objective function for LSTM hyperparameter optimization.
        
        Args:
            training_config: Training configuration
            data: Input data for training
            
        Returns:
            Objective function for optimization
        """
        def objective(trial: optuna.trial.Trial) -> float:
            # Define hyperparameter search space
            hyperparameters = {
                "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, log=True),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "l1_regularization": trial.suggest_float("l1_regularization", 0.0, 0.01),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 0.01)
            }
            
            try:
                # Create model trainer
                model_trainer = LSTMTimeSeriesForecaster(training_config)
                
                # Preprocess data
                X, y = model_trainer.preprocess_data(data)
                
                # Get dimensions
                n_samples, seq_len, n_features = X.shape
                model_trainer.input_dim = n_features
                model_trainer.output_dim = y.shape[1]
                
                # Split data
                train_size = int(n_samples * training_config.train_split_ratio)
                val_size = int(n_samples * training_config.validation_split_ratio)
                
                # Train set
                X_train, y_train = X[:train_size], y[:train_size]
                
                # Validation set
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                
                # Test set (not used in optimization)
                X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
                
                # Create data loaders
                train_loader, val_loader, test_loader = model_trainer.create_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                
                # Train model with early stopping
                metrics = model_trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    hyperparameters=hyperparameters
                )
                
                # Return validation loss as optimization metric
                val_loss = metrics.get("loss", float('inf'))
                
                return val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective
    
    def _create_gru_objective(
        self,
        training_config: TrainingConfig,
        data: np.ndarray
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        Create objective function for GRU hyperparameter optimization.
        
        Args:
            training_config: Training configuration
            data: Input data for training
            
        Returns:
            Objective function for optimization
        """
        def objective(trial: optuna.trial.Trial) -> float:
            # Define hyperparameter search space
            hyperparameters = {
                "hidden_dim": trial.suggest_int("hidden_dim", 16, 256, log=True),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "l1_regularization": trial.suggest_float("l1_regularization", 0.0, 0.01),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 0.01)
            }
            
            try:
                # Create model trainer
                model_trainer = GRUTimeSeriesForecaster(training_config)
                
                # Preprocess data
                X, y = model_trainer.preprocess_data(data)
                
                # Get dimensions
                n_samples, seq_len, n_features = X.shape
                model_trainer.input_dim = n_features
                model_trainer.output_dim = y.shape[1]
                
                # Split data
                train_size = int(n_samples * training_config.train_split_ratio)
                val_size = int(n_samples * training_config.validation_split_ratio)
                
                # Train set
                X_train, y_train = X[:train_size], y[:train_size]
                
                # Validation set
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                
                # Test set (not used in optimization)
                X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
                
                # Create data loaders
                train_loader, val_loader, test_loader = model_trainer.create_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                
                # Train model with early stopping
                metrics = model_trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    hyperparameters=hyperparameters
                )
                
                # Return validation loss as optimization metric
                val_loss = metrics.get("loss", float('inf'))
                
                return val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective
    
    def _create_transformer_objective(
        self,
        training_config: TrainingConfig,
        data: np.ndarray
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        Create objective function for Transformer hyperparameter optimization.
        
        Args:
            training_config: Training configuration
            data: Input data for training
            
        Returns:
            Objective function for optimization
        """
        def objective(trial: optuna.trial.Trial) -> float:
            # Define hyperparameter search space
            hyperparameters = {
                "d_model": trial.suggest_int("d_model", 16, 256, log=True),
                "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
                "num_encoder_layers": trial.suggest_int("num_encoder_layers", 1, 6),
                "dim_feedforward": trial.suggest_int("dim_feedforward", 64, 512, log=True),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "l1_regularization": trial.suggest_float("l1_regularization", 0.0, 0.01),
                "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 0.01),
                "scheduler": trial.suggest_categorical("scheduler", ["cosine", "plateau", "step", None])
            }
            
            # Make sure nhead divides d_model evenly
            while hyperparameters["d_model"] % hyperparameters["nhead"] != 0:
                hyperparameters["d_model"] += 1
            
            try:
                # Create model trainer
                model_trainer = TransformerTimeSeriesForecaster(training_config)
                
                # Preprocess data
                X, y = model_trainer.preprocess_data(data)
                
                # Get dimensions
                n_samples, seq_len, n_features = X.shape
                model_trainer.input_dim = n_features
                model_trainer.output_dim = y.shape[1]
                
                # Split data
                train_size = int(n_samples * training_config.train_split_ratio)
                val_size = int(n_samples * training_config.validation_split_ratio)
                
                # Train set
                X_train, y_train = X[:train_size], y[:train_size]
                
                # Validation set
                X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
                
                # Test set (not used in optimization)
                X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
                
                # Create data loaders
                train_loader, val_loader, test_loader = model_trainer.create_dataloaders(
                    X_train, y_train, X_val, y_val, X_test, y_test
                )
                
                # Train model with early stopping
                metrics = model_trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    hyperparameters=hyperparameters
                )
                
                # Return validation loss as optimization metric
                val_loss = metrics.get("loss", float('inf'))
                
                return val_loss
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective
    
    def _create_garch_objective(
        self,
        training_config: TrainingConfig,
        data: np.ndarray
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        Create objective function for GARCH hyperparameter optimization.
        
        Args:
            training_config: Training configuration
            data: Input data for training
            
        Returns:
            Objective function for optimization
        """
        def objective(trial: optuna.trial.Trial) -> float:
            # Define hyperparameter search space
            hyperparameters = {
                "p": trial.suggest_int("p", 1, 3),
                "q": trial.suggest_int("q", 1, 3),
                "mean": trial.suggest_categorical("mean", ["Zero", "Constant", "AR"]),
                "vol": trial.suggest_categorical("vol", ["GARCH", "EGARCH", "GJR"]),
                "dist": trial.suggest_categorical("dist", ["normal", "studentst", "skewstudent"])
            }
            
            try:
                # Create model trainer
                model_trainer = GARCHModel(training_config)
                
                # Train model
                metrics = model_trainer.train(
                    data=data,
                    hyperparameters=hyperparameters
                )
                
                # For GARCH models, we want to minimize information criteria (AIC/BIC)
                # or maximize log-likelihood
                objective_value = metrics.get("aic", float('inf'))
                
                return objective_value
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective
    
    def _create_hawkes_objective(
        self,
        training_config: TrainingConfig,
        data: np.ndarray
    ) -> Callable[[optuna.trial.Trial], float]:
        """
        Create objective function for Hawkes Process hyperparameter optimization.
        
        Args:
            training_config: Training configuration
            data: Input data for training
            
        Returns:
            Objective function for optimization
        """
        def objective(trial: optuna.trial.Trial) -> float:
            # Define hyperparameter search space
            hyperparameters = {
                "mu": trial.suggest_float("mu", 0.001, 1.0, log=True),
                "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
                "beta": trial.suggest_float("beta", 0.1, 20.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["exponential", "power_law"])
            }
            
            # Ensure stability condition for exponential kernel
            if hyperparameters["kernel"] == "exponential" and hyperparameters["alpha"] >= hyperparameters["beta"]:
                hyperparameters["beta"] = hyperparameters["alpha"] * 1.1  # Ensure alpha < beta
            
            try:
                # Create model trainer
                model_trainer = HawkesProcess(training_config)
                
                # For Hawkes Process, we need to extract event times
                # Assume data is a time series of events (1 for event, 0 for no event)
                if len(data.shape) > 1:
                    # Extract the target column (assuming it's the event indicator)
                    target_idx = list(training_config.features).index(training_config.target)
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
                
                # For Hawkes Process, we want to maximize log-likelihood or minimize AIC/BIC
                objective_value = -metrics.get("log_likelihood", -float('inf'))  # Negate log-likelihood for minimization
                
                return objective_value
                
            except Exception as e:
                logger.warning(f"Trial failed with error: {str(e)}")
                raise optuna.exceptions.TrialPruned()
        
        return objective