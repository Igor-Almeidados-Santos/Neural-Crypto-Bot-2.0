"""
Use case for evaluating trained machine learning models.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import uuid

from model_training.domain.entities.model import Model, ModelStatus, ModelType
from model_training.domain.entities.evaluation_metric import EvaluationMetric, MetricType, MetricDirection
from model_training.domain.entities.training_config import TrainingConfig
from model_training.infrastructure.experiment_tracker import ExperimentTracker
from model_training.infrastructure.model_repository import ModelRepository

# Import model implementations
from model_training.models.time_series.lstm_model import LSTMTimeSeriesForecaster
from model_training.models.time_series.gru_model import GRUTimeSeriesForecaster
from model_training.models.time_series.transformer_model import TransformerTimeSeriesForecaster
from model_training.models.statistical.garch_model import GARCHModel
from model_training.models.statistical.hawkes_process import HawkesProcess


logger = logging.getLogger(__name__)


class EvaluateModelUseCase:
    """Use case for evaluating trained machine learning models."""
    
    def __init__(
        self,
        model_repository: ModelRepository,
        experiment_tracker: ExperimentTracker
    ):
        """
        Initialize EvaluateModelUseCase.
        
        Args:
            model_repository: Repository for model storage
            experiment_tracker: Tracker for experiment logging
        """
        self.model_repository = model_repository
        self.experiment_tracker = experiment_tracker
    
    def execute(
        self,
        model: Model,
        data: pd.DataFrame,
        evaluation_data_split: str = "test",
        custom_metrics: Optional[List[str]] = None
    ) -> Model:
        """
        Execute the use case to evaluate a model.
        
        Args:
            model: Model entity to evaluate
            data: Data for evaluation
            evaluation_data_split: Data split to use for evaluation ('train', 'validation', 'test')
            custom_metrics: Optional list of custom metrics to compute
            
        Returns:
            Updated model entity with evaluation metrics
        """
        if model.status not in [ModelStatus.TRAINED, ModelStatus.EVALUATED]:
            raise ValueError(f"Cannot evaluate model with status: {model.status}")
        
        # Update model status
        model.update_status(ModelStatus.EVALUATING)
        
        # Start tracking evaluation
        run_id = self.experiment_tracker.start_experiment(
            name=f"evaluate_{model.name}_{model.version}",
            tags=model.tags + ["evaluation"],
            params={"evaluation_data_split": evaluation_data_split}
        )
        
        try:
            # Load model from repository
            model_trainer = self._load_model_trainer(model)
            
            # Preprocess data
            preprocessed_data = self._preprocess_data(data, model.training_config)
            
            # Evaluate model
            logger.info(f"Evaluating {model.type.value} model {model.name} version {model.version}...")
            
            metrics = self._evaluate_model(
                model_trainer=model_trainer,
                data=preprocessed_data,
                data_split=evaluation_data_split,
                custom_metrics=custom_metrics
            )
            
            # Add metrics to model
            for metric_name, metric_value in metrics.items():
                # Determine metric type and direction
                metric_type, direction = self._get_metric_info(metric_name)
                
                # Create evaluation metric entity
                evaluation_metric = EvaluationMetric(
                    name=metric_name,
                    value=metric_value,
                    type=metric_type,
                    direction=direction,
                    dataset_split=evaluation_data_split,
                    timestamp=datetime.utcnow()
                )
                
                # Add metric to model
                model.add_metric(evaluation_metric)
                
                # Log metric to experiment tracker
                self.experiment_tracker.log_metric(metric_name, metric_value)
            
            # Update model status
            model.update_status(ModelStatus.EVALUATED)
            
            # End tracking
            self.experiment_tracker.end_experiment(status="success")
            
            return model
            
        except Exception as e:
            # Handle failure
            model.update_status(ModelStatus.FAILED)
            model.metadata["evaluation_error"] = str(e)
            
            # Log failure
            self.experiment_tracker.log_param("error", str(e))
            self.experiment_tracker.end_experiment(status="failed")
            
            logger.error(f"Model evaluation failed: {str(e)}", exc_info=True)
            
            # Re-raise exception
            raise
    
    def _load_model_trainer(self, model: Model) -> Any:
        """
        Load model trainer from repository.
        
        Args:
            model: Model entity
            
        Returns:
            Model trainer instance
        """
        if not model.model_artifacts_path:
            raise ValueError(f"Model {model.id} has no artifacts path")
        
        # Create appropriate model trainer based on model type
        if model.type == ModelType.LSTM:
            model_trainer = LSTMTimeSeriesForecaster(model.training_config)
        elif model.type == ModelType.GRU:
            model_trainer = GRUTimeSeriesForecaster(model.training_config)
        elif model.type == ModelType.TRANSFORMER:
            model_trainer = TransformerTimeSeriesForecaster(model.training_config)
        elif model.type == ModelType.GARCH:
            model_trainer = GARCHModel(model.training_config)
        elif model.type == ModelType.HAWKES:
            model_trainer = HawkesProcess(model.training_config)
        else:
            raise ValueError(f"Unsupported model type: {model.type}")
        
        # Load model from repository
        self.model_repository.load_model(
            model_id=model.id,
            model_trainer=model_trainer,
            artifacts_path=model.model_artifacts_path
        )
        
        return model_trainer
    
    def _preprocess_data(
        self,
        data: pd.DataFrame,
        training_config: TrainingConfig
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Preprocess data for model evaluation.
        
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
        
        # Apply normalization using stored parameters
        norm_params = training_config.additional_params.get("normalization_params", {})
        
        if training_config.normalization == "standard" and norm_params:
            mean = pd.Series(norm_params.get("mean", {}))
            std = pd.Series(norm_params.get("std", {}))
            selected_data = (selected_data - mean) / std
            
        elif training_config.normalization == "minmax" and norm_params:
            min_vals = pd.Series(norm_params.get("min", {}))
            max_vals = pd.Series(norm_params.get("max", {}))
            selected_data = (selected_data - min_vals) / (max_vals - min_vals)
            
        elif training_config.normalization == "robust" and norm_params:
            q1 = pd.Series(norm_params.get("q1", {}))
            iqr = pd.Series(norm_params.get("iqr", {}))
            selected_data = (selected_data - q1) / iqr
        
        # Convert to numpy array for most models
        if training_config.additional_params.get("return_dataframe", False):
            return selected_data
        else:
            return selected_data.values
    
    def _evaluate_model(
        self,
        model_trainer: Any,
        data: Union[np.ndarray, pd.DataFrame],
        data_split: str = "test",
        custom_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            model_trainer: Model trainer instance
            data: Preprocessed data
            data_split: Data split to use for evaluation
            custom_metrics: Optional list of custom metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        # The evaluation process depends on the model type
        if isinstance(model_trainer, (LSTMTimeSeriesForecaster, GRUTimeSeriesForecaster, TransformerTimeSeriesForecaster)):
            # For neural network models, we need to create sequences
            X, y_true = model_trainer.preprocess_data(data)
            
            # Make predictions
            y_pred = model_trainer.predict(X)
            
            # Compute standard metrics
            metrics = self._compute_time_series_metrics(y_true, y_pred, custom_metrics)
            
            return metrics
            
        elif isinstance(model_trainer, GARCHModel):
            # For GARCH model, evaluate volatility forecasting
            # Split data for evaluation
            if data_split == "test":
                # Use last portion of data for testing
                split_ratio = model_trainer.config.test_split_ratio
                split_idx = int((1 - split_ratio) * len(data))
                eval_data = data[split_idx:]
            elif data_split == "validation":
                # Use validation portion
                train_ratio = model_trainer.config.train_split_ratio
                val_ratio = model_trainer.config.validation_split_ratio
                start_idx = int(train_ratio * len(data))
                end_idx = int((train_ratio + val_ratio) * len(data))
                eval_data = data[start_idx:end_idx]
            else:
                # Use training portion
                split_ratio = model_trainer.config.train_split_ratio
                split_idx = int(split_ratio * len(data))
                eval_data = data[:split_idx]
            
            # Get volatility forecasts
            horizon = len(eval_data) if len(eval_data) < 30 else 30  # Use at most 30 days for forecast horizon
            forecasts = model_trainer.predict(horizon=horizon)
            
            # Compute volatility metrics
            metrics = self._compute_volatility_metrics(eval_data, forecasts)
            
            return metrics
            
        elif isinstance(model_trainer, HawkesProcess):
            # For Hawkes Process, evaluate point process modeling
            # Convert data to event times
            if len(data.shape) > 1:
                # Extract the target column (assuming it's the event indicator)
                target_idx = list(model_trainer.config.features).index(model_trainer.config.target)
                event_series = data[:, target_idx]
            else:
                event_series = data
            
            # Convert to event times (indices where event_series == 1)
            event_times = np.where(event_series == 1)[0]
            
            # Split data for evaluation
            if data_split == "test":
                # Use last portion of event times for testing
                split_ratio = model_trainer.config.test_split_ratio
                split_idx = int((1 - split_ratio) * len(event_times))
                eval_events = event_times[split_idx:]
                past_events = event_times[:split_idx]
            elif data_split == "validation":
                # Use validation portion
                train_ratio = model_trainer.config.train_split_ratio
                val_ratio = model_trainer.config.validation_split_ratio
                start_idx = int(train_ratio * len(event_times))
                end_idx = int((train_ratio + val_ratio) * len(event_times))
                eval_events = event_times[start_idx:end_idx]
                past_events = event_times[:start_idx]
            else:
                # For training evaluation, use all events (not ideal but practical)
                eval_events = event_times
                past_events = event_times
            
            # Compute point process metrics
            metrics = self._compute_point_process_metrics(past_events, eval_events, model_trainer)
            
            return metrics
            
        else:
            raise ValueError(f"Unsupported model trainer type: {type(model_trainer)}")
    
    def _compute_time_series_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        custom_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for time series forecasting.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            custom_metrics: Optional list of custom metrics to compute
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        metrics["mse"] = float(mse)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics["rmse"] = float(rmse)
        
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        metrics["mae"] = float(mae)
        
        # Mean Absolute Percentage Error (avoid division by zero)
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics["mape"] = float(mape)
        
        # R-squared (coefficient of determination)
        if np.var(y_true) > 0:
            r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
            metrics["r2"] = float(r2)
        
        # Directional Accuracy (for financial time series)
        # Measures how often the predicted direction matches the actual direction
        if y_true.shape[0] > 1:
            true_direction = np.sign(y_true[1:] - y_true[:-1])
            pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
            directional_accuracy = np.mean(true_direction == pred_direction)
            metrics["directional_accuracy"] = float(directional_accuracy)
        
        # Custom metrics
        if custom_metrics:
            for metric_name in custom_metrics:
                if metric_name == "sharpe_ratio" and y_pred.shape[0] > 1:
                    # Simple Sharpe ratio calculation (return / volatility)
                    returns = y_pred[1:] - y_pred[:-1]
                    if np.std(returns) > 0:
                        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
                        metrics["sharpe_ratio"] = float(sharpe)
                
                elif metric_name == "sortino_ratio" and y_pred.shape[0] > 1:
                    # Sortino ratio (return / downside volatility)
                    returns = y_pred[1:] - y_pred[:-1]
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                        sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)  # Annualized
                        metrics["sortino_ratio"] = float(sortino)
                
                elif metric_name == "max_drawdown" and y_pred.shape[0] > 1:
                    # Maximum drawdown
                    peak = y_pred[0]
                    max_dd = 0
                    for value in y_pred[1:]:
                        if value > peak:
                            peak = value
                        dd = (peak - value) / peak
                        if dd > max_dd:
                            max_dd = dd
                    metrics["max_drawdown"] = float(max_dd)
        
        return metrics
    
    def _compute_volatility_metrics(
        self,
        actual_data: np.ndarray,
        forecasts: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for volatility forecasting.
        
        Args:
            actual_data: Actual data for the forecast period
            forecasts: Dictionary of forecast results including volatility
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Extract volatility forecasts
        forecast_volatility = forecasts.get("volatility", np.array([]))
        
        # Calculate realized volatility from actual returns
        if len(actual_data.shape) > 1:
            # Extract returns from multidimensional data (assuming first column is price)
            prices = actual_data[:, 0]
        else:
            # Use data directly as prices
            prices = actual_data
            
        # Calculate log returns
        if len(prices) > 1:
            returns = np.diff(np.log(prices)) * 100
            realized_volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Compare forecasted volatility with realized volatility
            if len(forecast_volatility) > 0:
                mean_forecast_vol = np.mean(forecast_volatility)
                
                # Mean Absolute Error of volatility
                vol_mae = np.abs(mean_forecast_vol - realized_volatility)
                metrics["volatility_mae"] = float(vol_mae)
                
                # Relative Error of volatility
                if realized_volatility > 0:
                    vol_rel_error = np.abs(mean_forecast_vol - realized_volatility) / realized_volatility
                    metrics["volatility_relative_error"] = float(vol_rel_error)
        
        # Add GARCH-specific metrics if available
        for key in ["log_likelihood", "aic", "bic"]:
            if key in forecasts:
                metrics[key] = float(forecasts[key])
        
        return metrics
    
    def _compute_point_process_metrics(
        self,
        past_events: np.ndarray,
        eval_events: np.ndarray,
        model_trainer: HawkesProcess
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for point process modeling.
        
        Args:
            past_events: Past event times used for prediction
            eval_events: Event times for evaluation
            model_trainer: Hawkes Process model trainer
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Get the time range for evaluation
        if len(eval_events) > 0:
            t_start = min(eval_events)
            t_end = max(eval_events)
            
            # Calculate expected number of events
            expected_events = model_trainer.expected_events(t_start, t_end, past_events)
            actual_events = len(eval_events)
            
            # Error in event count prediction
            count_error = np.abs(expected_events - actual_events)
            metrics["event_count_error"] = float(count_error)
            
            # Relative error in event count prediction
            if actual_events > 0:
                rel_count_error = count_error / actual_events
                metrics["relative_event_count_error"] = float(rel_count_error)
            
            # Calculate log-likelihood of the observed events under the model
            if hasattr(model_trainer, "fitted_params") and model_trainer.fitted_params is not None:
                mu = model_trainer.fitted_params["mu"]
                alpha = model_trainer.fitted_params["alpha"]
                beta = model_trainer.fitted_params["beta"]
                
                # Calculate log-likelihood
                ll = 0.0
                
                # Sum of log-intensities at event times
                for i, t in enumerate(eval_events):
                    # Get intensity at event time
                    intensity_t = model_trainer.intensity(t, past_events, alpha, beta, mu)
                    
                    # Add log-intensity to log-likelihood
                    if intensity_t > 0:
                        ll += np.log(intensity_t)
                
                # Second term: Integral of intensity function over evaluation period
                if model_trainer.kernel == "exponential":
                    # Calculate integral term analytically
                    integral_term = mu * (t_end - t_start)
                    
                    # For each past event, calculate its contribution to the integral
                    for t in past_events:
                        if t < t_start:
                            time_diff_start = t_start - t
                            time_diff_end = t_end - t
                            integral_term += alpha / beta * (np.exp(-beta * time_diff_start) - np.exp(-beta * time_diff_end))
                else:
                    # Use numerical integration (approximation)
                    grid_size = 1000
                    time_grid = np.linspace(t_start, t_end, grid_size)
                    dt = (t_end - t_start) / (grid_size - 1)
                    
                    intensity_values = np.zeros(grid_size)
                    for i, t in enumerate(time_grid):
                        intensity_values[i] = model_trainer.intensity(t, past_events, alpha, beta, mu)
                    
                    integral_term = np.sum(intensity_values) * dt
                
                # Complete log-likelihood
                ll -= integral_term
                metrics["log_likelihood"] = float(ll)
                
                # Add AIC and BIC if available
                n = len(eval_events)
                k = 3  # Number of parameters (mu, alpha, beta)
                aic = 2 * k - 2 * ll
                bic = k * np.log(n) - 2 * ll
                
                metrics["aic"] = float(aic)
                metrics["bic"] = float(bic)
                
                # Branching ratio (average number of triggered events per event)
                branching_ratio = model_trainer.branching_ratio()
                metrics["branching_ratio"] = float(branching_ratio)
        
        return metrics
    
    def _get_metric_info(self, metric_name: str) -> Tuple[MetricType, MetricDirection]:
        """
        Get metric type and direction (higher/lower better) for a given metric name.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of MetricType and MetricDirection
        """
        # Dictionary mapping metric names to (type, direction) tuples
        metric_info = {
            # Regression metrics
            "mse": (MetricType.MSE, MetricDirection.LOWER_BETTER),
            "rmse": (MetricType.RMSE, MetricDirection.LOWER_BETTER),
            "mae": (MetricType.MAE, MetricDirection.LOWER_BETTER),
            "mape": (MetricType.MAPE, MetricDirection.LOWER_BETTER),
            "r2": (MetricType.R2, MetricDirection.HIGHER_BETTER),
            
            # Time series specific
            "directional_accuracy": (MetricType.DIRECTIONAL_ACCURACY, MetricDirection.HIGHER_BETTER),
            "sharpe_ratio": (MetricType.SHARPE_RATIO, MetricDirection.HIGHER_BETTER),
            "sortino_ratio": (MetricType.SORTINO_RATIO, MetricDirection.HIGHER_BETTER),
            "max_drawdown": (MetricType.MAX_DRAWDOWN, MetricDirection.LOWER_BETTER),
            
            # Volatility metrics
            "volatility_mae": (MetricType.MAE, MetricDirection.LOWER_BETTER),
            "volatility_relative_error": (MetricType.MAPE, MetricDirection.LOWER_BETTER),
            
            # Point process metrics
            "event_count_error": (MetricType.MAE, MetricDirection.LOWER_BETTER),
            "relative_event_count_error": (MetricType.MAPE, MetricDirection.LOWER_BETTER),
            "log_likelihood": (MetricType.CUSTOM, MetricDirection.HIGHER_BETTER),
            "aic": (MetricType.CUSTOM, MetricDirection.LOWER_BETTER),
            "bic": (MetricType.CUSTOM, MetricDirection.LOWER_BETTER),
            "branching_ratio": (MetricType.CUSTOM, MetricDirection.LOWER_BETTER)
        }
        
        # Return metric info if found, default to custom otherwise
        return metric_info.get(metric_name, (MetricType.CUSTOM, MetricDirection.HIGHER_BETTER))training_config)
        elif model.type == ModelType.GARCH:
            model_trainer = GARCHModel(model.training_config)
        elif model.type == ModelType.HAWKES:
            model_trainer = HawkesProcess(model.training_config)
        else:
            raise ValueError(f"Unsupported model type: {model.type}")
        
        # Load model from repository
        self.model_repository.load_model(
            model_id=model.id,
            model_trainer=model_trainer,
            artifacts_path=model.model_artifacts_path
        )
        
        return model_trainer
    
    def _preprocess_data(
        self,
        data: pd.DataFrame,
        training_config: TrainingConfig
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Preprocess data for model evaluation.
        
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
        
        # Apply normalization using stored parameters
        norm_params = training_config.additional_params.get("normalization_params", {})
        
        if training_config.normalization == "standard" and norm_params:
            mean = pd.Series(norm_params.get("mean", {}))
            std = pd.Series(norm_params.get("std", {}))
            selected_data = (selected_data - mean) / std
            
        elif training_config.normalization == "minmax" and norm_params:
            min_vals = pd.Series(norm_params.get("min", {}))
            max_vals = pd.Series(norm_params.get("max", {}))
            selected_data = (selected_data - min_vals) / (max_vals - min_vals)
            
        elif training_config.normalization == "robust" and norm_params:
            q1 = pd.Series(norm_params.get("q1", {}))
            iqr = pd.Series(norm_params.get("iqr", {}))
            selected_data = (selected_data - q1) / iqr
        
        # Convert to numpy array for most models
        if training_config.additional_params.get("return_dataframe", False):
            return selected_data
        else:
            return selected_data.values
    
    def _evaluate_model(
        self,
        model_trainer: Any,
        data: Union[np.ndarray, pd.DataFrame],
        data_split: str = "test",
        custom_metrics: Optional[List[str]] = None
    ) -> Dict[str, float]: