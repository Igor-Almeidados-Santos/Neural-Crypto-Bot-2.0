"""
GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model for volatility forecasting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import joblib
from statsmodels.tsa.stattools import adfuller

from src.model_training.domain.entities.training_config import TrainingConfig


class GARCHModel:
    """GARCH model for volatility forecasting."""
    
    def __init__(
        self,
        config: TrainingConfig,
        p: int = 1,
        q: int = 1,
        mean: str = "Zero",
        vol: str = "GARCH",
        dist: str = "normal"
    ):
        """
        Initialize GARCH model.
        
        Args:
            config: Training configuration
            p: GARCH order (volatility persistence)
            q: ARCH order (error persistence)
            mean: Mean model ('Zero', 'Constant', 'AR', 'ARX', 'HAR', 'HARX')
            vol: Volatility model ('GARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'GJR')
            dist: Error distribution ('normal', 'studentst', 'skewstudent', 'ged')
        """
        self.config = config
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        self.model = None
        self.result = None
        self.fitted_model = None
        self.is_stationary = False
        self.transformation_applied = False
        self.differencing_order = 0
    
    def check_stationarity(self, data: np.ndarray, threshold: float = 0.05) -> bool:
        """
        Check if the time series is stationary using Augmented Dickey-Fuller test.
        
        Args:
            data: Time series data
            threshold: Significance threshold
            
        Returns:
            bool: True if stationary, False otherwise
        """
        result = adfuller(data)
        p_value = result[1]
        self.is_stationary = p_value < threshold
        return self.is_stationary
    
    def make_stationary(self, data: np.ndarray) -> np.ndarray:
        """
        Transform time series to make it stationary.
        
        Args:
            data: Time series data
            
        Returns:
            np.ndarray: Stationary time series
        """
        # Try differencing until stationary
        diff_data = data.copy()
        self.differencing_order = 0
        
        # Check initial stationarity
        if self.check_stationarity(diff_data):
            return diff_data
        
        # Apply differencing
        self.transformation_applied = True
        max_diff = 2  # Maximum order of differencing
        
        for i in range(1, max_diff + 1):
            diff_data = np.diff(diff_data)
            self.differencing_order = i
            
            if self.check_stationarity(diff_data):
                break
        
        return diff_data
    
    def reverse_transformation(self, forecasts: np.ndarray, last_values: np.ndarray) -> np.ndarray:
        """
        Reverse the transformation applied to make the series stationary.
        
        Args:
            forecasts: Forecasted values
            last_values: Last values of the original series needed for reverting differencing
            
        Returns:
            np.ndarray: Transformed forecasts
        """
        if not self.transformation_applied:
            return forecasts
        
        reverted = forecasts.copy()
        
        # Revert differencing
        for i in range(self.differencing_order):
            # For each level of differencing, we need an additional last value
            last_val = last_values[-(i+1)]
            reverted = np.cumsum(reverted) + last_val
        
        return reverted
    
    def prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare data for GARCH model, including making it stationary if needed.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of prepared data and original last values (if transformed)
        """
        # For GARCH, we need returns not prices
        returns = np.diff(np.log(data)) * 100
        
        # Store last values for reverting transformations later
        last_values = None
        
        # Check if stationary
        if not self.check_stationarity(returns):
            last_values = returns[-self.differencing_order:] if self.differencing_order > 0 else None
            returns = self.make_stationary(returns)
        
        return returns, last_values
    
    def train(
        self, 
        data: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Train GARCH model.
        
        Args:
            data: Input data
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Update hyperparameters if provided
        if hyperparameters:
            self.p = hyperparameters.get("p", self.p)
            self.q = hyperparameters.get("q", self.q)
            self.mean = hyperparameters.get("mean", self.mean)
            self.vol = hyperparameters.get("vol", self.vol)
            self.dist = hyperparameters.get("dist", self.dist)
        
        # Prepare data
        returns, self.last_values = self.prepare_data(data)
        
        # Split data
        train_size = int(len(returns) * (self.config.train_split_ratio + self.config.validation_split_ratio))
        train_returns = returns[:train_size]
        test_returns = returns[train_size:]
        
        # Create model
        self.model = arch_model(
            train_returns,
            p=self.p,
            q=self.q,
            mean=self.mean,
            vol=self.vol,
            dist=self.dist
        )
        
        # Fit model
        self.result = self.model.fit(
            disp="off",
            update_freq=0
        )
        
        # Save fitted model
        self.fitted_model = self.result
        
        # Make predictions on test set
        forecasts = self.result.forecast(horizon=len(test_returns))
        forecast_variance = forecasts.variance.values[-1]
        
        # Calculate metrics
        mse = np.mean((test_returns**2 - forecast_variance)**2)
        mae = np.mean(np.abs(test_returns**2 - forecast_variance))
        
        # Calculate log-likelihood
        ll = self.result.loglikelihood
        
        # Calculate AIC and BIC
        aic = self.result.aic
        bic = self.result.bic
        
        return {
            "mse": mse,
            "mae": mae,
            "log_likelihood": ll,
            "aic": aic,
            "bic": bic
        }
    
    def predict(self, horizon: int = 10) -> Dict[str, np.ndarray]:
        """
        Generate volatility forecasts.
        
        Args:
            horizon: Forecast horizon
            
        Returns:
            Dictionary of forecast results including mean and variance
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        # Generate forecasts
        forecasts = self.fitted_model.forecast(horizon=horizon)
        
        # Extract forecast components
        forecast_mean = forecasts.mean.values[-1]
        forecast_variance = forecasts.variance.values[-1]
        forecast_residual_variance = forecasts.residual_variance.values[-1]
        
        # Convert to volatility (standard deviation)
        forecast_volatility = np.sqrt(forecast_variance)
        
        # If transformation was applied, revert it
        if self.transformation_applied and self.last_values is not None:
            forecast_mean = self.reverse_transformation(forecast_mean, self.last_values)
            # Note: Volatility doesn't need to be reverted in the same way
        
        return {
            "mean": forecast_mean,
            "variance": forecast_variance,
            "volatility": forecast_volatility,
            "residual_variance": forecast_residual_variance
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        # Save model parameters and state
        model_state = {
            "p": self.p,
            "q": self.q,
            "mean": self.mean,
            "vol": self.vol,
            "dist": self.dist,
            "is_stationary": self.is_stationary,
            "transformation_applied": self.transformation_applied,
            "differencing_order": self.differencing_order,
            "last_values": self.last_values,
            "config": self.config.to_dict(),
            "model_params": self.fitted_model.params.to_dict() if hasattr(self.fitted_model.params, 'to_dict') else dict(self.fitted_model.params),
            "model_summary": str(self.fitted_model.summary())
        }
        
        # Save using joblib for more complex objects
        joblib.dump(model_state, filepath)
    
    def load(self, filepath: str, data: Optional[np.ndarray] = None) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            data: Optional data to recreate the model (needed for ARCH models)
        """
        # Load model state
        model_state = joblib.load(filepath)
        
        # Set model attributes
        self.p = model_state["p"]
        self.q = model_state["q"]
        self.mean = model_state["mean"]
        self.vol = model_state["vol"]
        self.dist = model_state["dist"]
        self.is_stationary = model_state["is_stationary"]
        self.transformation_applied = model_state["transformation_applied"]
        self.differencing_order = model_state["differencing_order"]
        self.last_values = model_state["last_values"]
        
        # Load config
        self.config = TrainingConfig.from_dict(model_state["config"])
        
        # If data is provided, recreate the model
        if data is not None:
            # Prepare data
            returns, _ = self.prepare_data(data)
            
            # Create model
            self.model = arch_model(
                returns,
                p=self.p,
                q=self.q,
                mean=self.mean,
                vol=self.vol,
                dist=self.dist
            )
            
            # Create a dummy result object
            self.fitted_model = self.model.fix(model_state["model_params"])
        else:
            # Without data, we can only load parameters but can't recreate the model
            print("Warning: Data not provided. Model loaded with parameters only.")
            self.model = None
            self.fitted_model = None
    
    def plot_volatility(self, data: np.ndarray, forecast_horizon: int = 10, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot historical and forecasted volatility.
        
        Args:
            data: Historical data
            forecast_horizon: Forecast horizon
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
        
        # Prepare data
        returns, _ = self.prepare_data(data)
        
        # Get conditional volatility (historical)
        conditional_vol = np.sqrt(self.fitted_model.conditional_volatility)
        
        # Generate forecasts
        forecasts = self.predict(horizon=forecast_horizon)
        forecast_vol = forecasts["volatility"]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical volatility
        ax.plot(range(len(conditional_vol)), conditional_vol, label='Historical Volatility')
        
        # Plot forecasted volatility
        ax.plot(
            range(len(conditional_vol), len(conditional_vol) + forecast_horizon),
            forecast_vol,
            'r--',
            label='Forecasted Volatility'
        )
        
        # Add vertical line to separate historical and forecasted data
        ax.axvline(x=len(conditional_vol), color='k', linestyle='--')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.set_title(f'{self.vol}({self.p},{self.q}) Volatility Forecast')
        ax.legend()
        
        return fig
        """