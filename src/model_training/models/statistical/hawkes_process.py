"""
Hawkes Process model for point process modeling of market events.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.optimize import minimize
from datetime import datetime, timedelta

from src.model_training.domain.entities.training_config import TrainingConfig


class HawkesProcess:
    """Hawkes Process model for point process modeling of market events."""
    
    def __init__(
        self,
        config: TrainingConfig,
        mu: float = 0.1,
        alpha: float = 0.5,
        beta: float = 1.0,
        kernel: str = "exponential"
    ):
        """
        Initialize Hawkes Process model.
        
        Args:
            config: Training configuration
            mu: Background intensity (baseline rate)
            alpha: Excitation parameter (jump size)
            beta: Decay parameter (how quickly events' influence decay)
            kernel: Kernel function ('exponential', 'power_law')
        """
        self.config = config
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.kernel = kernel
        
        # Fitted parameters
        self.fitted_params = None
        
        # Validate parameters
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate model parameters."""
        if self.mu <= 0:
            raise ValueError("Parameter mu must be positive")
        
        if self.alpha < 0:
            raise ValueError("Parameter alpha must be non-negative")
        
        if self.beta <= 0:
            raise ValueError("Parameter beta must be positive")
        
        if self.kernel not in ["exponential", "power_law"]:
            raise ValueError(f"Unsupported kernel function: {self.kernel}")
    
    def kernel_function(self, t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Kernel function defining the impact of past events.
        
        Args:
            t: Time differences
            alpha: Excitation parameter
            beta: Decay parameter
            
        Returns:
            np.ndarray: Kernel values
        """
        if self.kernel == "exponential":
            # Exponential decay: alpha * exp(-beta * t)
            return alpha * np.exp(-beta * t)
        elif self.kernel == "power_law":
            # Power law decay: alpha * (beta + t)^(-1.5)
            return alpha * np.power(beta + t, -1.5)
        else:
            raise ValueError(f"Unsupported kernel function: {self.kernel}")
    
    def intensity(self, t: float, event_times: np.ndarray, alpha: float, beta: float, mu: float) -> float:
        """
        Calculate the intensity function at time t.
        
        Args:
            t: Time point
            event_times: Array of past event times
            alpha: Excitation parameter
            beta: Decay parameter
            mu: Background intensity
            
        Returns:
            float: Intensity at time t
        """
        # Filter events that occurred before time t
        past_events = event_times[event_times < t]
        
        if len(past_events) == 0:
            return mu
        
        # Calculate time differences
        time_diffs = t - past_events
        
        # Apply kernel function to get impact of past events
        past_effects = self.kernel_function(time_diffs, alpha, beta)
        
        # Total intensity is the sum of background intensity and effects of past events
        return mu + np.sum(past_effects)
    
    def log_likelihood(self, params: np.ndarray, event_times: np.ndarray, end_time: float) -> float:
        """
        Calculate the negative log-likelihood for parameter estimation.
        
        Args:
            params: Model parameters [mu, alpha, beta]
            event_times: Array of event times
            end_time: End time of observation period
            
        Returns:
            float: Negative log-likelihood
        """
        mu, alpha, beta = params
        
        # Validate parameters
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf
        
        # Branching ratio stability condition for exponential kernel
        if self.kernel == "exponential" and alpha >= beta:
            return np.inf
        
        # Sort event times
        event_times = np.sort(event_times)
        
        # Initialize log-likelihood
        ll = 0.0
        
        # First term: Sum of log-intensities at event times
        for i, t in enumerate(event_times):
            # Get intensity at event time
            intensity_t = self.intensity(t, event_times[:i], alpha, beta, mu)
            
            # Add log-intensity to log-likelihood
            if intensity_t > 0:
                ll += np.log(intensity_t)
            else:
                return np.inf
        
        # Second term: Integral of intensity function over observation period
        # For exponential kernel, this can be calculated analytically
        if self.kernel == "exponential":
            # Calculate integral term
            integral_term = mu * end_time
            
            # For each event, calculate its contribution to the integral
            for t in event_times:
                remaining_time = end_time - t
                integral_term += alpha / beta * (1 - np.exp(-beta * remaining_time))
        else:
            # For other kernels, use numerical integration (approximation)
            # This is a simple approximation using uniform time grid
            grid_size = 1000
            time_grid = np.linspace(0, end_time, grid_size)
            dt = end_time / (grid_size - 1)
            
            intensity_values = np.zeros(grid_size)
            for i, t in enumerate(time_grid):
                intensity_values[i] = self.intensity(t, event_times, alpha, beta, mu)
            
            integral_term = np.sum(intensity_values) * dt
        
        # Negative log-likelihood
        neg_ll = -ll + integral_term
        
        return neg_ll
    
    def train(
        self, 
        event_times: np.ndarray,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Fit Hawkes Process model to event data.
        
        Args:
            event_times: Array of event times
            hyperparameters: Optional hyperparameters to override defaults
            
        Returns:
            Dictionary of fitted parameters and metrics
        """
        # Update hyperparameters if provided
        if hyperparameters:
            self.mu = hyperparameters.get("mu", self.mu)
            self.alpha = hyperparameters.get("alpha", self.alpha)
            self.beta = hyperparameters.get("beta", self.beta)
            self.kernel = hyperparameters.get("kernel", self.kernel)
            self._validate_params()
        
        # Sort event times
        event_times = np.sort(event_times)
        
        # End time of observation period
        end_time = event_times[-1]
        
        # Initial parameters
        initial_params = np.array([self.mu, self.alpha, self.beta])
        
        # Bounds for parameters (all positive)
        bounds = [(1e-10, None), (0, None), (1e-10, None)]
        
        # Optimize log-likelihood
        result = minimize(
            fun=self.log_likelihood,
            x0=initial_params,
            args=(event_times, end_time),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Check if optimization was successful
        if not result.success:
            print(f"Warning: Optimization did not converge. Message: {result.message}")
        
        # Extract fitted parameters
        mu_fit, alpha_fit, beta_fit = result.x
        
        # Save fitted parameters
        self.fitted_params = {
            "mu": mu_fit,
            "alpha": alpha_fit,
            "beta": beta_fit
        }
        
        # Calculate AIC and BIC
        n = len(event_times)
        k = 3  # Number of parameters (mu, alpha, beta)
        aic = 2 * k + 2 * result.fun
        bic = k * np.log(n) + 2 * result.fun
        
        # Calculate log-likelihood
        log_likelihood = -result.fun
        
        # Return metrics
        return {
            "mu": mu_fit,
            "alpha": alpha_fit,
            "beta": beta_fit,
            "log_likelihood": log_likelihood,
            "aic": aic,
            "bic": bic,
            "converged": result.success
        }
    
    def simulate(
        self, 
        T: float, 
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate events from the Hawkes Process using Ogata's thinning algorithm.
        
        Args:
            T: End time of simulation period
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Simulated event times
        """
        if self.fitted_params is None:
            # Use initial parameters if not fitted
            mu = self.mu
            alpha = self.alpha
            beta = self.beta
        else:
            # Use fitted parameters
            mu = self.fitted_params["mu"]
            alpha = self.fitted_params["alpha"]
            beta = self.fitted_params["beta"]
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Ogata's thinning algorithm
        # Initialize simulation
        t = 0.0
        event_times = []
        
        # Maximum intensity (upper bound)
        max_lambda = mu
        
        while t < T:
            # Sample waiting time from exponential distribution with rate max_lambda
            dt = np.random.exponential(scale=1/max_lambda)
            
            # Update time
            t += dt
            
            # If past end time, break
            if t >= T:
                break
            
            # Calculate intensity at new time
            intensity_t = self.intensity(t, np.array(event_times), alpha, beta, mu)
            
            # Accept or reject
            if np.random.random() <= intensity_t / max_lambda:
                # Accept event
                event_times.append(t)
                
                # Update maximum intensity
                # For exponential kernel, the intensity is highest right after an event
                if self.kernel == "exponential":
                    max_lambda = self.intensity(t, np.array(event_times[:-1]), alpha, beta, mu) + alpha
                else:
                    # For other kernels, we need a more conservative bound
                    max_lambda = max(max_lambda, intensity_t * 1.2)
        
        return np.array(event_times)
    
    def predict_intensity(
        self,
        future_times: np.ndarray,
        past_events: np.ndarray
    ) -> np.ndarray:
        """
        Predict intensity function at future times based on past events.
        
        Args:
            future_times: Array of future time points
            past_events: Array of past event times
            
        Returns:
            np.ndarray: Predicted intensity at each future time
        """
        if self.fitted_params is None:
            raise ValueError("Model not trained yet")
        
        # Use fitted parameters
        mu = self.fitted_params["mu"]
        alpha = self.fitted_params["alpha"]
        beta = self.fitted_params["beta"]
        
        # Calculate intensity at each future time
        intensity = np.zeros(len(future_times))
        for i, t in enumerate(future_times):
            intensity[i] = self.intensity(t, past_events, alpha, beta, mu)
        
        return intensity
    
    def expected_events(
        self,
        t_start: float,
        t_end: float,
        past_events: np.ndarray,
        resolution: int = 1000
    ) -> float:
        """
        Calculate expected number of events in a future time interval.
        
        Args:
            t_start: Start time of prediction interval
            t_end: End time of prediction interval
            past_events: Array of past event times
            resolution: Number of points for numerical integration
            
        Returns:
            float: Expected number of events
        """
        if self.fitted_params is None:
            raise ValueError("Model not trained yet")
        
        # Generate time grid for numerical integration
        time_grid = np.linspace(t_start, t_end, resolution)
        dt = (t_end - t_start) / (resolution - 1)
        
        # Calculate intensity at each grid point
        intensity = self.predict_intensity(time_grid, past_events)
        
        # Integrate intensity to get expected number of events
        expected_count = np.sum(intensity) * dt
        
        return expected_count
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        # Save model parameters and state
        model_state = {
            "mu": self.mu,
            "alpha": self.alpha,
            "beta": self.beta,
            "kernel": self.kernel,
            "fitted_params": self.fitted_params,
            "config": self.config.to_dict()
        }
        
        # Save model
        joblib.dump(model_state, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        # Load model state
        model_state = joblib.load(filepath)
        
        # Set model attributes
        self.mu = model_state["mu"]
        self.alpha = model_state["alpha"]
        self.beta = model_state["beta"]
        self.kernel = model_state["kernel"]
        self.fitted_params = model_state["fitted_params"]
        
        # Load config
        self.config = TrainingConfig.from_dict(model_state["config"])
        
        # Validate parameters
        self._validate_params()
    
    def plot_intensity(
        self,
        event_times: np.ndarray,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        resolution: int = 1000,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot intensity function and events.
        
        Args:
            event_times: Array of event times
            t_start: Start time for plotting (default: min event time)
            t_end: End time for plotting (default: max event time)
            resolution: Number of points for intensity function plotting
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if t_start is None:
            t_start = min(event_times) if len(event_times) > 0 else 0
        
        if t_end is None:
            t_end = max(event_times) if len(event_times) > 0 else 10
        
        # Generate time grid for intensity function
        time_grid = np.linspace(t_start, t_end, resolution)
        
        # Use fitted parameters if available, otherwise use initial parameters
        if self.fitted_params is not None:
            mu = self.fitted_params["mu"]
            alpha = self.fitted_params["alpha"]
            beta = self.fitted_params["beta"]
        else:
            mu = self.mu
            alpha = self.alpha
            beta = self.beta
        
        # Calculate intensity at each grid point
        intensity = np.zeros(len(time_grid))
        for i, t in enumerate(time_grid):
            intensity[i] = self.intensity(t, event_times, alpha, beta, mu)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot intensity function
        ax.plot(time_grid, intensity, 'b-', label='Intensity')
        
        # Plot background intensity
        ax.axhline(y=mu, color='g', linestyle='--', label='Background Rate (μ)')
        
        # Plot events as vertical lines
        for t in event_times:
            if t_start <= t <= t_end:
                ax.axvline(x=t, color='r', alpha=0.3)
        
        # Plot events as ticks on x-axis
        ax.plot(event_times, np.zeros(len(event_times)), 'ro', alpha=0.5, label='Events')
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Intensity')
        title = f'Hawkes Process Intensity Function\n'
        title += f'μ={mu:.4f}, α={alpha:.4f}, β={beta:.4f}, Kernel: {self.kernel}'
        ax.set_title(title)
        ax.legend()
        
        return fig
    
    def branching_ratio(self) -> float:
        """
        Calculate the branching ratio (average number of triggered events per event).
        
        Returns:
            float: Branching ratio
        """
        if self.fitted_params is not None:
            alpha = self.fitted_params["alpha"]
            beta = self.fitted_params["beta"]
        else:
            alpha = self.alpha
            beta = self.beta
        
        if self.kernel == "exponential":
            # For exponential kernel, branching ratio is alpha/beta
            return alpha / beta
        elif self.kernel == "power_law":
            # For power law kernel with exponent -1.5, the branching ratio is harder to calculate
            # Approximation:
            return alpha * 2.0  # This is a rough approximation
        else:
            raise ValueError(f"Branching ratio calculation not implemented for kernel: {self.kernel}")
    
    def is_stationary(self) -> bool:
        """
        Check if the process is stationary (branching ratio < 1).
        
        Returns:
            bool: True if stationary, False otherwise
        """
        return self.branching_ratio() < 1.0