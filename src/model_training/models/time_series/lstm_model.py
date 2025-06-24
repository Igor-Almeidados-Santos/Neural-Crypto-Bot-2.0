"""
Long Short-Term Memory (LSTM) model for time series forecasting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from model_training.domain.entities.training_config import TrainingConfig


class LSTMModel(nn.Module):
    """LSTM model for time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_layers: Number of LSTM layers
            output_dim: Number of output features
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, output_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # We only need the last time step's output for prediction
        if self.bidirectional:
            # For bidirectional, we concatenate the last forward and backward outputs
            last_out = lstm_out[:, -1, :]
        else:
            # For unidirectional, we just use the last output
            last_out = lstm_out[:, -1, :]
        
        # Apply dropout
        last_out = self.dropout(last_out)
        
        # Apply output layer
        output = self.fc(last_out)
        
        return output


class LSTMLightningModule(pl.LightningModule):
    """PyTorch Lightning module for LSTM model."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.2,
        bidirectional: bool = False,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        loss_function: str = "mse",
        l1_regularization: float = 0.0,
        l2_regularization: float = 0.0
    ):
        """
        Initialize LSTM Lightning module.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            num_layers: Number of LSTM layers
            output_dim: Number of output features
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
            learning_rate: Learning rate
            optimizer: Optimizer to use ('adam', 'sgd', 'rmsprop')
            loss_function: Loss function to use ('mse', 'mae', 'huber')
            l1_regularization: L1 regularization strength
            l2_regularization: L2 regularization strength
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Set loss function
        if loss_function == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_function == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_function == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")
        
        # Regularization
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Add regularization
        if self.l1_regularization > 0:
            l1_reg = sum(p.abs().sum() for p in self.parameters())
            loss += self.l1_regularization * l1_reg
            
        if self.l2_regularization > 0:
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            loss += self.l2_regularization * l2_reg
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Calculate additional metrics
        mse = nn.MSELoss()(y_hat, y)
        mae = nn.L1Loss()(y_hat, y)
        
        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", mse)
        self.log("val_mae", mae)
        
        return {"val_loss": loss, "val_mse": mse, "val_mae": mae}
    
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Calculate additional metrics
        mse = nn.MSELoss()(y_hat, y)
        mae = nn.L1Loss()(y_hat, y)
        
        # Log metrics
        self.log("test_loss", loss)
        self.log("test_mse", mse)
        self.log("test_mae", mae)
        
        return {"test_loss": loss, "test_mse": mse, "test_mae": mae}
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        if self.hparams.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "rmsprop":
            return optim.RMSprop(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch
        return self(x)


class LSTMTimeSeriesForecaster:
    """LSTM-based time series forecaster for cryptocurrency price prediction."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize LSTM time series forecaster.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.trainer = None
        self.input_dim = None
        self.output_dim = None
    
    def preprocess_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess data for LSTM model.
        
        Args:
            data: Input data with shape (n_samples, n_features)
            
        Returns:
            Tuple of X (input sequences) and y (target values)
        """
        n_samples, n_features = data.shape
        sequence_length = self.config.sequence_length
        
        if sequence_length is None:
            raise ValueError("Sequence length must be specified in config")
        
        # Create sequences
        X, y = [], []
        for i in range(n_samples - sequence_length):
            X.append(data[i:i+sequence_length, :])
            # For simplicity, we're predicting the next value of the target feature
            # In a real implementation, you'd extract the target feature index
            target_idx = list(self.config.features).index(self.config.target)
            y.append(data[i+sequence_length, target_idx])
        
        return np.array(X), np.array(y).reshape(-1, 1)
    
    def create_dataloaders(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            X_val: Validation input sequences
            y_val: Validation target values
            X_test: Test input sequences
            y_test: Test target values
            
        Returns:
            Training, validation, and test DataLoaders
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        return train_loader, val_loader, test_loader
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader,
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Train LSTM model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            hyperparameters: Model hyperparameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract hyperparameters with defaults
        hidden_dim = hyperparameters.get("hidden_dim", 64)
        num_layers = hyperparameters.get("num_layers", 2)
        dropout = hyperparameters.get("dropout", 0.2)
        bidirectional = hyperparameters.get("bidirectional", False)
        learning_rate = hyperparameters.get("learning_rate", self.config.learning_rate)
        l1_regularization = hyperparameters.get("l1_regularization", 0.0)
        l2_regularization = hyperparameters.get("l2_regularization", 0.0)
        
        # Create model
        self.model = LSTMLightningModule(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=self.output_dim,
            dropout=dropout,
            bidirectional=bidirectional,
            learning_rate=learning_rate,
            optimizer=self.config.optimizer,
            loss_function=self.config.loss_function,
            l1_regularization=l1_regularization,
            l2_regularization=l2_regularization
        )
        
        # Configure callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                mode="min"
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="{epoch}-{val_loss:.4f}"
            )
        ]
        
        # Configure logger
        logger = TensorBoardLogger("lightning_logs", name=self.config.experiment_name or "lstm_model")
        
        # Configure trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            callbacks=callbacks,
            logger=logger,
            accelerator="gpu" if self.config.use_gpu and torch.cuda.is_available() else "cpu",
            devices=1,
            precision=16 if self.config.mixed_precision else 32,
        )
        
        # Train model
        self.trainer.fit(self.model, train_loader, val_loader)
        
        # Evaluate on test set
        test_results = self.trainer.test(self.model, test_loader, verbose=True)[0]
        
        # Return evaluation metrics
        return {
            "mse": test_results["test_mse"].item(),
            "mae": test_results["test_mae"].item(),
            "loss": test_results["test_loss"].item()
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Save model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "hyperparameters": self.model.hparams,
            "config": self.config.to_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        # Load checkpoint
        checkpoint = torch.load(filepath)
        
        # Set model attributes
        self.input_dim = checkpoint["input_dim"]
        self.output_dim = checkpoint["output_dim"]
        
        # Create model
        self.model = LSTMLightningModule(**checkpoint["hyperparameters"])
        
        # Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, sequences: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            sequences: Input sequences with shape (n_samples, sequence_length, n_features)
            
        Returns:
            Predictions with shape (n_samples, output_dim)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to PyTorch tensor
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sequences_tensor).numpy()
        
        return predictions