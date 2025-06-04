"""
Transformer model for time series forecasting.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.model_training.domain.entities.training_config import TrainingConfig


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    
    This adds positional information to the input embeddings since the
    transformer does not inherently capture sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimensionality of the model
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer model for time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "relu",
        output_dim: int = 1
    ):
        """
        Initialize Transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Dimensionality of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimensionality of feedforward network
            dropout: Dropout rate
            activation: Activation function
            output_dim: Number of output features
        """
        super().__init__()
        
        # Input projection layer to project features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask for transformer
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, mask=mask)
        
        # Use the representation of the last time step for prediction
        x = x[:, -1, :]
        
        # Apply output layer
        x = self.output_layer(x)
        
        return x


class TransformerLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Transformer model."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "relu",
        output_dim: int = 1,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        loss_function: str = "mse",
        l1_regularization: float = 0.0,
        l2_regularization: float = 0.0,
        scheduler: Optional[str] = "cosine"
    ):
        """
        Initialize Transformer Lightning module.
        
        Args:
            input_dim: Number of input features
            d_model: Dimensionality of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimensionality of feedforward network
            dropout: Dropout rate
            activation: Activation function
            output_dim: Number of output features
            learning_rate: Learning rate
            optimizer: Optimizer to use ('adam', 'adamw', 'sgd')
            loss_function: Loss function to use ('mse', 'mae', 'huber')
            l1_regularization: L1 regularization strength
            l2_regularization: L2 regularization strength
            scheduler: Learning rate scheduler ('cosine', 'plateau', 'step', None)
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            output_dim=output_dim
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
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Set optimizer
        if self.hparams.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
        # Set scheduler
        if self.hparams.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=10,
                eta_min=self.hparams.learning_rate / 10
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.hparams.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss"
                }
            }
        elif self.hparams.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer
    
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Prediction step."""
        if isinstance(batch, tuple):
            x = batch[0]
        else:
            x = batch
        return self(x)


class TransformerTimeSeriesForecaster:
    """Transformer-based time series forecaster for cryptocurrency price prediction."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize Transformer time series forecaster.
        
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
        Preprocess data for Transformer model.
        
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
        Train Transformer model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            hyperparameters: Model hyperparameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract hyperparameters with defaults
        d_model = hyperparameters.get("d_model", 64)
        nhead = hyperparameters.get("nhead", 8)
        num_encoder_layers = hyperparameters.get("num_encoder_layers", 3)
        dim_feedforward = hyperparameters.get("dim_feedforward", 256)
        dropout = hyperparameters.get("dropout", 0.1)
        activation = hyperparameters.get("activation", "relu")
        learning_rate = hyperparameters.get("learning_rate", self.config.learning_rate)
        l1_regularization = hyperparameters.get("l1_regularization", 0.0)
        l2_regularization = hyperparameters.get("l2_regularization", 0.0)
        scheduler = hyperparameters.get("scheduler", "cosine")
        
        # Create model
        self.model = TransformerLightningModule(
            input_dim=self.input_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            output_dim=self.output_dim,
            learning_rate=learning_rate,
            optimizer=self.config.optimizer,
            loss_function=self.config.loss_function,
            l1_regularization=l1_regularization,
            l2_regularization=l2_regularization,
            scheduler=scheduler
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
        logger = TensorBoardLogger("lightning_logs", name=self.config.experiment_name or "transformer_model")
        
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
        self.model = TransformerLightningModule(**checkpoint["hyperparameters"])
        
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