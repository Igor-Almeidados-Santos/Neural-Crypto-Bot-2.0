"""
Main entry point for the model training module.
"""
import os
import logging
import argparse
import pandas as pd
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from model_training.domain.entities.model import Model, ModelStatus, ModelType
from model_training.domain.entities.training_config import TrainingConfig
from model_training.domain.entities.evaluation_metric import EvaluationMetric, MetricType, MetricDirection

from model_training.application.use_cases.train_model_use_case import TrainModelUseCase
from model_training.application.use_cases.evaluate_model_use_case import EvaluateModelUseCase
from model_training.application.services.hyperparameter_optimization_service import HyperparameterOptimizationService
from model_training.application.services.model_registry_service import ModelRegistryService

from model_training.infrastructure.experiment_tracker import ExperimentTracker
from model_training.infrastructure.model_repository import ModelRepository


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)


def setup_infrastructure(base_dir: str) -> Tuple[
    ModelRepository,
    ModelRegistryService,
    HyperparameterOptimizationService,
    ExperimentTracker
]:
    """
    Set up infrastructure components.
    
    Args:
        base_dir: Base directory for infrastructure
        
    Returns:
        Tuple of infrastructure components
    """
    # Create directories
    model_storage_dir = os.path.join(base_dir, "model_storage")
    model_registry_dir = os.path.join(base_dir, "model_registry")
    experiment_dir = os.path.join(base_dir, "experiments")
    
    os.makedirs(model_storage_dir, exist_ok=True)
    os.makedirs(model_registry_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize infrastructure components
    model_repository = ModelRepository(storage_dir=model_storage_dir)
    model_registry_service = ModelRegistryService(registry_dir=model_registry_dir)
    hyperparameter_optimization_service = HyperparameterOptimizationService(
        study_persistence_path=os.path.join(base_dir, "optuna_studies")
    )
    experiment_tracker = ExperimentTracker(
        experiment_dir=experiment_dir,
        backend="mlflow"
    )
    
    return (
        model_repository,
        model_registry_service,
        hyperparameter_optimization_service,
        experiment_tracker
    )


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from file.
    
    Args:
        data_path: Path to data file
        
    Returns:
        Loaded data as DataFrame
    """
    if data_path.endswith(".csv"):
        return pd.read_csv(data_path)
    elif data_path.endswith((".xls", ".xlsx")):
        return pd.read_excel(data_path)
    elif data_path.endswith(".parquet"):
        return pd.read_parquet(data_path)
    elif data_path.endswith(".feather"):
        return pd.read_feather(data_path)
    elif data_path.endswith(".json"):
        return pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def load_training_config(config_path: str) -> TrainingConfig:
    """
    Load training configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Training configuration
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return TrainingConfig.from_dict(config_data)


def train_model(
    training_config: TrainingConfig,
    data: pd.DataFrame,
    infrastructure: Tuple[ModelRepository, ModelRegistryService, HyperparameterOptimizationService, ExperimentTracker],
    optimize_hyperparameters: bool = True,
    n_trials: int = 50,
    timeout: int = 3600,
    register_model: bool = True
) -> Model:
    """
    Train a model with the given configuration and data.
    
    Args:
        training_config: Training configuration
        data: Input data
        infrastructure: Infrastructure components
        optimize_hyperparameters: Whether to optimize hyperparameters
        n_trials: Number of optimization trials
        timeout: Optimization timeout in seconds
        register_model: Whether to register the model
        
    Returns:
        Trained model
    """
    model_repository, model_registry_service, hyperparameter_optimization_service, experiment_tracker = infrastructure
    
    # Create use case
    train_model_use_case = TrainModelUseCase(
        model_repository=model_repository,
        model_registry_service=model_registry_service,
        hyperparameter_optimization_service=hyperparameter_optimization_service,
        experiment_tracker=experiment_tracker
    )
    
    # Train model
    model = train_model_use_case.execute(
        training_config=training_config,
        data=data,
        optimize_hyperparameters=optimize_hyperparameters,
        n_trials=n_trials,
        timeout=timeout,
        register_model=register_model
    )
    
    return model


def evaluate_model(
    model: Model,
    data: pd.DataFrame,
    infrastructure: Tuple[ModelRepository, ModelRegistryService, HyperparameterOptimizationService, ExperimentTracker],
    evaluation_data_split: str = "test",
    custom_metrics: Optional[List[str]] = None
) -> Model:
    """
    Evaluate a trained model.
    
    Args:
        model: Model to evaluate
        data: Evaluation data
        infrastructure: Infrastructure components
        evaluation_data_split: Data split to use for evaluation
        custom_metrics: Optional custom metrics to compute
        
    Returns:
        Updated model with evaluation metrics
    """
    model_repository, _, _, experiment_tracker = infrastructure
    
    # Create use case
    evaluate_model_use_case = EvaluateModelUseCase(
        model_repository=model_repository,
        experiment_tracker=experiment_tracker
    )
    
    # Evaluate model
    updated_model = evaluate_model_use_case.execute(
        model=model,
        data=data,
        evaluation_data_split=evaluation_data_split,
        custom_metrics=custom_metrics
    )
    
    return updated_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Model Training")
    
    # Command options
    parser.add_argument("command", choices=["train", "evaluate", "list", "info"], help="Command to execute")
    
    # Common options
    parser.add_argument("--base-dir", default="./data", help="Base directory for infrastructure")
    
    # Training options
    parser.add_argument("--config", help="Path to training configuration file")
    parser.add_argument("--data", help="Path to data file")
    parser.add_argument("--no-optimize", action="store_true", help="Disable hyperparameter optimization")
    parser.add_argument("--trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Optimization timeout in seconds")
    parser.add_argument("--no-register", action="store_true", help="Disable model registration")
    
    # Evaluation options
    parser.add_argument("--model-id", help="ID of the model to evaluate or get info about")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Data split for evaluation")
    parser.add_argument("--metrics", help="Comma-separated list of custom metrics to compute")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set up infrastructure
    infrastructure = setup_infrastructure(args.base_dir)
    model_repository, model_registry_service, _, _ = infrastructure
    
    if args.command == "train":
        # Check required arguments
        if not args.config:
            logger.error("Training configuration file is required for training")
            return
        
        if not args.data:
            logger.error("Data file is required for training")
            return
        
        # Load data and configuration
        data = load_data(args.data)
        training_config = load_training_config(args.config)
        
        # Train model
        logger.info(f"Training model with configuration from {args.config}")
        model = train_model(
            training_config=training_config,
            data=data,
            infrastructure=infrastructure,
            optimize_hyperparameters=not args.no_optimize,
            n_trials=args.trials,
            timeout=args.timeout,
            register_model=not args.no_register
        )
        
        # Print results
        logger.info(f"Model training completed. Model ID: {model.id}")
        logger.info(f"Model performance: {model.performance_summary}")
        
    elif args.command == "evaluate":
        # Check required arguments
        if not args.model_id:
            logger.error("Model ID is required for evaluation")
            return
        
        if not args.data:
            logger.error("Data file is required for evaluation")
            return
        
        # Load model
        model = model_registry_service.get_model(args.model_id)
        if model is None:
            logger.error(f"Model not found: {args.model_id}")
            return
        
        # Load data
        data = load_data(args.data)
        
        # Parse custom metrics
        custom_metrics = args.metrics.split(",") if args.metrics else None
        
        # Evaluate model
        logger.info(f"Evaluating model {model.id} using {args.split} split")
        updated_model = evaluate_model(
            model=model,
            data=data,
            infrastructure=infrastructure,
            evaluation_data_split=args.split,
            custom_metrics=custom_metrics
        )
        
        # Print results
        logger.info(f"Model evaluation completed. Model ID: {updated_model.id}")
        logger.info(f"Evaluation metrics: {updated_model.performance_summary}")
        
    elif args.command == "list":
        # List models in registry
        models = model_registry_service.list_models()
        
        if not models:
            logger.info("No models found in registry")
            return
        
        # Print models
        logger.info(f"Found {len(models)} models in registry:")
        for i, model_info in enumerate(models, 1):
            logger.info(f"{i}. ID: {model_info['id']}, Name: {model_info['name']}, Version: {model_info['version']}, Status: {model_info['status']}")
        
    elif args.command == "info":
        # Check required arguments
        if not args.model_id:
            logger.error("Model ID is required for model info")
            return
        
        # Get model
        model = model_registry_service.get_model(args.model_id)
        if model is None:
            logger.error(f"Model not found: {args.model_id}")
            return
        
        # Print model info
        logger.info(f"Model ID: {model.id}")
        logger.info(f"Name: {model.name}")
        logger.info(f"Version: {model.version}")
        logger.info(f"Type: {model.type.value}")
        logger.info(f"Status: {model.status.name}")
        logger.info(f"Created at: {model.created_at.isoformat()}")
        logger.info(f"Updated at: {model.updated_at.isoformat()}")
        logger.info(f"Performance: {model.performance_summary}")
        logger.info(f"Tags: {model.tags}")
        logger.info(f"Artifacts path: {model.model_artifacts_path}")
        
        # Print training configuration summary
        config = model.training_config
        logger.info("Training configuration:")
        logger.info(f"  Dataset ID: {config.dataset_id}")
        logger.info(f"  Features: {len(config.features)} features")
        logger.info(f"  Target: {config.target}")
        logger.info(f"  Mode: {config.mode.value}")
        logger.info(f"  Primary metric: {config.primary_metric}")
        logger.info(f"  Assets: {config.assets}")
        logger.info(f"  Timeframe: {config.timeframe}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in main: {str(e)}")
        raise


def create_sample_config():
    """Create a sample training configuration for testing."""
    config = TrainingConfig(
        dataset_id="crypto_btc_1h",
        features=["open", "high", "low", "close", "volume", "rsi_14", "sma_20", "ema_50", "macd", "macd_signal"],
        target="close",
        sequence_length=60,
        forecast_horizon=10,
        batch_size=32,
        epochs=100,
        early_stopping_patience=15,
        learning_rate=0.001,
        optimizer="adam",
        loss_function="mse",
        use_gpu=True,
        normalization="standard",
        primary_metric="mse",
        evaluation_metrics=["mse", "mae", "r2", "directional_accuracy"],
        assets=["BTC/USDT"],
        timeframe="1h",
        experiment_name="btc_price_prediction",
        tags=["cryptocurrency", "bitcoin", "price_prediction", "lstm"]
    )
    
    return config


def run_example():
    """Run an example training workflow."""
    # Set up infrastructure
    infrastructure = setup_infrastructure("./data")
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="H")
    data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.normal(9000, 1000, 1000),
        "high": np.random.normal(9200, 1000, 1000),
        "low": np.random.normal(8800, 1000, 1000),
        "close": np.random.normal(9100, 1000, 1000),
        "volume": np.random.normal(100, 20, 1000),
        "rsi_14": np.random.normal(50, 15, 1000),
        "sma_20": np.random.normal(9000, 800, 1000),
        "ema_50": np.random.normal(9000, 700, 1000),
        "macd": np.random.normal(0, 50, 1000),
        "macd_signal": np.random.normal(0, 40, 1000)
    })
    
    # Ensure price consistency
    for i in range(1, len(data)):
        data.loc[i, "open"] = data.loc[i-1, "close"] * (1 + np.random.normal(0, 0.01))
        data.loc[i, "close"] = data.loc[i, "open"] * (1 + np.random.normal(0, 0.015))
        data.loc[i, "high"] = max(data.loc[i, "open"], data.loc[i, "close"]) * (1 + np.random.normal(0.005, 0.005))
        data.loc[i, "low"] = min(data.loc[i, "open"], data.loc[i, "close"]) * (1 - np.random.normal(0.005, 0.005))
    
    # Save sample data
    os.makedirs("./data/samples", exist_ok=True)
    data.to_csv("./data/samples/btc_1h_sample.csv", index=False)
    
    # Create sample configuration
    config = create_sample_config()
    
    # Save sample configuration
    with open("./data/samples/btc_prediction_config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Train model
    model = train_model(
        training_config=config,
        data=data,
        infrastructure=infrastructure,
        optimize_hyperparameters=True,
        n_trials=10,  # Use fewer trials for example
        timeout=600,  # Use shorter timeout for example
        register_model=True
    )
    
    # Evaluate model
    updated_model = evaluate_model(
        model=model,
        data=data,
        infrastructure=infrastructure,
        evaluation_data_split="test",
        custom_metrics=["sharpe_ratio", "sortino_ratio", "max_drawdown"]
    )
    
    # Print results
    logger.info(f"Example completed. Model ID: {updated_model.id}")
    logger.info(f"Performance: {updated_model.performance_summary}")
    
    return updated_model
