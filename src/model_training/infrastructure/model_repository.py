"""
Repository for storing and retrieving machine learning models.
"""
from typing import Any, Dict, List, Optional, Union
import logging
import os
import json
import shutil
import joblib
import torch
import pickle
from datetime import datetime

from src.model_training.domain.entities.model import Model, ModelType
from src.model_training.models.time_series.lstm_model import LSTMTimeSeriesForecaster
from src.model_training.models.time_series.gru_model import GRUTimeSeriesForecaster
from src.model_training.models.time_series.transformer_model import TransformerTimeSeriesForecaster
from src.model_training.models.statistical.garch_model import GARCHModel
from src.model_training.models.statistical.hawkes_process import HawkesProcess


logger = logging.getLogger(__name__)


class ModelRepository:
    """Repository for storing and retrieving machine learning models."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize model repository.
        
        Args:
            storage_dir: Directory for storing models
        """
        self.storage_dir = storage_dir
        
        # Create directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
    
    def save_model(
        self,
        model_id: str,
        model_trainer: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a trained model to the repository.
        
        Args:
            model_id: Unique ID for the model
            model_trainer: Trained model trainer instance
            metadata: Optional metadata to store with the model
            
        Returns:
            Path to the saved model
        """
        # Create model directory
        model_dir = os.path.join(self.storage_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Determine model type and save appropriately
        model_path = None
        
        if isinstance(model_trainer, (LSTMTimeSeriesForecaster, GRUTimeSeriesForecaster, TransformerTimeSeriesForecaster)):
            # For PyTorch models
            model_path = os.path.join(model_dir, f"{model_id}.pt")
            model_trainer.save(model_path)
        elif isinstance(model_trainer, GARCHModel):
            # For GARCH models
            model_path = os.path.join(model_dir, f"{model_id}.joblib")
            model_trainer.save(model_path)
        elif isinstance(model_trainer, HawkesProcess):
            # For Hawkes Process models
            model_path = os.path.join(model_dir, f"{model_id}.joblib")
            model_trainer.save(model_path)
        else:
            # For other models, use pickle as a fallback
            model_path = os.path.join(model_dir, f"{model_id}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_trainer, f)
        
        # Save metadata if provided
        if metadata:
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Model {model_id} saved to {model_path}")
        
        return model_path
    
    def load_model(
        self,
        model_id: str,
        model_trainer: Any,
        artifacts_path: Optional[str] = None
    ) -> Any:
        """
        Load a model from the repository.
        
        Args:
            model_id: ID of the model to load
            model_trainer: Empty model trainer instance
            artifacts_path: Optional path to model artifacts (default: auto-detect)
            
        Returns:
            Loaded model trainer
        """
        # Determine model path
        if artifacts_path is None:
            model_dir = os.path.join(self.storage_dir, model_id)
            possible_extensions = [".pt", ".joblib", ".pkl"]
            
            for ext in possible_extensions:
                model_path = os.path.join(model_dir, f"{model_id}{ext}")
                if os.path.exists(model_path):
                    artifacts_path = model_path
                    break
            
            if artifacts_path is None:
                raise FileNotFoundError(f"No model artifacts found for model {model_id}")
        
        # Load model based on type
        if isinstance(model_trainer, (LSTMTimeSeriesForecaster, GRUTimeSeriesForecaster, TransformerTimeSeriesForecaster)):
            # For PyTorch models
            model_trainer.load(artifacts_path)
        elif isinstance(model_trainer, GARCHModel):
            # For GARCH models
            model_trainer.load(artifacts_path)
        elif isinstance(model_trainer, HawkesProcess):
            # For Hawkes Process models
            model_trainer.load(artifacts_path)
        else:
            # For other models, use pickle as a fallback
            with open(artifacts_path, 'rb') as f:
                loaded_model = pickle.load(f)
                
                # Transfer attributes to the provided model_trainer
                for attr, value in vars(loaded_model).items():
                    setattr(model_trainer, attr, value)
        
        logger.info(f"Model {model_id} loaded from {artifacts_path}")
        
        return model_trainer
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the repository.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if model was deleted, False otherwise
        """
        model_dir = os.path.join(self.storage_dir, model_id)
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            return False
        
        # Delete directory
        shutil.rmtree(model_dir)
        
        logger.info(f"Model {model_id} deleted")
        
        return True
    
    def list_models(self) -> List[str]:
        """
        List all models in the repository.
        
        Returns:
            List of model IDs
        """
        # List all subdirectories in storage_dir
        try:
            return [d for d in os.listdir(self.storage_dir) 
                   if os.path.isdir(os.path.join(self.storage_dir, d))]
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of model information if found, None otherwise
        """
        model_dir = os.path.join(self.storage_dir, model_id)
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            return None
        
        # Check for metadata file
        metadata_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Get list of model files
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        
        # Find model file
        model_file = None
        for f in model_files:
            if f.endswith((".pt", ".joblib", ".pkl")):
                model_file = f
                break
        
        # Prepare info
        info = {
            "id": model_id,
            "path": model_dir,
            "model_file": model_file,
            "model_path": os.path.join(model_dir, model_file) if model_file else None,
            "files": model_files,
            "created_at": datetime.fromtimestamp(os.path.getctime(model_dir)).isoformat(),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(model_dir)).isoformat()
        }
        
        # Add metadata
        info.update(metadata)
        
        return info
    
    def export_model(self, model_id: str, export_path: str) -> bool:
        """
        Export a model to a file or directory.
        
        Args:
            model_id: ID of the model to export
            export_path: Path to export to
            
        Returns:
            True if model was exported, False otherwise
        """
        model_dir = os.path.join(self.storage_dir, model_id)
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            return False
        
        if os.path.isdir(export_path):
            # Export to directory
            for item in os.listdir(model_dir):
                s = os.path.join(model_dir, item)
                d = os.path.join(export_path, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
            logger.info(f"Model {model_id} exported to directory {export_path}")
            return True
        else:
            # Export as a single file (archive)
            try:
                shutil.make_archive(export_path, 'zip', model_dir)
                logger.info(f"Model {model_id} exported to archive {export_path}.zip")
                return True
            except Exception as e:
                logger.error(f"Error exporting model: {str(e)}")
                return False
    
    def import_model(self, model_id: str, import_path: str) -> bool:
        """
        Import a model from a file or directory.
        
        Args:
            model_id: ID to assign to the imported model
            import_path: Path to import from
            
        Returns:
            True if model was imported, False otherwise
        """
        model_dir = os.path.join(self.storage_dir, model_id)
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        if os.path.isdir(import_path):
            # Import from directory
            for item in os.listdir(import_path):
                s = os.path.join(import_path, item)
                d = os.path.join(model_dir, item)
                if os.path.isfile(s):
                    shutil.copy2(s, d)
            logger.info(f"Model imported from directory {import_path} as {model_id}")
            return True
        elif import_path.endswith(".zip"):
            # Import from zip archive
            try:
                shutil.unpack_archive(import_path, model_dir, 'zip')
                logger.info(f"Model imported from archive {import_path} as {model_id}")
                return True
            except Exception as e:
                logger.error(f"Error importing model: {str(e)}")
                return False
        else:
            logger.error(f"Unsupported import path: {import_path}")
            return False
    
    def copy_model(self, source_id: str, target_id: str) -> bool:
        """
        Copy a model within the repository.
        
        Args:
            source_id: ID of the source model
            target_id: ID to assign to the copied model
            
        Returns:
            True if model was copied, False otherwise
        """
        source_dir = os.path.join(self.storage_dir, source_id)
        target_dir = os.path.join(self.storage_dir, target_id)
        
        if not os.path.exists(source_dir):
            logger.warning(f"Source model directory not found: {source_dir}")
            return False
        
        if os.path.exists(target_dir):
            logger.warning(f"Target model directory already exists: {target_dir}")
            return False
        
        # Copy directory
        shutil.copytree(source_dir, target_dir)
        
        # Update model ID in any metadata
        metadata_path = os.path.join(target_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if "id" in metadata:
                metadata["id"] = target_id
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Rename model file if needed
        for item in os.listdir(target_dir):
            if item.startswith(source_id) and item.endswith((".pt", ".joblib", ".pkl")):
                os.rename(
                    os.path.join(target_dir, item),
                    os.path.join(target_dir, item.replace(source_id, target_id))
                )
        
        logger.info(f"Model {source_id} copied to {target_id}")
        
        return True
    
    def get_model_size(self, model_id: str) -> Optional[int]:
        """
        Get the size of a model in bytes.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Size in bytes if model found, None otherwise
        """
        model_dir = os.path.join(self.storage_dir, model_id)
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            return None
        
        # Calculate total size
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
        
        return total_size
    
    def get_model_file(self, model_id: str) -> Optional[str]:
        """
        Get the path to the main model file.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Path to model file if found, None otherwise
        """
        model_dir = os.path.join(self.storage_dir, model_id)
        
        if not os.path.exists(model_dir):
            logger.warning(f"Model directory not found: {model_dir}")
            return None
        
        # Find model file
        for item in os.listdir(model_dir):
            if item.endswith((".pt", ".joblib", ".pkl")):
                return os.path.join(model_dir, item)
        
        return None