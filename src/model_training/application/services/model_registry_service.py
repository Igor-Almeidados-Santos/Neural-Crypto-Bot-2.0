"""
Service for managing the model registry.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import json
import shutil
from datetime import datetime
import pandas as pd

from src.model_training.domain.entities.model import Model, ModelStatus, ModelType


logger = logging.getLogger(__name__)


class ModelRegistryService:
    """Service for managing the model registry."""
    
    def __init__(self, registry_dir: str):
        """
        Initialize model registry service.
        
        Args:
            registry_dir: Directory for the model registry
        """
        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.metadata_file = os.path.join(registry_dir, "metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize metadata file if it doesn't exist
        if not os.path.exists(self.metadata_file):
            self._initialize_metadata()
    
    def _initialize_metadata(self) -> None:
        """Initialize empty metadata file."""
        metadata = {
            "models": {},
            "last_updated": datetime.utcnow().isoformat()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to file."""
        metadata["last_updated"] = datetime.utcnow().isoformat()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def register_model(self, model: Model) -> None:
        """
        Register a model in the registry.
        
        Args:
            model: Model to register
        """
        # Load current metadata
        metadata = self._load_metadata()
        
        # Check if model already exists
        if model.id in metadata["models"]:
            logger.warning(f"Model {model.id} already exists in registry, updating")
        
        # Create model directory
        model_dir = os.path.join(self.models_dir, model.id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model metadata
        model_metadata = model.to_dict()
        model_metadata_file = os.path.join(model_dir, "metadata.json")
        with open(model_metadata_file, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # If model has artifacts, create a symbolic link or copy
        if model.model_artifacts_path and os.path.exists(model.model_artifacts_path):
            artifacts_dir = os.path.join(model_dir, "artifacts")
            os.makedirs(artifacts_dir, exist_ok=True)
            
            # Get artifact filename
            artifact_filename = os.path.basename(model.model_artifacts_path)
            artifact_dest = os.path.join(artifacts_dir, artifact_filename)
            
            # Copy artifact file
            shutil.copy2(model.model_artifacts_path, artifact_dest)
            
            # Update model metadata
            model_metadata["model_artifacts_path"] = artifact_dest
        
        # Update registry metadata
        metadata["models"][model.id] = {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "type": model.type.value,
            "status": model.status.name,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat(),
            "metadata_path": model_metadata_file
        }
        
        # Save updated metadata
        self._save_metadata(metadata)
        
        logger.info(f"Model {model.name} version {model.version} registered with ID {model.id}")
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """
        Get a model from the registry by ID.
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            Model if found, None otherwise
        """
        # Load metadata
        metadata = self._load_metadata()
        
        # Check if model exists
        if model_id not in metadata["models"]:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        
        # Get model metadata file path
        model_metadata_path = metadata["models"][model_id]["metadata_path"]
        
        # Load model metadata
        with open(model_metadata_path, 'r') as f:
            model_metadata = json.load(f)
        
        # Create and return model entity
        return Model.from_dict(model_metadata)
    
    def get_latest_model(self, model_name: str, status: Optional[ModelStatus] = None) -> Optional[Model]:
        """
        Get the latest version of a model by name and optional status.
        
        Args:
            model_name: Name of the model
            status: Optional status filter
            
        Returns:
            Latest model if found, None otherwise
        """
        # Load metadata
        metadata = self._load_metadata()
        
        # Filter models by name and status
        matching_models = []
        for model_id, model_info in metadata["models"].items():
            if model_info["name"] == model_name:
                if status is None or model_info["status"] == status.name:
                    matching_models.append(model_info)
        
        if not matching_models:
            logger.warning(f"No models found with name {model_name}")
            return None
        
        # Sort by version (assuming version format is sortable)
        matching_models.sort(key=lambda x: x["version"], reverse=True)
        
        # Get latest model
        latest_model_info = matching_models[0]
        
        # Load model metadata
        with open(latest_model_info["metadata_path"], 'r') as f:
            model_metadata = json.load(f)
        
        # Create and return model entity
        return Model.from_dict(model_metadata)
    
    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry with optional filtering.
        
        Args:
            model_type: Optional model type filter
            status: Optional status filter
            tags: Optional tags filter (models must have all tags)
            
        Returns:
            List of model summaries
        """
        # Load metadata
        metadata = self._load_metadata()
        
        # Filter models
        filtered_models = []
        for model_id, model_info in metadata["models"].items():
            # Apply type filter
            if model_type is not None and model_info["type"] != model_type.value:
                continue
            
            # Apply status filter
            if status is not None and model_info["status"] != status.name:
                continue
            
            # Apply tags filter (need to load full metadata)
            if tags is not None:
                # Load model metadata
                with open(model_info["metadata_path"], 'r') as f:
                    model_metadata = json.load(f)
                
                # Check if model has all required tags
                model_tags = model_metadata.get("tags", [])
                if not all(tag in model_tags for tag in tags):
                    continue
            
            # Add to filtered list
            filtered_models.append(model_info)
        
        return filtered_models
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: ID of the model to delete
            
        Returns:
            True if model was deleted, False otherwise
        """
        # Load metadata
        metadata = self._load_metadata()
        
        # Check if model exists
        if model_id not in metadata["models"]:
            logger.warning(f"Model {model_id} not found in registry")
            return False
        
        # Get model directory
        model_dir = os.path.join(self.models_dir, model_id)
        
        # Delete model directory
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        
        # Remove from metadata
        del metadata["models"][model_id]
        
        # Save updated metadata
        self._save_metadata(metadata)
        
        logger.info(f"Model {model_id} deleted from registry")
        return True
    
    def update_model_status(self, model_id: str, status: ModelStatus) -> bool:
        """
        Update the status of a model in the registry.
        
        Args:
            model_id: ID of the model to update
            status: New status
            
        Returns:
            True if model was updated, False otherwise
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return False
        
        # Update status
        model.update_status(status)
        
        # Re-register model to update metadata
        self.register_model(model)
        
        logger.info(f"Model {model_id} status updated to {status.name}")
        return True
    
    def get_model_performance(self, model_id: str) -> Dict[str, float]:
        """
        Get performance metrics for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of performance metrics
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return {}
        
        # Return performance summary
        return model.performance_summary
    
    def compare_models(self, model_ids: List[str], metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare multiple models based on performance metrics.
        
        Args:
            model_ids: List of model IDs to compare
            metrics: Optional list of metrics to compare (default: all metrics)
            
        Returns:
            DataFrame with model comparison
        """
        # Get models
        models = []
        for model_id in model_ids:
            model = self.get_model(model_id)
            if model is not None:
                models.append(model)
        
        if not models:
            return pd.DataFrame()
        
        # Create comparison data
        comparison_data = []
        for model in models:
            model_data = {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "type": model.type.value,
                "status": model.status.name,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat()
            }
            
            # Add performance metrics
            for metric_name, metric_value in model.performance_summary.items():
                if metrics is None or metric_name in metrics:
                    model_data[metric_name] = metric_value
            
            comparison_data.append(model_data)
        
        # Create DataFrame
        return pd.DataFrame(comparison_data)
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """
        Get lineage information for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary with model lineage information
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return {}
        
        # Initialize lineage
        lineage = {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "type": model.type.value,
            "status": model.status.name,
            "created_at": model.created_at.isoformat()
        }
        
        # Check if model has a base model (for transfer learning)
        base_model_id = model.training_config.base_model_id
        if base_model_id:
            base_model = self.get_model(base_model_id)
            if base_model:
                lineage["base_model"] = {
                    "id": base_model.id,
                    "name": base_model.name,
                    "version": base_model.version,
                    "type": base_model.type.value
                }
                
                # Recursively get base model's lineage
                base_lineage = self.get_model_lineage(base_model_id)
                if base_lineage and "base_model" in base_lineage:
                    lineage["base_model"]["base_model"] = base_lineage["base_model"]
        
        return lineage
    
    def export_model(self, model_id: str, export_dir: str) -> bool:
        """
        Export a model to a directory.
        
        Args:
            model_id: ID of the model to export
            export_dir: Directory to export to
            
        Returns:
            True if model was exported, False otherwise
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return False
        
        # Create export directory
        os.makedirs(export_dir, exist_ok=True)
        
        # Export model metadata
        model_metadata = model.to_dict()
        with open(os.path.join(export_dir, "metadata.json"), 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Export model artifacts if available
        if model.model_artifacts_path and os.path.exists(model.model_artifacts_path):
            artifact_filename = os.path.basename(model.model_artifacts_path)
            artifact_dest = os.path.join(export_dir, artifact_filename)
            shutil.copy2(model.model_artifacts_path, artifact_dest)
        
        logger.info(f"Model {model_id} exported to {export_dir}")
        return True
    
    def import_model(self, import_dir: str) -> Optional[str]:
        """
        Import a model from a directory.
        
        Args:
            import_dir: Directory to import from
            
        Returns:
            ID of the imported model if successful, None otherwise
        """
        # Check if metadata file exists
        metadata_file = os.path.join(import_dir, "metadata.json")
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file not found in {import_dir}")
            return None
        
        # Load model metadata
        with open(metadata_file, 'r') as f:
            model_metadata = json.load(f)
        
        # Create model entity
        model = Model.from_dict(model_metadata)
        
        # Check for model artifacts
        artifact_path = model.model_artifacts_path
        if artifact_path:
            artifact_filename = os.path.basename(artifact_path)
            import_artifact_path = os.path.join(import_dir, artifact_filename)
            
            if os.path.exists(import_artifact_path):
                # Create a temporary artifact path
                temp_artifacts_dir = os.path.join(self.registry_dir, "temp_artifacts")
                os.makedirs(temp_artifacts_dir, exist_ok=True)
                
                temp_artifact_path = os.path.join(temp_artifacts_dir, artifact_filename)
                shutil.copy2(import_artifact_path, temp_artifact_path)
                
                # Update model artifact path
                model.model_artifacts_path = temp_artifact_path
            else:
                logger.warning(f"Artifact file {artifact_filename} not found in {import_dir}")
                model.model_artifacts_path = None
        
        # Register model
        self.register_model(model)
        
        logger.info(f"Model {model.name} version {model.version} imported with ID {model.id}")
        return model.id
    
    def tag_model(self, model_id: str, tags: List[str]) -> bool:
        """
        Add tags to a model.
        
        Args:
            model_id: ID of the model
            tags: Tags to add
            
        Returns:
            True if model was tagged, False otherwise
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return False
        
        # Add tags
        model.tags.extend([tag for tag in tags if tag not in model.tags])
        
        # Re-register model to update metadata
        self.register_model(model)
        
        logger.info(f"Model {model_id} tagged with {tags}")
        return True
    
    def untag_model(self, model_id: str, tags: List[str]) -> bool:
        """
        Remove tags from a model.
        
        Args:
            model_id: ID of the model
            tags: Tags to remove
            
        Returns:
            True if model was untagged, False otherwise
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return False
        
        # Remove tags
        model.tags = [tag for tag in model.tags if tag not in tags]
        
        # Re-register model to update metadata
        self.register_model(model)
        
        logger.info(f"Tags {tags} removed from model {model_id}")
        return True
    
    def add_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Add metadata to a model.
        
        Args:
            model_id: ID of the model
            metadata: Metadata to add
            
        Returns:
            True if metadata was added, False otherwise
        """
        # Get model
        model = self.get_model(model_id)
        if model is None:
            return False
        
        # Add metadata
        model.metadata.update(metadata)
        
        # Re-register model to update metadata
        self.register_model(model)
        
        logger.info(f"Metadata added to model {model_id}")
        return True