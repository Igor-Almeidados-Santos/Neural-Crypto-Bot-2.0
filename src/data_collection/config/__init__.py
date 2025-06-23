"""
Configuration Package

Provides centralized configuration management for the data collection module.
"""

import os
from pathlib import Path
from typing import Optional
from .settings import DataCollectionSettings, Environment

# Global settings instance
_settings: Optional[DataCollectionSettings] = None


def get_settings() -> DataCollectionSettings:
    """Get the global settings instance"""
    global _settings
    
    if _settings is None:
        _settings = load_settings()
    
    return _settings


def load_settings(config_path: Optional[str] = None) -> DataCollectionSettings:
    """
    Load settings from configuration file or environment variables.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        DataCollectionSettings instance
    """
    # Try to load from file first
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            if config_file.suffix == '.yaml' or config_file.suffix == '.yml':
                return DataCollectionSettings.from_yaml(config_file)
            elif config_file.suffix == '.json':
                return DataCollectionSettings.from_json(config_file)
    
    # Try default config files
    possible_files = [
        "config/data_collection.yaml",
        "config/data_collection.yml",
        "config/data_collection.json",
        "data_collection.yaml",
        "data_collection.yml",
        "data_collection.json"
    ]
    
    for file_path in possible_files:
        if Path(file_path).exists():
            if file_path.endswith(('.yaml', '.yml')):
                return DataCollectionSettings.from_yaml(file_path)
            elif file_path.endswith('.json'):
                return DataCollectionSettings.from_json(file_path)
    
    # Fall back to environment variables
    return DataCollectionSettings()


def reload_settings(config_path: Optional[str] = None) -> DataCollectionSettings:
    """Reload settings from configuration"""
    global _settings
    _settings = load_settings(config_path)
    return _settings