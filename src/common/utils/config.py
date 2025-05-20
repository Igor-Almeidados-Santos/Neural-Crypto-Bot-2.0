"""
Módulo de configuração para o Neural Crypto Bot.

Este módulo fornece classes e funções para carregar e gerenciar configurações
a partir de variáveis de ambiente e arquivos de configuração.
"""
# src/common/utils/config.py
import os
import json
from typing import Dict, Any, Optional, Union, TypeVar, Type, cast
from pathlib import Path
import logging
from dotenv import load_dotenv
import yaml

# Define type variable for type hints
T = TypeVar('T')

# Load environment variables from .env file
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

# Cache for config values
_config_cache: Dict[str, Any] = {}

def get_config() -> Dict[str, Any]:
    """Get the application configuration.
    
    Returns:
        A dictionary with configuration values.
    """
    global _config_cache
    
    # Return cached config if available
    if _config_cache:
        return _config_cache
    
    # Build config dictionary
    config = {}
    
    # Load environment variables
    for key, value in os.environ.items():
        config[key] = value
    
    # Load config file if present
    config_path = os.environ.get('CONFIG_FILE')
    if config_path and Path(config_path).exists():
        try:
            file_extension = Path(config_path).suffix.lower()
            
            if file_extension == '.json':
                with open(config_path, 'r') as f:
                    config.update(json.load(f))
            elif file_extension in ('.yaml', '.yml'):
                with open(config_path, 'r') as f:
                    config.update(yaml.safe_load(f))
            else:
                logger.warning(f"Unsupported config file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Cache the config
    _config_cache = config
    
    return config

def get_config_value(key: str, default: Optional[T] = None) -> Union[str, T]:
    """Get a configuration value.
    
    Args:
        key: The key of the value to get.
        default: The default value to return if the key is not found.
        
    Returns:
        The configuration value, or the default value if not found.
    """
    config = get_config()
    return config.get(key, default)

def get_config_int(key: str, default: Optional[int] = None) -> int:
    """Get an integer configuration value.
    
    Args:
        key: The key of the value to get.
        default: The default value to return if the key is not found or cannot be converted to an integer.
        
    Returns:
        The configuration value as an integer, or the default value if not found or cannot be converted.
    """
    value = get_config_value(key, default)
    
    if value is None:
        return cast(int, default)
    
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert config value '{key}' to int: {value}")
        return cast(int, default)

def get_config_float(key: str, default: Optional[float] = None) -> float:
    """Get a float configuration value.
    
    Args:
        key: The key of the value to get.
        default: The default value to return if the key is not found or cannot be converted to a float.
        
    Returns:
        The configuration value as a float, or the default value if not found or cannot be converted.
    """
    value = get_config_value(key, default)
    
    if value is None:
        return cast(float, default)
    
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert config value '{key}' to float: {value}")
        return cast(float, default)

def get_config_bool(key: str, default: Optional[bool] = None) -> bool:
    """Get a boolean configuration value.
    
    Args:
        key: The key of the value to get.
        default: The default value to return if the key is not found or cannot be converted to a boolean.
        
    Returns:
        The configuration value as a boolean, or the default value if not found or cannot be converted.
    """
    value = get_config_value(key, default)
    
    if value is None:
        return cast(bool, default)
    
    if isinstance(value, bool):
        return value
    
    if isinstance(value, (int, float)):
        return bool(value)
    
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 'y', 't')
    
    logger.warning(f"Could not convert config value '{key}' to bool: {value}")
    return cast(bool, default)

def get_config_list(key: str, default: Optional[list] = None, separator: str = ',') -> list:
    """Get a list configuration value.
    
    Args:
        key: The key of the value to get.
        default: The default value to return if the key is not found.
        separator: The separator to use for splitting the string.
        
    Returns:
        The configuration value as a list, or the default value if not found.
    """
    value = get_config_value(key, default)
    
    if value is None:
        return cast(list, default or [])
    
    if isinstance(value, list):
        return value
    
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    
    logger.warning(f"Could not convert config value '{key}' to list: {value}")
    return cast(list, default or [])

def get_config_dict(key: str, default: Optional[dict] = None) -> dict:
    """Get a dictionary configuration value.
    
    Args:
        key: The key of the value to get.
        default: The default value to return if the key is not found.
        
    Returns:
        The configuration value as a dictionary, or the default value if not found.
    """
    value = get_config_value(key, default)
    
    if value is None:
        return cast(dict, default or {})
    
    if isinstance(value, dict):
        return value
    
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Could not convert config value '{key}' to dict: {value}")
            return cast(dict, default or {})
    
    logger.warning(f"Could not convert config value '{key}' to dict: {value}")
    return cast(dict, default or {})

def refresh_config() -> None:
    """Refresh the configuration cache."""
    global _config_cache
    _config_cache = {}