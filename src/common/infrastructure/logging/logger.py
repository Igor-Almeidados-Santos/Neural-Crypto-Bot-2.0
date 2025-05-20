"""
Módulo de logging para o Neural Crypto Bot.

Este módulo fornece funções para configurar e usar o sistema de logging
de forma padronizada em todo o aplicativo.
"""
# src/common/infrastructure/logging/logger.py
import logging
import json
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
from pythonjsonlogger import jsonlogger
import structlog
from structlog.processors import TimeStamper, JSONRenderer
from structlog.contextvars import merge_contextvars
from structlog.stdlib import add_log_level, add_logger_name

from ...utils.config import get_config

# Configure default logging format
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for Python's standard logging module."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """Add custom fields to the log record.
        
        Args:
            log_record: The log record to add fields to.
            record: The original logging.LogRecord instance.
            message_dict: The message dictionary.
        """
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format if not present
        if 'timestamp' not in log_record:
            log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level name
        if 'level' not in log_record:
            log_record['level'] = record.levelname
        
        # Add service name
        log_record['service'] = os.environ.get('SERVICE_NAME', 'neural-crypto-bot')
        
        # Add environment
        log_record['environment'] = os.environ.get('ENVIRONMENT', 'development')
        
        # Add process ID and thread ID
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        
        # Add file name, function name and line number
        log_record['file'] = record.pathname
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno

def configure_logging() -> None:
    """Configure the logging system."""
    config = get_config()
    log_level = getattr(logging, config.get('LOG_LEVEL', 'INFO'))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create a stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    
    # Create formatter
    formatter = CustomJsonFormatter(
        '%(timestamp)s %(level)s %(name)s %(message)s'
    )
    
    # Set formatter
    stdout_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(stdout_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.
    
    Args:
        name: The name of the logger.
        
    Returns:
        A logger instance with the given name.
    """
    logger = logging.getLogger(name)
    
    # Ensure the logger has the correct level
    config = get_config()
    log_level = getattr(logging, config.get('LOG_LEVEL', 'INFO'))
    logger.setLevel(log_level)
    
    return logger

# Configure logging at module import
configure_logging()