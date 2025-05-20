# src/common/application/base_service.py
from abc import ABC
from typing import TypeVar, Generic, Any, Dict, Optional
import logging
from datetime import datetime
import time
from uuid import uuid4

from ..infrastructure.logging.logger import get_logger

T = TypeVar('T')

class BaseService(ABC, Generic[T]):
    """Base class for all application services.
    
    Application services orchestrate and coordinate the execution of use cases
    and domain logic. They typically handle transaction management, logging,
    event publication, and other cross-cutting concerns.
    
    Attributes:
        logger: The logger instance for this service.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize a new instance of BaseService.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger will be used.
        """
        self.logger = logger or get_logger(self.__class__.__name__)
        self.request_id = str(uuid4())
    
    async def execute(self, operation: str, input_data: Dict[str, Any] = None) -> T:
        """Execute a service operation with input data.
        
        Args:
            operation: The name of the operation to execute.
            input_data: The input data for the operation.
            
        Returns:
            The result of the operation.
            
        Raises:
            NotImplementedError: If the operation is not implemented.
        """
        start_time = time.time()
        self.logger.info(f"Starting operation: {operation}", extra={
            "request_id": self.request_id,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        try:
            # Check if the operation method exists
            if not hasattr(self, operation) or not callable(getattr(self, operation)):
                raise NotImplementedError(f"Operation {operation} is not implemented")
            
            # Execute the operation
            result = await getattr(self, operation)(**(input_data or {}))
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Operation {operation} completed successfully", extra={
                "request_id": self.request_id,
                "operation": operation,
                "elapsed_time_ms": int(elapsed_time * 1000),
                "timestamp": datetime.utcnow().isoformat(),
            })
            
            return result
        except Exception as ex:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Operation {operation} failed: {str(ex)}", extra={
                "request_id": self.request_id,
                "operation": operation,
                "elapsed_time_ms": int(elapsed_time * 1000),
                "exception": str(ex),
                "exception_type": type(ex).__name__,
                "timestamp": datetime.utcnow().isoformat(),
            })
            raise