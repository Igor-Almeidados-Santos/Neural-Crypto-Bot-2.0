"""
UnitOfWork implementa o padrão Unit of Work para garantir transações consistentes.

O padrão Unit of Work rastreia as mudanças no sistema e coordena a persistência
de todas as alterações como uma unidade atômica.
"""
# src/common/infrastructure/database/unit_of_work.py
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar, Generic, Optional
import logging

from ..logging.logger import get_logger

T = TypeVar('T')

class UnitOfWork(Generic[T], ABC):
    """Base class for implementing the Unit of Work pattern.
    
    The Unit of Work pattern maintains a list of objects affected by a business transaction
    and coordinates the writing out of changes and the resolution of concurrency problems.
    
    Attributes:
        repositories: The repositories that this unit of work manages.
        logger: The logger for this unit of work.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize a new instance of UnitOfWork.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger will be used.
        """
        self.logger = logger or get_logger(self.__class__.__name__)
        self.repositories = {}
    
    async def __aenter__(self) -> 'UnitOfWork[T]':
        """Enter the context manager.
        
        Returns:
            This unit of work instance.
        """
        await self.begin()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager.
        
        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The exception traceback, if an exception was raised.
        """
        if exc_type:
            await self.rollback()
            self.logger.error(f"Transaction rolled back due to error: {exc_val}")
        else:
            try:
                await self.commit()
            except Exception as e:
                await self.rollback()
                self.logger.error(f"Transaction rolled back due to commit error: {str(e)}")
                raise
    
    @abstractmethod
    async def begin(self):
        """Begin a new transaction."""
        pass
    
    @abstractmethod
    async def commit(self):
        """Commit the current transaction."""
        pass
    
    @abstractmethod
    async def rollback(self):
        """Rollback the current transaction."""
        pass
    
    @abstractmethod
    def register_repository(self, name: str, repository: Any):
        """Register a repository with this unit of work.
        
        Args:
            name: The name to use for accessing the repository.
            repository: The repository instance.
        """
        pass
    
    async def execute_in_transaction(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function within a transaction.
        
        Args:
            func: The function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
            
        Returns:
            The result of the function.
        """
        async with self:
            return await func(*args, **kwargs)