# src/common/application/base_use_case.py
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

I = TypeVar('I')  # Input
O = TypeVar('O')  # Output

class BaseUseCase(Generic[I, O], ABC):
    """Base class for all use cases.
    
    Use cases represent application-specific business rules.
    They encapsulate all the business logic of a specific use case in the application.
    They are independent of any external agency, such as databases or UI.
    
    Attributes:
        None
    """
    
    @abstractmethod
    async def execute(self, input_dto: I) -> O:
        """Execute the use case.
        
        Args:
            input_dto: The input data transfer object.
            
        Returns:
            The output data transfer object.
        """
        pass