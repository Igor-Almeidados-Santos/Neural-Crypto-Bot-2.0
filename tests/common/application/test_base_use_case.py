# tests/common/application/test_base_use_case.py
import pytest
from typing import Any, Dict
import asyncio
from src.common.application.base_use_case import BaseUseCase

class TestInput:
    """A test input class."""
    def __init__(self, value: str):
        self.value = value

class TestOutput:
    """A test output class."""
    def __init__(self, result: str):
        self.result = result

class TestUseCase(BaseUseCase[TestInput, TestOutput]):
    """A test use case implementation."""
    
    async def execute(self, input_dto: TestInput) -> TestOutput:
        """Execute the use case."""
        # Simple processing: append " processed" to the input value
        result = input_dto.value + " processed"
        return TestOutput(result)

class FailingUseCase(BaseUseCase[TestInput, TestOutput]):
    """A test use case that fails."""
    
    async def execute(self, input_dto: TestInput) -> TestOutput:
        """Execute the use case and raise an exception."""
        raise ValueError("Test error")

class TestBaseUseCase:
    """Tests for the BaseUseCase class."""
    
    @pytest.mark.asyncio
    async def test_execute(self):
        """Test the execute method."""
        use_case = TestUseCase()
        input_dto = TestInput("test")
        
        output = await use_case.execute(input_dto)
        
        assert isinstance(output, TestOutput)
        assert output.result == "test processed"
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test the execute method with an error."""
        use_case = FailingUseCase()
        input_dto = TestInput("test")
        
        with pytest.raises(ValueError) as excinfo:
            await use_case.execute(input_dto)
        
        assert str(excinfo.value) == "Test error"