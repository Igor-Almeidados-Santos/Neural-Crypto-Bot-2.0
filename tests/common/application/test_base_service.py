# tests/common/application/test_base_service.py
import pytest
import logging
from typing import Any, Dict
import asyncio
from datetime import datetime, timezone
from common.application.base_service import BaseService

class TestService(BaseService):
    """A test service implementation."""
    
    async def get_data(self, id: str) -> Dict[str, Any]:
        """Get data by ID."""
        return {"id": id, "name": "Test", "timestamp": datetime.now(timezone.utc)}
    
    async def process_data(self, data: Dict[str, Any]) -> str:
        """Process data and return a result."""
        return f"Processed {data['name']}"
    
    async def failing_operation(self) -> None:
        """An operation that fails."""
        raise ValueError("Test error")

class TestBaseService:
    """Tests for the BaseService class."""
    
    @pytest.mark.asyncio
    async def test_execute_get_data(self):
        """Test executing the get_data operation."""
        service = TestService()
        result = await service.execute("get_data", {"id": "123"})
        
        assert result["id"] == "123"
        assert result["name"] == "Test"
        assert isinstance(result["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_execute_process_data(self):
        """Test executing the process_data operation."""
        service = TestService()
        result = await service.execute("process_data", {"data": {"name": "Sample"}})
        
        assert result == "Processed Sample"
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_operation(self):
        """Test executing a nonexistent operation."""
        service = TestService()
        
        with pytest.raises(NotImplementedError) as excinfo:
            await service.execute("nonexistent_operation")
        
        assert "Operation nonexistent_operation is not implemented" in str(excinfo.value)
    
    @pytest.mark.asyncio
    async def test_execute_failing_operation(self):
        """Test executing an operation that fails."""
        service = TestService()
        
        with pytest.raises(ValueError) as excinfo:
            await service.execute("failing_operation")
        
        assert str(excinfo.value) == "Test error"
    
    def test_logger_initialization(self):
        """Test that the logger is properly initialized."""
        service = TestService()
        assert service.logger is not None
        assert service.logger.name == "TestService"
    
    def test_request_id_generation(self):
        """Test that a request ID is generated."""
        service = TestService()
        assert service.request_id is not None
        assert isinstance(service.request_id, str)