# tests/common/infrastructure/test_event_bus.py
import pytest
from typing import List, Any
import asyncio
from datetime import datetime, timezone
import uuid
from src.common.domain.base_event import BaseEvent
from src.common.infrastructure.event_bus import InMemoryEventBus

class TestEvent(BaseEvent):
    """A test event class."""
    def __init__(self, id=None, timestamp=None, aggregate_id=None, data=None):
        # Garantir valores padrão corretos, especialmente para timestamp
        timestamp = timestamp or datetime.utcnow()
        super().__init__(id, timestamp, aggregate_id)
        self.data = data

class AnotherTestEvent(BaseEvent):
    """Another test event class."""
    def __init__(self, id=None, timestamp=None, aggregate_id=None, data=None):
        # Garantir valores padrão corretos
        id = id or str(uuid.uuid4())
        timestamp = timestamp or datetime.utcnow()
        
        super().__init__(id, timestamp, aggregate_id)
        self.data = data

class TestEventBus:
    """Tests for the InMemoryEventBus class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a test event bus."""
        return InMemoryEventBus()
    
    @pytest.mark.asyncio
    async def test_publish_with_no_handlers(self, event_bus):
        """Test publishing an event with no handlers."""
        event = TestEvent(data={"key": "value"})
        
        # This should not raise any exceptions
        await event_bus.publish(event)
    
    @pytest.mark.asyncio
    async def test_publish_with_sync_handler(self, event_bus):
        """Test publishing an event with a synchronous handler."""
        received_events: List[TestEvent] = []
        
        def handler(event: TestEvent):
            received_events.append(event)
        
        event_bus.subscribe(TestEvent, handler)
        
        event = TestEvent(data={"key": "value"})
        await event_bus.publish(event)
        
        # Allow the handler to execute
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        assert received_events[0].id == event.id
        assert received_events[0].data == event.data
    
    @pytest.mark.asyncio
    async def test_publish_with_async_handler(self, event_bus):
        """Test publishing an event with an asynchronous handler."""
        received_events: List[TestEvent] = []
        
        async def handler(event: TestEvent):
            received_events.append(event)
        
        event_bus.subscribe_async(TestEvent, handler)
        
        event = TestEvent(data={"key": "value"})
        await event_bus.publish(event)
        
        assert len(received_events) == 1
        assert received_events[0].id == event.id
        assert received_events[0].data == event.data
    
    @pytest.mark.asyncio
    async def test_publish_with_multiple_handlers(self, event_bus):
        """Test publishing an event with multiple handlers."""
        sync_received_events: List[TestEvent] = []
        async_received_events: List[TestEvent] = []
        
        def sync_handler(event: TestEvent):
            sync_received_events.append(event)
        
        async def async_handler(event: TestEvent):
            async_received_events.append(event)
        
        event_bus.subscribe(TestEvent, sync_handler)
        event_bus.subscribe_async(TestEvent, async_handler)
        
        event = TestEvent(data={"key": "value"})
        await event_bus.publish(event)
        
        # Allow the sync handler to execute
        await asyncio.sleep(0.1)
        
        assert len(sync_received_events) == 1
        assert sync_received_events[0].id == event.id
        assert sync_received_events[0].data == event.data
        
        assert len(async_received_events) == 1
        assert async_received_events[0].id == event.id
        assert async_received_events[0].data == event.data
    
    @pytest.mark.asyncio
    async def test_publish_with_multiple_event_types(self, event_bus):
        """Test publishing different event types."""
        test_events: List[TestEvent] = []
        another_events: List[AnotherTestEvent] = []
        
        def test_handler(event: TestEvent):
            test_events.append(event)
        
        def another_handler(event: AnotherTestEvent):
            another_events.append(event)
        
        event_bus.subscribe(TestEvent, test_handler)
        event_bus.subscribe(AnotherTestEvent, another_handler)
        
        test_event = TestEvent(data={"key": "value1"})
        another_event = AnotherTestEvent(data={"key": "value2"})
        
        await event_bus.publish(test_event)
        await event_bus.publish(another_event)
        
        # Allow the handlers to execute
        await asyncio.sleep(0.1)
        
        assert len(test_events) == 1
        assert test_events[0].id == test_event.id
        assert test_events[0].data == test_event.data
        
        assert len(another_events) == 1
        assert another_events[0].id == another_event.id
        assert another_events[0].data == another_event.data
    
    @pytest.mark.asyncio
    async def test_unsubscribe_sync_handler(self, event_bus):
        """Test unsubscribing a synchronous handler."""
        received_events: List[TestEvent] = []
        
        def handler(event: TestEvent):
            received_events.append(event)
        
        event_bus.subscribe(TestEvent, handler)
        
        # Publish first event
        event1 = TestEvent(data={"key": "value1"})
        await event_bus.publish(event1)
        
        # Allow the handler to execute
        await asyncio.sleep(0.1)
        
        assert len(received_events) == 1
        
        # Unsubscribe the handler
        event_bus.unsubscribe(TestEvent, handler)
        
        # Publish second event
        event2 = TestEvent(data={"key": "value2"})
        await event_bus.publish(event2)
        
        # Allow the handler to execute
        await asyncio.sleep(0.1)
        
        # The handler should not have been called for the second event
        assert len(received_events) == 1
    
    @pytest.mark.asyncio
    async def test_unsubscribe_async_handler(self, event_bus):
        """Test unsubscribing an asynchronous handler."""
        received_events: List[TestEvent] = []
        
        async def handler(event: TestEvent):
            received_events.append(event)
        
        event_bus.subscribe_async(TestEvent, handler)
        
        # Publish first event
        event1 = TestEvent(data={"key": "value1"})
        await event_bus.publish(event1)
        
        assert len(received_events) == 1
        
        # Unsubscribe the handler
        event_bus.unsubscribe_async(TestEvent, handler)
        
        # Publish second event
        event2 = TestEvent(data={"key": "value2"})
        await event_bus.publish(event2)
        
        # The handler should not have been called for the second event
        assert len(received_events) == 1