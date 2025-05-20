# tests/common/domain/test_base_event.py
import pytest
from datetime import datetime, timezone
import uuid
from src.common.domain.base_event import BaseEvent

class TestEvent(BaseEvent):
    """A test event class."""
    def __init__(self, id=None, timestamp=None, aggregate_id=None, data=None):
        super().__init__(id, timestamp, aggregate_id)
        self.data = data
        
    def to_dict(self):
        data = super().to_dict()
        data['data'] = self.data
        return data
    
    @classmethod
    def from_dict(cls, data):
        event = super().from_dict(data)
        event.data = data.get('data')
        return event

class TestBaseEvent:
    """Tests for the BaseEvent class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        event = TestEvent()
        
        assert event.id is not None
        assert isinstance(event.id, str)
        assert uuid.UUID(event.id, version=4)  # Validate UUID format
        
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)
        
        assert event.event_type == "TestEvent"
        assert event.aggregate_id is None
    
    def test_init_with_values(self):
        """Test initialization with provided values."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        aggregate_id = str(uuid.uuid4())
        data = {"key": "value"}
        
        event = TestEvent(
            id=event_id,
            timestamp=timestamp,
            aggregate_id=aggregate_id,
            data=data
        )
        
        assert event.id == event_id
        assert event.timestamp == timestamp
        assert event.event_type == "TestEvent"
        assert event.aggregate_id == aggregate_id
        assert event.data == data
    
    def test_post_init(self):
        """Test that __post_init__ sets the event_type correctly."""
        event = TestEvent()
        assert event.event_type == "TestEvent"
    
    def test_to_dict(self):
        """Test the to_dict method."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        aggregate_id = str(uuid.uuid4())
        data = {"key": "value"}
        
        event = TestEvent(
            id=event_id,
            timestamp=timestamp,
            aggregate_id=aggregate_id,
            data=data
        )
        
        event_dict = event.to_dict()
        
        assert event_dict['id'] == event_id
        assert event_dict['timestamp'] == timestamp.isoformat()
        assert event_dict['event_type'] == "TestEvent"
        assert event_dict['aggregate_id'] == aggregate_id
        assert event_dict['data'] == data
    
    def test_from_dict(self):
        """Test the from_dict method."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        aggregate_id = str(uuid.uuid4())
        data = {"key": "value"}
        
        event_dict = {
            'id': event_id,
            'timestamp': timestamp.isoformat(),
            'event_type': "TestEvent",
            'aggregate_id': aggregate_id,
            'data': data
        }
        
        event = TestEvent.from_dict(event_dict)
        
        assert event.id == event_id
        assert event.timestamp.isoformat() == timestamp.isoformat()
        assert event.event_type == "TestEvent"
        assert event.aggregate_id == aggregate_id
        assert event.data == data