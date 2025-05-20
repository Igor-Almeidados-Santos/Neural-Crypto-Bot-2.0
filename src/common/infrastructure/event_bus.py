"""
EventBus implementa o padrão Publish-Subscribe para comunicação entre serviços.

O EventBus permite que diferentes partes do sistema publiquem e assinem eventos,
permitindo um acoplamento fraco e comunicação assíncrona.
"""
# src/common/infrastructure/event_bus.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type, Callable, Optional, Set
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import json

from ..domain.base_event import BaseEvent
from ..infrastructure.logging.logger import get_logger

EventHandler = Callable[[BaseEvent], None]
AsyncEventHandler = Callable[[BaseEvent], Any]

class EventBus(ABC):
    """Base class for all event buses.
    
    Event buses are responsible for dispatching events to their handlers.
    They decouple event publishers from event subscribers.
    
    Attributes:
        logger: The logger for this event bus.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize a new instance of EventBus.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger will be used.
        """
        self.logger = logger or get_logger(self.__class__.__name__)
        
    @abstractmethod
    async def publish(self, event: BaseEvent):
        """Publish an event.
        
        Args:
            event: The event to publish.
        """
        pass
    
    @abstractmethod
    def subscribe(self, event_type: Type[BaseEvent], handler: EventHandler):
        """Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to.
            handler: The handler function to call when an event of the given type is published.
        """
        pass
    
    @abstractmethod
    def subscribe_async(self, event_type: Type[BaseEvent], handler: AsyncEventHandler):
        """Subscribe to an event type with an async handler.
        
        Args:
            event_type: The type of event to subscribe to.
            handler: The async handler function to call when an event of the given type is published.
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: Type[BaseEvent], handler: EventHandler):
        """Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from.
            handler: The handler function to unsubscribe.
        """
        pass
    
    @abstractmethod
    def unsubscribe_async(self, event_type: Type[BaseEvent], handler: AsyncEventHandler):
        """Unsubscribe from an event type with an async handler.
        
        Args:
            event_type: The type of event to unsubscribe from.
            handler: The async handler function to unsubscribe.
        """
        pass

class InMemoryEventBus(EventBus):
    """In-memory implementation of EventBus.
    
    This event bus stores event handlers in memory and dispatches events
    synchronously to their handlers.
    
    Attributes:
        handlers: A dictionary mapping event types to lists of handler functions.
        async_handlers: A dictionary mapping event types to lists of async handler functions.
        logger: The logger for this event bus.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize a new instance of InMemoryEventBus.
        
        Args:
            logger: Optional logger instance. If not provided, a default logger will be used.
        """
        super().__init__(logger)
        self.handlers: Dict[Type[BaseEvent], List[EventHandler]] = {}
        self.async_handlers: Dict[Type[BaseEvent], List[AsyncEventHandler]] = {}
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.RLock()
        
    async def publish(self, event: BaseEvent):
        """Publish an event.
        
        Args:
            event: The event to publish.
        """
        event_type = type(event)
        self.logger.debug(f"Publishing event: {event_type.__name__}", extra={
            "event_id": event.id,
            "event_type": event_type.__name__,
            "aggregate_id": event.aggregate_id,
            "timestamp": event.timestamp.isoformat(),
        })
        
        # Handle synchronous handlers in separate threads
        sync_handlers = self.handlers.get(event_type, [])
        if sync_handlers:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for handler in sync_handlers:
                task = loop.run_in_executor(self._executor, handler, event)
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks)
        
        # Handle asynchronous handlers
        async_handlers = self.async_handlers.get(event_type, [])
        if async_handlers:
            await asyncio.gather(*[handler(event) for handler in async_handlers])
    
    def subscribe(self, event_type: Type[BaseEvent], handler: EventHandler):
        """Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to.
            handler: The handler function to call when an event of the given type is published.
        """
        with self._lock:
            if event_type not in self.handlers:
                self.handlers[event_type] = []
            
            if handler not in self.handlers[event_type]:
                self.handlers[event_type].append(handler)
                self.logger.debug(f"Subscribed handler to event: {event_type.__name__}")
    
    def subscribe_async(self, event_type: Type[BaseEvent], handler: AsyncEventHandler):
        """Subscribe to an event type with an async handler.
        
        Args:
            event_type: The type of event to subscribe to.
            handler: The async handler function to call when an event of the given type is published.
        """
        with self._lock:
            if event_type not in self.async_handlers:
                self.async_handlers[event_type] = []
            
            if handler not in self.async_handlers[event_type]:
                self.async_handlers[event_type].append(handler)
                self.logger.debug(f"Subscribed async handler to event: {event_type.__name__}")
    
    def unsubscribe(self, event_type: Type[BaseEvent], handler: EventHandler):
        """Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from.
            handler: The handler function to unsubscribe.
        """
        with self._lock:
            if event_type in self.handlers and handler in self.handlers[event_type]:
                self.handlers[event_type].remove(handler)
                self.logger.debug(f"Unsubscribed handler from event: {event_type.__name__}")
    
    def unsubscribe_async(self, event_type: Type[BaseEvent], handler: AsyncEventHandler):
        """Unsubscribe from an event type with an async handler.
        
        Args:
            event_type: The type of event to unsubscribe from.
            handler: The async handler function to unsubscribe.
        """
        with self._lock:
            if event_type in self.async_handlers and handler in self.async_handlers[event_type]:
                self.async_handlers[event_type].remove(handler)
                self.logger.debug(f"Unsubscribed async handler from event: {event_type.__name__}")