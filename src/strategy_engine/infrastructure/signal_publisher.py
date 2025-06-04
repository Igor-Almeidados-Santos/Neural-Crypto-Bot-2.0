"""
Signal Publisher - Infrastructure for publishing trading signals

This module implements the signal publishing infrastructure with support
for multiple message brokers, event streaming, and real-time notifications.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Callable, Set
from uuid import UUID
from datetime import datetime, timezone
from decimal import Decimal
import asyncio
import json
import logging
from dataclasses import asdict

# Import domain entities and events
from ..domain.entities.signal import Signal, SignalStatus, SignalType
from ..domain.events.signal_generated_event import SignalGeneratedEvent
from ..domain.events.position_changed_event import PositionChangedEvent


class PublisherError(Exception):
    """Base exception for publisher operations"""
    pass


class ConnectionError(PublisherError):
    """Raised when connection to message broker fails"""
    pass


class PublishError(PublisherError):
    """Raised when signal publishing fails"""
    pass


class ISignalPublisher(ABC):
    """Abstract interface for signal publishing"""
    
    @abstractmethod
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish a trading signal"""
        pass
    
    @abstractmethod
    async def publish_event(self, event: Any) -> bool:
        """Publish a domain event"""
        pass
    
    @abstractmethod
    async def subscribe_to_signals(self, callback: Callable, filter_criteria: Optional[Dict] = None) -> None:
        """Subscribe to signal notifications"""
        pass
    
    @abstractmethod
    async def unsubscribe_from_signals(self, callback: Callable) -> None:
        """Unsubscribe from signal notifications"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """Start the publisher"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the publisher"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check publisher health"""
        pass


class InMemorySignalPublisher(ISignalPublisher):
    """In-memory signal publisher for testing and development"""
    
    def __init__(self):
        self.subscribers: List[Callable] = []
        self.signal_queue: List[Signal] = []
        self.event_queue: List[Any] = []
        self.is_running = False
        self.logger = logging.getLogger(__name__)
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish a trading signal to in-memory subscribers"""
        try:
            self.signal_queue.append(signal)
            
            # Notify all subscribers
            for subscriber in self.subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(signal)
                    else:
                        subscriber(signal)
                except Exception as e:
                    self.logger.error(f"Error in signal subscriber: {e}")
            
            self.logger.info(f"Published signal {signal.id} for {signal.symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish signal: {e}")
            return False
    
    async def publish_event(self, event: Any) -> bool:
        """Publish a domain event"""
        try:
            self.event_queue.append(event)
            self.logger.debug(f"Published event {event.event_type.value}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
            return False
    
    async def subscribe_to_signals(self, callback: Callable, filter_criteria: Optional[Dict] = None) -> None:
        """Subscribe to signal notifications"""
        if filter_criteria:
            # Wrap callback with filter
            async def filtered_callback(signal: Signal):
                if self._matches_filter(signal, filter_criteria):
                    await callback(signal) if asyncio.iscoroutinefunction(callback) else callback(signal)
            self.subscribers.append(filtered_callback)
        else:
            self.subscribers.append(callback)
    
    async def unsubscribe_from_signals(self, callback: Callable) -> None:
        """Unsubscribe from signal notifications"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def start(self) -> None:
        """Start the publisher"""
        self.is_running = True
        self.logger.info("In-memory signal publisher started")
    
    async def stop(self) -> None:
        """Stop the publisher"""
        self.is_running = False
        self.subscribers.clear()
        self.logger.info("In-memory signal publisher stopped")
    
    async def health_check(self) -> bool:
        """Check publisher health"""
        return self.is_running
    
    def _matches_filter(self, signal: Signal, filter_criteria: Dict) -> bool:
        """Check if signal matches filter criteria"""
        for key, value in filter_criteria.items():
            if hasattr(signal, key):
                signal_value = getattr(signal, key)
                if isinstance(value, list):
                    if signal_value not in value:
                        return False
                elif signal_value != value:
                    return False
            else:
                return False
        return True
    
    def get_published_signals(self) -> List[Signal]:
        """Get all published signals (for testing)"""
        return self.signal_queue.copy()
    
    def get_published_events(self) -> List[Any]:
        """Get all published events (for testing)"""
        return self.event_queue.copy()
    
    def clear_queues(self) -> None:
        """Clear all queues (for testing)"""
        self.signal_queue.clear()
        self.event_queue.clear()


class KafkaSignalPublisher(ISignalPublisher):
    """Kafka-based signal publisher for production use"""
    
    def __init__(self, 
                 bootstrap_servers: str = "localhost:9092",
                 signal_topic: str = "trading-signals",
                 event_topic: str = "domain-events",
                 **kafka_config):
        self.bootstrap_servers = bootstrap_servers
        self.signal_topic = signal_topic
        self.event_topic = event_topic
        self.kafka_config = kafka_config
        self.producer = None
        self.consumer = None
        self.is_running = False
        self.subscribers: Dict[str, Callable] = {}
        self.consumer_tasks: Set[asyncio.Task] = set()
        self.logger = logging.getLogger(__name__)
    
    async def _get_redis_connection(self):
        """Get Redis connection"""
        if not self.redis:
            import redis.asyncio as redis
            self.redis = redis.from_url(self.redis_url, **self.redis_config)
        return self.redis
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish trading signal to Redis Stream"""
        try:
            redis_conn = await self._get_redis_connection()
            
            signal_data = {
                "type": "signal",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "signal_id": str(signal.id),
                "strategy_id": str(signal.strategy_id),
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "signal_type": signal.signal_type.value,
                "quantity": str(signal.quantity),
                "price": str(signal.current_price),
                "data": json.dumps(signal.to_dict())
            }
            
            await redis_conn.xadd(self.signal_stream, signal_data)
            self.logger.info(f"Published signal {signal.id} to Redis stream {self.signal_stream}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish signal to Redis: {e}")
            return False
    
    async def publish_event(self, event: Any) -> bool:
        """Publish domain event to Redis Stream"""
        try:
            redis_conn = await self._get_redis_connection()
            
            event_data = {
                "type": "event",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": event.event_type.value if hasattr(event, 'event_type') else "unknown",
                "aggregate_id": str(event.aggregate_id) if hasattr(event, 'aggregate_id') else "",
                "data": json.dumps(event.to_dict() if hasattr(event, 'to_dict') else asdict(event))
            }
            
            await redis_conn.xadd(self.event_stream, event_data)
            self.logger.debug(f"Published event to Redis stream {self.event_stream}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish event to Redis: {e}")
            return False
    
    async def subscribe_to_signals(self, callback: Callable, filter_criteria: Optional[Dict] = None) -> None:
        """Subscribe to signal notifications from Redis Stream"""
        subscriber_id = f"subscriber_{id(callback)}"
        self.subscribers[subscriber_id] = callback
        
        # Start consumer task for this subscriber
        consumer_task = asyncio.create_task(
            self._consume_signals(subscriber_id, callback, filter_criteria)
        )
        self.consumer_tasks.add(consumer_task)
    
    async def unsubscribe_from_signals(self, callback: Callable) -> None:
        """Unsubscribe from signal notifications"""
        subscriber_id = f"subscriber_{id(callback)}"
        self.subscribers.pop(subscriber_id, None)
        
        # Cancel corresponding consumer task
        for task in list(self.consumer_tasks):
            if task.get_name() == subscriber_id:
                task.cancel()
                self.consumer_tasks.discard(task)
                break
    
    async def _consume_signals(self, subscriber_id: str, callback: Callable, filter_criteria: Optional[Dict]):
        """Consume signals from Redis Stream for a specific subscriber"""
        try:
            redis_conn = await self._get_redis_connection()
            
            # Create consumer group if it doesn't exist
            try:
                await redis_conn.xgroup_create(
                    self.signal_stream, 
                    f"group_{subscriber_id}", 
                    id="0", 
                    mkstream=True
                )
            except Exception:
                pass  # Group already exists
            
            while self.is_running:
                try:
                    # Read messages from stream
                    messages = await redis_conn.xreadgroup(
                        f"group_{subscriber_id}",
                        subscriber_id,
                        {self.signal_stream: ">"},
                        count=10,
                        block=1000
                    )
                    
                    for stream, msgs in messages:
                        for msg_id, fields in msgs:
                            try:
                                if fields.get(b"type") == b"signal":
                                    signal_data = json.loads(fields[b"data"].decode())
                                    signal = Signal.from_dict(signal_data)
                                    
                                    # Apply filter if specified
                                    if filter_criteria and not self._matches_filter(signal, filter_criteria):
                                        continue
                                    
                                    # Call subscriber callback
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(signal)
                                    else:
                                        callback(signal)
                                    
                                    # Acknowledge message
                                    await redis_conn.xack(self.signal_stream, f"group_{subscriber_id}", msg_id)
                                    
                            except Exception as e:
                                self.logger.error(f"Error processing signal message: {e}")
                
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in Redis signal consumer: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.logger.error(f"Fatal error in Redis signal consumer: {e}")
    
    def _matches_filter(self, signal: Signal, filter_criteria: Dict) -> bool:
        """Check if signal matches filter criteria"""
        for key, value in filter_criteria.items():
            if hasattr(signal, key):
                signal_value = getattr(signal, key)
                if isinstance(value, list):
                    if signal_value not in value:
                        return False
                elif signal_value != value:
                    return False
            else:
                return False
        return True
    
    async def start(self) -> None:
        """Start the Redis publisher"""
        await self._get_redis_connection()
        self.is_running = True
        self.logger.info("Redis signal publisher started")
    
    async def stop(self) -> None:
        """Stop the Redis publisher"""
        self.is_running = False
        
        # Cancel all consumer tasks
        for task in self.consumer_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        self.consumer_tasks.clear()
        self.subscribers.clear()
        
        if self.redis:
            await self.redis.close()
        
        self.logger.info("Redis signal publisher stopped")
    
    async def health_check(self) -> bool:
        """Check Redis publisher health"""
        try:
            redis_conn = await self._get_redis_connection()
            await redis_conn.ping()
            return True
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False


class WebSocketSignalPublisher(ISignalPublisher):
    """WebSocket-based signal publisher for real-time web clients"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.server = None
        self.connected_clients: Set = set()
        self.is_running = False
        self.logger = logging.getLogger(__name__)
    
    async def _handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        self.connected_clients.add(websocket)
        self.logger.info(f"New WebSocket client connected from {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.discard(websocket)
            self.logger.info(f"WebSocket client disconnected from {websocket.remote_address}")
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish signal to all connected WebSocket clients"""
        if not self.connected_clients:
            return True
        
        try:
            message = {
                "type": "signal",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": signal.to_dict()
            }
            
            message_json = json.dumps(message, default=str)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.connected_clients:
                try:
                    await client.send(message_json)
                except Exception as e:
                    self.logger.warning(f"Failed to send signal to client: {e}")
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients
            
            self.logger.info(f"Published signal {signal.id} to {len(self.connected_clients)} WebSocket clients")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish signal via WebSocket: {e}")
            return False
    
    async def publish_event(self, event: Any) -> bool:
        """Publish event to all connected WebSocket clients"""
        if not self.connected_clients:
            return True
        
        try:
            message = {
                "type": "event",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": event.to_dict() if hasattr(event, 'to_dict') else asdict(event)
            }
            
            message_json = json.dumps(message, default=str)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.connected_clients:
                try:
                    await client.send(message_json)
                except Exception:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.connected_clients -= disconnected_clients
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish event via WebSocket: {e}")
            return False
    
    async def subscribe_to_signals(self, callback: Callable, filter_criteria: Optional[Dict] = None) -> None:
        """WebSocket subscriptions are handled automatically when clients connect"""
        pass
    
    async def unsubscribe_from_signals(self, callback: Callable) -> None:
        """WebSocket unsubscriptions are handled automatically when clients disconnect"""
        pass
    
    async def start(self) -> None:
        """Start the WebSocket server"""
        try:
            import websockets
            self.server = await websockets.serve(self._handle_client, self.host, self.port)
            self.is_running = True
            self.logger.info(f"WebSocket signal publisher started on {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket server: {e}")
            raise ConnectionError(f"Failed to start WebSocket server: {e}")
    
    async def stop(self) -> None:
        """Stop the WebSocket server"""
        self.is_running = False
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all client connections
        if self.connected_clients:
            await asyncio.gather(
                *[client.close() for client in self.connected_clients],
                return_exceptions=True
            )
        
        self.connected_clients.clear()
        self.logger.info("WebSocket signal publisher stopped")
    
    async def health_check(self) -> bool:
        """Check WebSocket server health"""
        return self.is_running and self.server is not None


class CompositeSignalPublisher(ISignalPublisher):
    """Composite publisher that can publish to multiple backends simultaneously"""
    
    def __init__(self, publishers: List[ISignalPublisher]):
        self.publishers = publishers
        self.logger = logging.getLogger(__name__)
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish signal to all publishers"""
        results = []
        for publisher in self.publishers:
            try:
                result = await publisher.publish_signal(signal)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in publisher {type(publisher).__name__}: {e}")
                results.append(False)
        
        # Return True if at least one publisher succeeded
        return any(results)
    
    async def publish_event(self, event: Any) -> bool:
        """Publish event to all publishers"""
        results = []
        for publisher in self.publishers:
            try:
                result = await publisher.publish_event(event)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error in publisher {type(publisher).__name__}: {e}")
                results.append(False)
        
        return any(results)
    
    async def subscribe_to_signals(self, callback: Callable, filter_criteria: Optional[Dict] = None) -> None:
        """Subscribe to signals on all publishers"""
        for publisher in self.publishers:
            try:
                await publisher.subscribe_to_signals(callback, filter_criteria)
            except Exception as e:
                self.logger.error(f"Error subscribing to {type(publisher).__name__}: {e}")
    
    async def unsubscribe_from_signals(self, callback: Callable) -> None:
        """Unsubscribe from signals on all publishers"""
        for publisher in self.publishers:
            try:
                await publisher.unsubscribe_from_signals(callback)
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {type(publisher).__name__}: {e}")
    
    async def start(self) -> None:
        """Start all publishers"""
        for publisher in self.publishers:
            try:
                await publisher.start()
            except Exception as e:
                self.logger.error(f"Error starting {type(publisher).__name__}: {e}")
    
    async def stop(self) -> None:
        """Stop all publishers"""
        for publisher in self.publishers:
            try:
                await publisher.stop()
            except Exception as e:
                self.logger.error(f"Error stopping {type(publisher).__name__}: {e}")
    
    async def health_check(self) -> bool:
        """Check health of all publishers"""
        health_results = []
        for publisher in self.publishers:
            try:
                health = await publisher.health_check()
                health_results.append(health)
            except Exception as e:
                self.logger.error(f"Health check failed for {type(publisher).__name__}: {e}")
                health_results.append(False)
        
        # Return True if all publishers are healthy
        return all(health_results)


class SignalPublisherFactory:
    """Factory for creating signal publisher instances"""
    
    @staticmethod
    def create_publisher(publisher_type: str, **config) -> ISignalPublisher:
        """Create a signal publisher instance"""
        if publisher_type == "memory":
            return InMemorySignalPublisher()
        elif publisher_type == "kafka":
            return KafkaSignalPublisher(**config)
        elif publisher_type == "redis":
            return RedisSignalPublisher(**config)
        elif publisher_type == "websocket":
            return WebSocketSignalPublisher(**config)
        elif publisher_type == "composite":
            publishers = []
            for pub_config in config.get("publishers", []):
                pub_type = pub_config.pop("type")
                publisher = SignalPublisherFactory.create_publisher(pub_type, **pub_config)
                publishers.append(publisher)
            return CompositeSignalPublisher(publishers)
        else:
            raise ValueError(f"Unknown publisher type: {publisher_type}")
    
    @staticmethod
    def create_production_publisher(**config) -> ISignalPublisher:
        """Create a production-ready composite publisher"""
        publishers = []
        
        # Add Kafka for reliable message delivery
        if config.get("kafka_enabled", True):
            kafka_config = config.get("kafka", {})
            publishers.append(KafkaSignalPublisher(**kafka_config))
        
        # Add Redis for fast in-memory streaming
        if config.get("redis_enabled", True):
            redis_config = config.get("redis", {})
            publishers.append(RedisSignalPublisher(**redis_config))
        
        # Add WebSocket for real-time web clients
        if config.get("websocket_enabled", True):
            ws_config = config.get("websocket", {})
            publishers.append(WebSocketSignalPublisher(**ws_config))
        
        return CompositeSignalPublisher(publishers)


# Notification handlers for different signal types
class SignalNotificationHandler:
    """Handler for processing signal notifications and triggering actions"""
    
    def __init__(self, publisher: ISignalPublisher):
        self.publisher = publisher
        self.handlers: Dict[SignalType, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, signal_type: SignalType, handler: Callable):
        """Register a handler for specific signal type"""
        if signal_type not in self.handlers:
            self.handlers[signal_type] = []
        self.handlers[signal_type].append(handler)
    
    async def handle_signal(self, signal: Signal):
        """Process signal and trigger appropriate handlers"""
        try:
            # Publish signal first
            await self.publisher.publish_signal(signal)
            
            # Trigger type-specific handlers
            if signal.signal_type in self.handlers:
                for handler in self.handlers[signal.signal_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(signal)
                        else:
                            handler(signal)
                    except Exception as e:
                        self.logger.error(f"Error in signal handler: {e}")
        
        except Exception as e:
            self.logger.error(f"Error handling signal {signal.id}: {e}")
    
    async def start(self):
        """Start the notification handler"""
        await self.publisher.start()
    
    async def stop(self):
        """Stop the notification handler"""
        await self.publisher.stop()
    
    async def _create_consumer(self):
        """Create Kafka consumer for subscriptions"""
        try:
            from aiokafka import AIOKafkaConsumer
            self.consumer = AIOKafkaConsumer(
                self.signal_topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id="signal_subscribers",
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                **self.kafka_config
            )
            await self.consumer.start()
            self.logger.info("Kafka consumer created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create Kafka consumer: {e}")
            raise ConnectionError(f"Failed to connect to Kafka consumer: {e}")
    
    async def publish_signal(self, signal: Signal) -> bool:
        """Publish trading signal to Kafka"""
        try:
            if not self.producer:
                await self._create_producer()
            
            signal_data = {
                "type": "signal",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": signal.to_dict()
            }
            
            await self.producer.send(self.signal_topic, signal_data)
            self.logger.info(f"Published signal {signal.id} to Kafka topic {self.signal_topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish signal to Kafka: {e}")
            raise PublishError(f"Failed to publish signal: {e}")
    
    async def publish_event(self, event: Any) -> bool:
        """Publish domain event to Kafka"""
        try:
            if not self.producer:
                await self._create_producer()
            
            event_data = {
                "type": "event",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": event.to_dict() if hasattr(event, 'to_dict') else asdict(event)
            }
            
            await self.producer.send(self.event_topic, event_data)
            self.logger.debug(f"Published event to Kafka topic {self.event_topic}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to publish event to Kafka: {e}")
            return False
    
    async def subscribe_to_signals(self, callback: Callable, filter_criteria: Optional[Dict] = None) -> None:
        """Subscribe to signal notifications from Kafka"""
        subscriber_id = f"subscriber_{id(callback)}"
        
        if filter_criteria:
            # Store filter criteria with callback
            self.subscribers[subscriber_id] = (callback, filter_criteria)
        else:
            self.subscribers[subscriber_id] = (callback, None)
        
        # Start consumer if not already running
        if not self.consumer_task:
            self.consumer_task = asyncio.create_task(self._consume_messages())
    
    async def unsubscribe_from_signals(self, callback: Callable) -> None:
        """Unsubscribe from signal notifications"""
        subscriber_id = f"subscriber_{id(callback)}"
        self.subscribers.pop(subscriber_id, None)
    
    async def _consume_messages(self):
        """Consume messages from Kafka and notify subscribers"""
        try:
            if not self.consumer:
                await self._create_consumer()
            
            async for message in self.consumer:
                try:
                    message_data = message.value
                    
                    if message_data.get("type") == "signal":
                        signal_dict = message_data.get("data", {})
                        signal = Signal.from_dict(signal_dict)
                        
                        # Notify all subscribers
                        for callback, filter_criteria in self.subscribers.values():
                            try:
                                if filter_criteria and not self._matches_filter(signal, filter_criteria):
                                    continue
                                
                                if asyncio.iscoroutinefunction(callback):
                                    await callback(signal)
                                else:
                                    callback(signal)
                                    
                            except Exception as e:
                                self.logger.error(f"Error in signal subscriber: {e}")
                
                except Exception as e:
                    self.logger.error(f"Error processing Kafka message: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in Kafka consumer: {e}")
    
    def _matches_filter(self, signal: Signal, filter_criteria: Dict) -> bool:
        """Check if signal matches filter criteria"""
        for key, value in filter_criteria.items():
            if hasattr(signal, key):
                signal_value = getattr(signal, key)
                if isinstance(value, list):
                    if signal_value not in value:
                        return False
                elif signal_value != value:
                    return False
            else:
                return False
        return True
    
    async def start(self) -> None:
        """Start the Kafka publisher"""
        await self._create_producer()
        self.is_running = True
        self.logger.info("Kafka signal publisher started")
    
    async def stop(self) -> None:
        """Stop the Kafka publisher"""
        self.is_running = False
        
        if self.consumer_task:
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
        
        if self.producer:
            await self.producer.stop()
        
        if self.consumer:
            await self.consumer.stop()
        
        self.logger.info("Kafka signal publisher stopped")
    
    async def health_check(self) -> bool:
        """Check Kafka publisher health"""
        try:
            if not self.producer:
                return False
            
            # Try to send a health check message
            health_data = {
                "type": "health_check",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            future = await self.producer.send(self.signal_topic, health_data)
            await future
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False


class RedisSignalPublisher(ISignalPublisher):
    """Redis-based signal publisher using Redis Streams"""
    
    def __init__(self, 
                redis_url: str = "redis://localhost:6379",
                signal_stream: str = "signals",
                event_stream: str = "events",
                 **redis_config):
        self.redis_url = redis_url
        self.signal_stream = signal_stream
        self.event_stream = event_stream
        self.redis_config = redis_config
        self.redis = None
        self.is_running = False