"""
Domain Events - Event-driven architecture for the strategy engine

This module implements domain events for signal generation, position changes,
and other critical domain events with event sourcing capabilities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Type, Union
from uuid import UUID, uuid4
import json


class EventType(Enum):
    """Event type enumeration"""
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_VALIDATED = "SIGNAL_VALIDATED"
    SIGNAL_EXECUTED = "SIGNAL_EXECUTED"
    SIGNAL_CANCELLED = "SIGNAL_CANCELLED"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_UPDATED = "POSITION_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_RISK_BREACH = "POSITION_RISK_BREACH"
    
    STRATEGY_STARTED = "STRATEGY_STARTED"
    STRATEGY_STOPPED = "STRATEGY_STOPPED"
    STRATEGY_PAUSED = "STRATEGY_PAUSED"
    STRATEGY_RESUMED = "STRATEGY_RESUMED"
    STRATEGY_ERROR = "STRATEGY_ERROR"
    
    PORTFOLIO_REBALANCED = "PORTFOLIO_REBALANCED"
    RISK_LIMIT_BREACHED = "RISK_LIMIT_BREACHED"
    MARKET_DATA_UPDATED = "MARKET_DATA_UPDATED"
    
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"


class EventPriority(Enum):
    """Event priority enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class DomainEvent(ABC):
    """
    Base class for all domain events in the strategy engine.
    Provides event sourcing capabilities and metadata tracking.
    """
    
    # Event identity
    event_id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.SIGNAL_GENERATED
    aggregate_id: UUID = field(default_factory=uuid4)  # ID of the aggregate that generated the event
    aggregate_type: str = ""  # Type of aggregate (Strategy, Position, Signal, etc.)
    
    # Event metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    correlation_id: Optional[UUID] = None  # For tracing related events
    causation_id: Optional[UUID] = None   # ID of the event that caused this event
    
    # Event properties
    priority: EventPriority = EventPriority.MEDIUM
    source: str = ""  # Source component that generated the event
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def get_event_data(self) -> Dict[str, Any]:
        """Get event-specific data for serialization"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "aggregate_id": str(self.aggregate_id),
            "aggregate_type": self.aggregate_type,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "priority": self.priority.value,
            "source": self.source,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data,
            "metadata": self.metadata,
            "event_data": self.get_event_data()
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string"""
        return json.dumps(self.to_dict(), default=str)


@dataclass(frozen=True)
class SignalGeneratedEvent(DomainEvent):
    """Event fired when a new trading signal is generated"""
    
    event_type: EventType = field(default=EventType.SIGNAL_GENERATED, init=False)
    aggregate_type: str = field(default="Signal", init=False)
    
    # Signal-specific data
    signal_id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    direction: str = ""  # BUY, SELL, HOLD
    signal_type: str = ""  # ENTRY, EXIT, SCALE_IN, etc.
    strength: str = ""  # WEAK, MODERATE, STRONG, VERY_STRONG
    quantity: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    target_price: Optional[Decimal] = None
    confidence: Decimal = Decimal("0")
    expected_return: Optional[Decimal] = None
    risk_score: Optional[Decimal] = None
    reason: str = ""
    features: Dict[str, Any] = field(default_factory=dict)
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "signal_id": str(self.signal_id),
            "strategy_id": str(self.strategy_id),
            "symbol": self.symbol,
            "direction": self.direction,
            "signal_type": self.signal_type,
            "strength": self.strength,
            "quantity": str(self.quantity),
            "current_price": str(self.current_price),
            "target_price": str(self.target_price) if self.target_price else None,
            "confidence": str(self.confidence),
            "expected_return": str(self.expected_return) if self.expected_return else None,
            "risk_score": str(self.risk_score) if self.risk_score else None,
            "reason": self.reason,
            "features": self.features
        }


@dataclass(frozen=True)
class SignalExecutedEvent(DomainEvent):
    """Event fired when a trading signal is executed"""
    
    event_type: EventType = field(default=EventType.SIGNAL_EXECUTED, init=False)
    aggregate_type: str = field(default="Signal", init=False)
    
    # Execution-specific data
    signal_id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    executed_quantity: Decimal = Decimal("0")
    executed_price: Decimal = Decimal("0")
    execution_fees: Decimal = Decimal("0")
    slippage: Optional[Decimal] = None
    market_impact: Optional[Decimal] = None
    exchange: str = ""
    order_id: Optional[str] = None
    execution_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fill_ratio: Decimal = Decimal("0")
    remaining_quantity: Decimal = Decimal("0")
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "signal_id": str(self.signal_id),
            "strategy_id": str(self.strategy_id),
            "symbol": self.symbol,
            "executed_quantity": str(self.executed_quantity),
            "executed_price": str(self.executed_price),
            "execution_fees": str(self.execution_fees),
            "slippage": str(self.slippage) if self.slippage else None,
            "market_impact": str(self.market_impact) if self.market_impact else None,
            "exchange": self.exchange,
            "order_id": self.order_id,
            "execution_time": self.execution_time.isoformat(),
            "fill_ratio": str(self.fill_ratio),
            "remaining_quantity": str(self.remaining_quantity)
        }


@dataclass(frozen=True)
class PositionChangedEvent(DomainEvent):
    """Event fired when a position is created, updated, or closed"""
    
    event_type: EventType = field(default=EventType.POSITION_UPDATED, init=False)
    aggregate_type: str = field(default="Position", init=False)
    
    # Position-specific data
    position_id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    position_type: str = ""  # LONG, SHORT, NEUTRAL
    status: str = ""  # OPEN, CLOSED, PARTIALLY_CLOSED
    change_type: str = ""  # OPENED, UPDATED, CLOSED, TRADE_ADDED
    
    # Position metrics
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    return_percentage: Decimal = Decimal("0")
    
    # Change details
    quantity_change: Decimal = Decimal("0")
    price_change: Decimal = Decimal("0")
    pnl_change: Decimal = Decimal("0")
    
    # Risk metrics
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    risk_breached: bool = False
    risk_reason: Optional[str] = None
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "position_id": str(self.position_id),
            "strategy_id": str(self.strategy_id),
            "symbol": self.symbol,
            "position_type": self.position_type,
            "status": self.status,
            "change_type": self.change_type,
            "quantity": str(self.quantity),
            "average_entry_price": str(self.average_entry_price),
            "current_price": str(self.current_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "total_pnl": str(self.total_pnl),
            "return_percentage": str(self.return_percentage),
            "quantity_change": str(self.quantity_change),
            "price_change": str(self.price_change),
            "pnl_change": str(self.pnl_change),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "risk_breached": self.risk_breached,
            "risk_reason": self.risk_reason
        }


@dataclass(frozen=True)
class StrategyStatusChangedEvent(DomainEvent):
    """Event fired when strategy status changes"""
    
    event_type: EventType = field(default=EventType.STRATEGY_STARTED, init=False)
    aggregate_type: str = field(default="Strategy", init=False)
    
    # Strategy-specific data
    strategy_id: UUID = field(default_factory=uuid4)
    strategy_name: str = ""
    old_status: str = ""
    new_status: str = ""
    reason: str = ""
    
    # Strategy metrics
    allocated_capital: Decimal = Decimal("0")
    used_capital: Decimal = Decimal("0")
    available_capital: Decimal = Decimal("0")
    current_positions: int = 0
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    
    # Performance snapshot
    total_return: Decimal = Decimal("0")
    win_rate: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "strategy_id": str(self.strategy_id),
            "strategy_name": self.strategy_name,
            "old_status": self.old_status,
            "new_status": self.new_status,
            "reason": self.reason,
            "allocated_capital": str(self.allocated_capital),
            "used_capital": str(self.used_capital),
            "available_capital": str(self.available_capital),
            "current_positions": self.current_positions,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "total_return": str(self.total_return),
            "win_rate": str(self.win_rate),
            "sharpe_ratio": str(self.sharpe_ratio),
            "max_drawdown": str(self.max_drawdown)
        }


@dataclass(frozen=True)
class RiskLimitBreachedEvent(DomainEvent):
    """Event fired when risk limits are breached"""
    
    event_type: EventType = field(default=EventType.RISK_LIMIT_BREACHED, init=False)
    aggregate_type: str = field(default="RiskManager", init=False)
    priority: EventPriority = field(default=EventPriority.CRITICAL, init=False)
    
    # Risk-specific data
    strategy_id: UUID = field(default_factory=uuid4)
    position_id: Optional[UUID] = None
    symbol: str = ""
    risk_type: str = ""  # STOP_LOSS, MAX_DRAWDOWN, VAR_LIMIT, POSITION_SIZE, etc.
    limit_value: Decimal = Decimal("0")
    current_value: Decimal = Decimal("0")
    breach_percentage: Decimal = Decimal("0")
    severity: str = ""  # WARNING, CRITICAL, EMERGENCY
    
    # Context
    recommended_action: str = ""
    auto_action_taken: bool = False
    action_taken: str = ""
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "strategy_id": str(self.strategy_id),
            "position_id": str(self.position_id) if self.position_id else None,
            "symbol": self.symbol,
            "risk_type": self.risk_type,
            "limit_value": str(self.limit_value),
            "current_value": str(self.current_value),
            "breach_percentage": str(self.breach_percentage),
            "severity": self.severity,
            "recommended_action": self.recommended_action,
            "auto_action_taken": self.auto_action_taken,
            "action_taken": self.action_taken
        }


@dataclass(frozen=True)
class MarketDataUpdatedEvent(DomainEvent):
    """Event fired when market data is updated"""
    
    event_type: EventType = field(default=EventType.MARKET_DATA_UPDATED, init=False)
    aggregate_type: str = field(default="MarketData", init=False)
    priority: EventPriority = field(default=EventPriority.LOW, init=False)
    
    # Market data specific
    symbol: str = ""
    exchange: str = ""
    price: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    
    # Price changes
    price_change: Decimal = Decimal("0")
    price_change_pct: Decimal = Decimal("0")
    
    # Additional metrics
    volatility: Optional[Decimal] = None
    momentum: Optional[Decimal] = None
    trend: Optional[str] = None
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "exchange": self.exchange,
            "price": str(self.price),
            "volume": str(self.volume),
            "bid": str(self.bid) if self.bid else None,
            "ask": str(self.ask) if self.ask else None,
            "spread": str(self.spread) if self.spread else None,
            "price_change": str(self.price_change),
            "price_change_pct": str(self.price_change_pct),
            "volatility": str(self.volatility) if self.volatility else None,
            "momentum": str(self.momentum) if self.momentum else None,
            "trend": self.trend
        }


@dataclass(frozen=True)
class ExecutionCompletedEvent(DomainEvent):
    """Event fired when order execution is completed"""
    
    event_type: EventType = field(default=EventType.EXECUTION_COMPLETED, init=False)
    aggregate_type: str = field(default="Execution", init=False)
    
    # Execution details
    order_id: str = ""
    signal_id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    side: str = ""  # BUY, SELL
    order_type: str = ""  # MARKET, LIMIT, STOP
    
    # Execution results
    requested_quantity: Decimal = Decimal("0")
    executed_quantity: Decimal = Decimal("0")
    requested_price: Optional[Decimal] = None
    executed_price: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    
    # Performance metrics
    slippage: Decimal = Decimal("0")
    market_impact: Decimal = Decimal("0")
    execution_time_ms: int = 0
    latency_ms: int = 0
    
    # Exchange info
    exchange: str = ""
    venue: str = ""
    execution_id: str = ""
    
    def get_event_data(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "signal_id": str(self.signal_id),
            "strategy_id": str(self.strategy_id),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "requested_quantity": str(self.requested_quantity),
            "executed_quantity": str(self.executed_quantity),
            "requested_price": str(self.requested_price) if self.requested_price else None,
            "executed_price": str(self.executed_price),
            "fees": str(self.fees),
            "slippage": str(self.slippage),
            "market_impact": str(self.market_impact),
            "execution_time_ms": self.execution_time_ms,
            "latency_ms": self.latency_ms,
            "exchange": self.exchange,
            "venue": self.venue,
            "execution_id": self.execution_id
        }


class EventBus(ABC):
    """Abstract event bus for publishing and subscribing to domain events"""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to the bus"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: EventType, handler) -> None:
        """Subscribe to events of a specific type"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: EventType, handler) -> None:
        """Unsubscribe from events of a specific type"""
        pass


class InMemoryEventBus(EventBus):
    """In-memory implementation of event bus for testing and development"""
    
    def __init__(self):
        self._handlers: Dict[EventType, List] = {}
        self._event_history: List[DomainEvent] = []
        self._max_history = 10000
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish an event to all registered handlers"""
        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)
        
        # Notify handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Error in event handler: {e}")
    
    async def subscribe(self, event_type: EventType, handler) -> None:
        """Subscribe a handler to events of a specific type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
    
    async def unsubscribe(self, event_type: EventType, handler) -> None:
        """Unsubscribe a handler from events of a specific type"""
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def get_events(self, 
                event_type: Optional[EventType] = None,
                aggregate_id: Optional[UUID] = None,
                since: Optional[datetime] = None) -> List[DomainEvent]:
        """Get events from history with optional filtering"""
        events = self._event_history.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events
    
    def clear_history(self) -> None:
        """Clear event history"""
        self._event_history.clear()


class EventStore:
    """Event store for persisting domain events with event sourcing capabilities"""
    
    def __init__(self):
        self._events: Dict[UUID, List[DomainEvent]] = {}
        self._snapshots: Dict[UUID, Dict[str, Any]] = {}
    
    async def save_events(self, 
                         aggregate_id: UUID, 
                         events: List[DomainEvent], 
                         expected_version: int = -1) -> None:
        """
        Save events for an aggregate with optimistic concurrency control
        
        Args:
            aggregate_id: ID of the aggregate
            events: List of events to save
            expected_version: Expected version for optimistic locking
        """
        if aggregate_id not in self._events:
            self._events[aggregate_id] = []
        
        current_version = len(self._events[aggregate_id])
        if expected_version != -1 and current_version != expected_version:
            raise Exception(f"Concurrency conflict: expected version {expected_version}, got {current_version}")
        
        # Assign version numbers to events
        for i, event in enumerate(events):
            event = dataclass_replace(event, version=current_version + i + 1)
            self._events[aggregate_id].append(event)
    
    async def get_events(self, 
                        aggregate_id: UUID, 
                        from_version: int = 0) -> List[DomainEvent]:
        """
        Get events for an aggregate from a specific version
        
        Args:
            aggregate_id: ID of the aggregate
            from_version: Starting version number
            
        Returns:
            List of events
        """
        if aggregate_id not in self._events:
            return []
        
        events = self._events[aggregate_id]
        return [e for e in events if e.version > from_version]
    
    async def save_snapshot(self, 
                           aggregate_id: UUID, 
                           snapshot: Dict[str, Any], 
                           version: int) -> None:
        """
        Save a snapshot of aggregate state
        
        Args:
            aggregate_id: ID of the aggregate
            snapshot: Snapshot data
            version: Version at which snapshot was taken
        """
        self._snapshots[aggregate_id] = {
            "data": snapshot,
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_snapshot(self, aggregate_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get the latest snapshot for an aggregate
        
        Args:
            aggregate_id: ID of the aggregate
            
        Returns:
            Snapshot data or None if no snapshot exists
        """
        return self._snapshots.get(aggregate_id)
    
    async def get_all_events(self, 
                            event_type: Optional[EventType] = None,
                            since: Optional[datetime] = None) -> List[DomainEvent]:
        """
        Get all events with optional filtering
        
        Args:
            event_type: Optional event type filter
            since: Optional timestamp filter
            
        Returns:
            List of events
        """
        all_events = []
        for events in self._events.values():
            all_events.extend(events)
        
        # Sort by timestamp
        all_events.sort(key=lambda e: e.timestamp)
        
        # Apply filters
        if event_type:
            all_events = [e for e in all_events if e.event_type == event_type]
        
        if since:
            all_events = [e for e in all_events if e.timestamp >= since]
        
        return all_events


def dataclass_replace(obj, **changes):
    """Helper function to replace fields in a frozen dataclass"""
    return type(obj)(**{**obj.__dict__, **changes})


class EventProjection(ABC):
    """Base class for event projections - read models built from events"""
    
    @abstractmethod
    async def handle_event(self, event: DomainEvent) -> None:
        """Handle a domain event and update the projection"""
        pass
    
    @abstractmethod
    async def rebuild(self, events: List[DomainEvent]) -> None:
        """Rebuild the entire projection from a list of events"""
        pass


class StrategyPerformanceProjection(EventProjection):
    """Projection for strategy performance metrics"""
    
    def __init__(self):
        self.strategies: Dict[UUID, Dict[str, Any]] = {}
    
    async def handle_event(self, event: DomainEvent) -> None:
        """Update strategy performance based on events"""
        if isinstance(event, StrategyStatusChangedEvent):
            strategy_id = event.strategy_id
            if strategy_id not in self.strategies:
                self.strategies[strategy_id] = {
                    "name": event.strategy_name,
                    "status": event.new_status,
                    "total_return": event.total_return,
                    "sharpe_ratio": event.sharpe_ratio,
                    "max_drawdown": event.max_drawdown,
                    "last_updated": event.timestamp
                }
            else:
                self.strategies[strategy_id].update({
                    "status": event.new_status,
                    "total_return": event.total_return,
                    "sharpe_ratio": event.sharpe_ratio,
                    "max_drawdown": event.max_drawdown,
                    "last_updated": event.timestamp
                })
        
        elif isinstance(event, PositionChangedEvent):
            strategy_id = event.strategy_id
            if strategy_id in self.strategies:
                # Update with latest P&L info
                self.strategies[strategy_id]["last_pnl"] = event.total_pnl
                self.strategies[strategy_id]["last_updated"] = event.timestamp
    
    async def rebuild(self, events: List[DomainEvent]) -> None:
        """Rebuild projection from all events"""
        self.strategies.clear()
        for event in events:
            await self.handle_event(event)
    
    def get_strategy_summary(self, strategy_id: UUID) -> Optional[Dict[str, Any]]:
        """Get performance summary for a strategy"""
        return self.strategies.get(strategy_id)
    
    def get_all_strategies(self) -> Dict[UUID, Dict[str, Any]]:
        """Get all strategy summaries"""
        return self.strategies.copy()


class EventHandler:
    """Base class for event handlers with error handling and retry logic"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.failed_events: List[DomainEvent] = []
    
    async def handle_with_retry(self, event: DomainEvent) -> bool:
        """Handle event with retry logic"""
        for attempt in range(self.max_retries + 1):
            try:
                await self.handle(event)
                return True
            except Exception as e:
                if attempt == self.max_retries:
                    self.failed_events.append(event)
                    print(f"Failed to handle event {event.event_id} after {self.max_retries} retries: {e}")
                    return False
                else:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        return False
    
    @abstractmethod
    async def handle(self, event: DomainEvent) -> None:
        """Handle the event - to be implemented by subclasses"""
        pass
    
    def get_failed_events(self) -> List[DomainEvent]:
        """Get list of events that failed to process"""
        return self.failed_events.copy()
    
    def clear_failed_events(self) -> None:
        """Clear the failed events list"""
        self.failed_events.clear()


# Event factory for creating events from different sources
class EventFactory:
    """Factory for creating domain events"""
    
    @staticmethod
    def create_signal_generated(
        signal_id: UUID,
        strategy_id: UUID,
        symbol: str,
        direction: str,
        quantity: Decimal,
        price: Decimal,
        confidence: Decimal,
        reason: str = "",
        **kwargs
    ) -> SignalGeneratedEvent:
        """Create a signal generated event"""
        return SignalGeneratedEvent(
            aggregate_id=signal_id,
            signal_id=signal_id,
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            current_price=price,
            confidence=confidence,
            reason=reason,
            source=kwargs.get("source", ""),
            **{k: v for k, v in kwargs.items() if k not in ["source"]}
        )
    
    @staticmethod
    def create_position_changed(
        position_id: UUID,
        strategy_id: UUID,
        symbol: str,
        change_type: str,
        **kwargs
    ) -> PositionChangedEvent:
        """Create a position changed event"""
        return PositionChangedEvent(
            aggregate_id=position_id,
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            change_type=change_type,
            **kwargs
        )
    
    @staticmethod
    def create_risk_breach(
        strategy_id: UUID,
        risk_type: str,
        limit_value: Decimal,
        current_value: Decimal,
        severity: str = "WARNING",
        **kwargs
    ) -> RiskLimitBreachedEvent:
        """Create a risk limit breached event"""
        breach_pct = ((current_value - limit_value) / limit_value * 100) if limit_value != 0 else Decimal("0")
        
        return RiskLimitBreachedEvent(
            aggregate_id=strategy_id,
            strategy_id=strategy_id,
            risk_type=risk_type,
            limit_value=limit_value,
            current_value=current_value,
            breach_percentage=breach_pct,
            severity=severity,
            **kwargs
        )