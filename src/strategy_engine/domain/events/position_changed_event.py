"""
Position Changed Event - Domain event for position lifecycle changes

This module implements the PositionChangedEvent for tracking all position
changes including creation, updates, trades, and closure with comprehensive
risk monitoring and P&L tracking.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from .signal_generated_event import DomainEvent, EventType, EventPriority


class PositionChangeType(Enum):
    """Types of position changes"""
    CREATED = "CREATED"
    OPENED = "OPENED"
    UPDATED = "UPDATED"
    TRADE_ADDED = "TRADE_ADDED"
    PRICE_UPDATED = "PRICE_UPDATED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"
    RISK_TRIGGERED = "RISK_TRIGGERED"
    STOP_LOSS_HIT = "STOP_LOSS_HIT"
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"
    LIQUIDATED = "LIQUIDATED"
    EXPIRED = "EXPIRED"


class RiskLevel(Enum):
    """Risk level indicators"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class TradeDetails:
    """Details of a trade that triggered the position change"""
    trade_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    side: str = ""  # BUY, SELL
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    exchange: str = ""
    order_id: Optional[str] = None
    execution_id: Optional[str] = None
    slippage: Optional[Decimal] = None
    market_impact: Optional[Decimal] = None


@dataclass(frozen=True)
class PositionMetrics:
    """Comprehensive position metrics"""
    # Basic metrics
    notional_value: Decimal = Decimal("0")
    margin_used: Decimal = Decimal("0")
    leverage: Decimal = Decimal("1")
    
    # P&L metrics
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")  # After fees
    return_percentage: Decimal = Decimal("0")
    
    # Risk metrics
    var_1d: Optional[Decimal] = None  # 1-day Value at Risk
    beta: Optional[Decimal] = None
    correlation: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    
    # Trading metrics
    holding_period_hours: Optional[float] = None
    max_unrealized_profit: Decimal = Decimal("0")
    max_unrealized_loss: Decimal = Decimal("0")
    avg_entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    
    # Fees and costs
    total_fees: Decimal = Decimal("0")
    funding_costs: Decimal = Decimal("0")
    borrowing_costs: Decimal = Decimal("0")


@dataclass(frozen=True)
class RiskStatus:
    """Current risk status of the position"""
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: Decimal = Decimal("0")  # 0-100
    
    # Risk limit checks
    stop_loss_distance: Optional[Decimal] = None
    stop_loss_triggered: bool = False
    take_profit_distance: Optional[Decimal] = None
    take_profit_triggered: bool = False
    
    # Portfolio risk
    portfolio_weight: Decimal = Decimal("0")
    correlation_risk: Decimal = Decimal("0")
    concentration_risk: Decimal = Decimal("0")
    
    # Market risk
    market_volatility: Optional[Decimal] = None
    liquidity_risk: Optional[Decimal] = None
    
    # Risk warnings
    risk_warnings: List[str] = field(default_factory=list)
    risk_violations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class MarketContext:
    """Market context at the time of position change"""
    market_price: Decimal = Decimal("0")
    bid_price: Optional[Decimal] = None
    ask_price: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    
    # Market indicators
    rsi: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    bb_position: Optional[Decimal] = None  # Bollinger Bands position
    
    # Market conditions
    market_trend: Optional[str] = None  # BULLISH, BEARISH, SIDEWAYS
    volatility_regime: Optional[str] = None  # LOW, NORMAL, HIGH, EXTREME
    liquidity_condition: Optional[str] = None  # GOOD, NORMAL, POOR
    
    # News and events
    has_news_impact: bool = False
    news_sentiment: Optional[str] = None  # POSITIVE, NEGATIVE, NEUTRAL
    economic_events: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PositionChangedEvent(DomainEvent):
    """
    Comprehensive event fired when a position changes state, is updated,
    or experiences any significant modification including trades, price updates,
    and risk status changes.
    """
    
    event_type: EventType = field(default=EventType.POSITION_UPDATED, init=False)
    aggregate_type: str = field(default="Position", init=False)
    
    # Position identification
    position_id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    exchange: str = ""
    
    # Change details
    change_type: PositionChangeType = PositionChangeType.UPDATED
    change_reason: str = ""
    previous_status: str = ""
    new_status: str = ""
    
    # Position state
    position_type: str = ""  # LONG, SHORT, NEUTRAL
    quantity: Decimal = Decimal("0")
    previous_quantity: Optional[Decimal] = None
    quantity_change: Decimal = Decimal("0")
    
    # Pricing
    current_price: Decimal = Decimal("0")
    previous_price: Optional[Decimal] = None
    price_change: Decimal = Decimal("0")
    price_change_pct: Decimal = Decimal("0")
    
    # P&L changes
    pnl_change: Decimal = Decimal("0")
    unrealized_pnl_change: Decimal = Decimal("0")
    realized_pnl_change: Decimal = Decimal("0")
    
    # Comprehensive metrics
    metrics: PositionMetrics = field(default_factory=PositionMetrics)
    risk_status: RiskStatus = field(default_factory=RiskStatus)
    market_context: MarketContext = field(default_factory=MarketContext)
    
    # Trade details (if change was caused by a trade)
    trade_details: Optional[TradeDetails] = None
    
    # Timeline
    position_opened_at: Optional[datetime] = None
    position_closed_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None
    
    # Strategy context
    strategy_name: str = ""
    strategy_type: str = ""
    signal_id: Optional[UUID] = None  # Signal that caused this change
    
    # Performance impact
    strategy_impact: Dict[str, Any] = field(default_factory=dict)
    portfolio_impact: Dict[str, Any] = field(default_factory=dict)
    
    def get_event_data(self) -> Dict[str, Any]:
        """Get comprehensive event data for serialization"""
        return {
            # Basic identification
            "position_id": str(self.position_id),
            "strategy_id": str(self.strategy_id),
            "symbol": self.symbol,
            "exchange": self.exchange,
            
            # Change details
            "change_type": self.change_type.value,
            "change_reason": self.change_reason,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            
            # Position state
            "position_type": self.position_type,
            "quantity": str(self.quantity),
            "previous_quantity": str(self.previous_quantity) if self.previous_quantity else None,
            "quantity_change": str(self.quantity_change),
            
            # Pricing
            "current_price": str(self.current_price),
            "previous_price": str(self.previous_price) if self.previous_price else None,
            "price_change": str(self.price_change),
            "price_change_pct": str(self.price_change_pct),
            
            # P&L changes
            "pnl_change": str(self.pnl_change),
            "unrealized_pnl_change": str(self.unrealized_pnl_change),
            "realized_pnl_change": str(self.realized_pnl_change),
            
            # Metrics
            "metrics": {
                "notional_value": str(self.metrics.notional_value),
                "margin_used": str(self.metrics.margin_used),
                "leverage": str(self.metrics.leverage),
                "unrealized_pnl": str(self.metrics.unrealized_pnl),
                "realized_pnl": str(self.metrics.realized_pnl),
                "total_pnl": str(self.metrics.total_pnl),
                "net_pnl": str(self.metrics.net_pnl),
                "return_percentage": str(self.metrics.return_percentage),
                "var_1d": str(self.metrics.var_1d) if self.metrics.var_1d else None,
                "beta": str(self.metrics.beta) if self.metrics.beta else None,
                "correlation": str(self.metrics.correlation) if self.metrics.correlation else None,
                "volatility": str(self.metrics.volatility) if self.metrics.volatility else None,
                "holding_period_hours": self.metrics.holding_period_hours,
                "max_unrealized_profit": str(self.metrics.max_unrealized_profit),
                "max_unrealized_loss": str(self.metrics.max_unrealized_loss),
                "avg_entry_price": str(self.metrics.avg_entry_price),
                "current_price": str(self.metrics.current_price),
                "total_fees": str(self.metrics.total_fees),
                "funding_costs": str(self.metrics.funding_costs),
                "borrowing_costs": str(self.metrics.borrowing_costs)
            },
            
            # Risk status
            "risk_status": {
                "risk_level": self.risk_status.risk_level.value,
                "risk_score": str(self.risk_status.risk_score),
                "stop_loss_distance": str(self.risk_status.stop_loss_distance) if self.risk_status.stop_loss_distance else None,
                "stop_loss_triggered": self.risk_status.stop_loss_triggered,
                "take_profit_distance": str(self.risk_status.take_profit_distance) if self.risk_status.take_profit_distance else None,
                "take_profit_triggered": self.risk_status.take_profit_triggered,
                "portfolio_weight": str(self.risk_status.portfolio_weight),
                "correlation_risk": str(self.risk_status.correlation_risk),
                "concentration_risk": str(self.risk_status.concentration_risk),
                "market_volatility": str(self.risk_status.market_volatility) if self.risk_status.market_volatility else None,
                "liquidity_risk": str(self.risk_status.liquidity_risk) if self.risk_status.liquidity_risk else None,
                "risk_warnings": self.risk_status.risk_warnings,
                "risk_violations": self.risk_status.risk_violations
            },
            
            # Market context
            "market_context": {
                "market_price": str(self.market_context.market_price),
                "bid_price": str(self.market_context.bid_price) if self.market_context.bid_price else None,
                "ask_price": str(self.market_context.ask_price) if self.market_context.ask_price else None,
                "spread": str(self.market_context.spread) if self.market_context.spread else None,
                "volume_24h": str(self.market_context.volume_24h) if self.market_context.volume_24h else None,
                "rsi": str(self.market_context.rsi) if self.market_context.rsi else None,
                "macd": str(self.market_context.macd) if self.market_context.macd else None,
                "bb_position": str(self.market_context.bb_position) if self.market_context.bb_position else None,
                "market_trend": self.market_context.market_trend,
                "volatility_regime": self.market_context.volatility_regime,
                "liquidity_condition": self.market_context.liquidity_condition,
                "has_news_impact": self.market_context.has_news_impact,
                "news_sentiment": self.market_context.news_sentiment,
                "economic_events": self.market_context.economic_events
            },
            
            # Trade details
            "trade_details": {
                "trade_id": str(self.trade_details.trade_id),
                "timestamp": self.trade_details.timestamp.isoformat(),
                "side": self.trade_details.side,
                "quantity": str(self.trade_details.quantity),
                "price": str(self.trade_details.price),
                "fees": str(self.trade_details.fees),
                "exchange": self.trade_details.exchange,
                "order_id": self.trade_details.order_id,
                "execution_id": self.trade_details.execution_id,
                "slippage": str(self.trade_details.slippage) if self.trade_details.slippage else None,
                "market_impact": str(self.trade_details.market_impact) if self.trade_details.market_impact else None
            } if self.trade_details else None,
            
            # Timeline
            "position_opened_at": self.position_opened_at.isoformat() if self.position_opened_at else None,
            "position_closed_at": self.position_closed_at.isoformat() if self.position_closed_at else None,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            
            # Strategy context
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type,
            "signal_id": str(self.signal_id) if self.signal_id else None,
            
            # Impact
            "strategy_impact": self.strategy_impact,
            "portfolio_impact": self.portfolio_impact
        }
    
    @classmethod
    def create_position_opened(cls,
                              position_id: UUID,
                              strategy_id: UUID,
                              symbol: str,
                              quantity: Decimal,
                              price: Decimal,
                              position_type: str,
                              **kwargs) -> PositionChangedEvent:
        """Factory method for position opened event"""
        
        metrics = PositionMetrics(
            notional_value=quantity * price,
            avg_entry_price=price,
            current_price=price
        )
        
        return cls(
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            change_type=PositionChangeType.OPENED,
            change_reason="New position opened",
            new_status="OPEN",
            position_type=position_type,
            quantity=quantity,
            current_price=price,
            metrics=metrics,
            position_opened_at=datetime.now(timezone.utc),
            priority=EventPriority.MEDIUM,
            **kwargs
        )
    
    @classmethod
    def create_position_closed(cls,
                              position_id: UUID,
                              strategy_id: UUID,
                              symbol: str,
                              final_pnl: Decimal,
                              close_reason: str,
                              **kwargs) -> PositionChangedEvent:
        """Factory method for position closed event"""
        
        return cls(
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            change_type=PositionChangeType.CLOSED,
            change_reason=close_reason,
            previous_status="OPEN",
            new_status="CLOSED",
            quantity=Decimal("0"),
            pnl_change=final_pnl,
            realized_pnl_change=final_pnl,
            position_closed_at=datetime.now(timezone.utc),
            priority=EventPriority.MEDIUM,
            **kwargs
        )
    
    @classmethod
    def create_trade_added(cls,
                          position_id: UUID,
                          strategy_id: UUID,
                          symbol: str,
                          trade_details: TradeDetails,
                          new_quantity: Decimal,
                          new_avg_price: Decimal,
                          **kwargs) -> PositionChangedEvent:
        """Factory method for trade added event"""
        
        return cls(
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            change_type=PositionChangeType.TRADE_ADDED,
            change_reason=f"Trade executed: {trade_details.side} {trade_details.quantity}",
            quantity=new_quantity,
            quantity_change=trade_details.quantity if trade_details.side == "BUY" else -trade_details.quantity,
            current_price=trade_details.price,
            trade_details=trade_details,
            last_trade_at=trade_details.timestamp,
            priority=EventPriority.MEDIUM,
            **kwargs
        )
    
    @classmethod
    def create_risk_triggered(cls,
                             position_id: UUID,
                             strategy_id: UUID,
                             symbol: str,
                             risk_type: str,
                             risk_level: RiskLevel,
                             risk_warnings: List[str],
                             **kwargs) -> PositionChangedEvent:
        """Factory method for risk triggered event"""
        
        risk_status = RiskStatus(
            risk_level=risk_level,
            risk_warnings=risk_warnings
        )
        
        return cls(
            position_id=position_id,
            strategy_id=strategy_id,
            symbol=symbol,
            change_type=PositionChangeType.RISK_TRIGGERED,
            change_reason=f"Risk limit triggered: {risk_type}",
            risk_status=risk_status,
            priority=EventPriority.HIGH if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else EventPriority.MEDIUM,
            **kwargs
        )
    
    def is_significant_change(self, threshold_pct: Decimal = Decimal("5")) -> bool:
        """Check if this represents a significant position change"""
        return (
            self.change_type in [
                PositionChangeType.OPENED,
                PositionChangeType.CLOSED,
                PositionChangeType.LIQUIDATED,
                PositionChangeType.RISK_TRIGGERED
            ] or
            abs(self.pnl_change) > (self.metrics.notional_value * threshold_pct / 100) or
            abs(self.price_change_pct) > threshold_pct or
            self.risk_status.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of risk-related information"""
        return {
            "risk_level": self.risk_status.risk_level.value,
            "risk_score": float(self.risk_status.risk_score),
            "has_warnings": len(self.risk_status.risk_warnings) > 0,
            "has_violations": len(self.risk_status.risk_violations) > 0,
            "stop_loss_triggered": self.risk_status.stop_loss_triggered,
            "take_profit_triggered": self.risk_status.take_profit_triggered,
            "warnings": self.risk_status.risk_warnings,
            "violations": self.risk_status.risk_violations
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        return {
            "total_pnl": float(self.metrics.total_pnl),
            "net_pnl": float(self.metrics.net_pnl),
            "return_percentage": float(self.metrics.return_percentage),
            "unrealized_pnl": float(self.metrics.unrealized_pnl),
            "realized_pnl": float(self.metrics.realized_pnl),
            "notional_value": float(self.metrics.notional_value),
            "total_fees": float(self.metrics.total_fees),
            "holding_period_hours": self.metrics.holding_period_hours
        }