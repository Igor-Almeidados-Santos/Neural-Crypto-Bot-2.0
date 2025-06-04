"""
Signal Entity - Core domain model for trading signals

This module implements the Signal domain entity with comprehensive
signal generation, validation, and execution tracking capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from uuid import UUID, uuid4


class SignalType(Enum):
    """Signal type enumeration"""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"
    REBALANCE = "REBALANCE"
    HEDGE = "HEDGE"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class SignalDirection(Enum):
    """Signal direction enumeration"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Signal strength enumeration"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"


class SignalStatus(Enum):
    """Signal status enumeration"""
    PENDING = "PENDING"
    VALIDATED = "VALIDATED"
    EXECUTED = "EXECUTED"
    PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"


class TimeInForce(Enum):
    """Time in force enumeration"""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day order
    GTD = "GTD"  # Good Till Date


@dataclass(frozen=True)
class SignalMetrics:
    """Signal performance and quality metrics"""
    confidence: Decimal = Decimal("0")  # 0-100
    expected_return: Optional[Decimal] = None
    expected_risk: Optional[Decimal] = None
    sharpe_ratio: Optional[Decimal] = None
    win_probability: Optional[Decimal] = None
    risk_reward_ratio: Optional[Decimal] = None
    volatility_forecast: Optional[Decimal] = None
    correlation_score: Optional[Decimal] = None
    momentum_score: Optional[Decimal] = None
    mean_reversion_score: Optional[Decimal] = None
    
    def __post_init__(self):
        """Validate metrics"""
        if not (0 <= self.confidence <= 100):
            raise ValueError("Confidence must be between 0 and 100")


@dataclass(frozen=True)
class RiskParameters:
    """Risk management parameters for signal execution"""
    max_position_size: Optional[Decimal] = None
    max_portfolio_weight: Optional[Decimal] = None  # 0-1
    stop_loss_pct: Optional[Decimal] = None
    take_profit_pct: Optional[Decimal] = None
    max_slippage_pct: Optional[Decimal] = None
    max_drawdown_pct: Optional[Decimal] = None
    var_limit: Optional[Decimal] = None  # Value at Risk limit
    exposure_limit: Optional[Decimal] = None
    correlation_limit: Optional[Decimal] = None


@dataclass(frozen=True)
class ExecutionConstraints:
    """Execution constraints for signal processing"""
    min_liquidity: Optional[Decimal] = None
    max_market_impact: Optional[Decimal] = None
    preferred_venues: List[str] = field(default_factory=list)
    avoid_venues: List[str] = field(default_factory=list)
    execution_style: str = "MARKET"  # MARKET, LIMIT, TWAP, VWAP, etc.
    participation_rate: Optional[Decimal] = None  # For TWAP/VWAP
    urgency: str = "NORMAL"  # LOW, NORMAL, HIGH, URGENT
    hidden_order: bool = False
    iceberg_ratio: Optional[Decimal] = None


@dataclass
class Signal:
    """
    Core Signal entity representing a trading signal with comprehensive
    metadata, risk parameters, and execution tracking.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    parent_signal_id: Optional[UUID] = None  # For related signals
    
    # Signal classification
    signal_type: SignalType = SignalType.ENTRY
    direction: SignalDirection = SignalDirection.HOLD
    strength: SignalStrength = SignalStrength.MODERATE
    status: SignalStatus = SignalStatus.PENDING
    
    # Market data
    symbol: str = ""
    exchange: str = ""
    current_price: Decimal = Decimal("0")
    target_price: Optional[Decimal] = None
    
    # Execution parameters
    quantity: Decimal = Decimal("0")
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP, STOP_LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    
    # Timestamps
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Performance and risk
    metrics: SignalMetrics = field(default_factory=SignalMetrics)
    risk_parameters: RiskParameters = field(default_factory=RiskParameters)
    execution_constraints: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    
    # Execution tracking
    executed_quantity: Decimal = Decimal("0")
    executed_price: Optional[Decimal] = None
    execution_fees: Decimal = Decimal("0")
    slippage: Optional[Decimal] = None
    market_impact: Optional[Decimal] = None
    
    # Source and reasoning
    source: str = ""  # Strategy name or source identifier
    reason: str = ""  # Human-readable reason for signal
    features: Dict[str, Any] = field(default_factory=dict)  # Input features
    model_outputs: Dict[str, Any] = field(default_factory=dict)  # ML model outputs
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    related_positions: List[UUID] = field(default_factory=list)
    
    def __post_init__(self):
        """Post-initialization validation"""
        self._validate_signal()
    
    def _validate_signal(self) -> None:
        """Validate signal integrity"""
        if not self.symbol:
            raise ValueError("Signal symbol cannot be empty")
        
        if self.current_price < 0:
            raise ValueError("Current price cannot be negative")
        
        if self.quantity < 0:
            raise ValueError("Signal quantity cannot be negative")
        
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("Limit price must be positive")
        
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("Stop price must be positive")
        
        if self.expires_at and self.expires_at <= self.generated_at:
            raise ValueError("Expiration time must be after generation time")
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is valid for execution"""
        try:
            self._validate_signal()
            return (
                self.status in [SignalStatus.PENDING, SignalStatus.VALIDATED] and
                (not self.expires_at or self.expires_at > datetime.now(timezone.utc)) and
                self.quantity > 0
            )
        except ValueError:
            return False
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def fill_ratio(self) -> Decimal:
        """Calculate fill ratio (executed / total quantity)"""
        if self.quantity == 0:
            return Decimal("0")
        return self.executed_quantity / self.quantity
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to execute"""
        return max(Decimal("0"), self.quantity - self.executed_quantity)
    
    @property
    def is_fully_executed(self) -> bool:
        """Check if signal is fully executed"""
        return self.executed_quantity >= self.quantity
    
    @property
    def is_partially_executed(self) -> bool:
        """Check if signal is partially executed"""
        return Decimal("0") < self.executed_quantity < self.quantity
    
    @property
    def average_execution_price(self) -> Optional[Decimal]:
        """Get average execution price"""
        return self.executed_price
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of signal"""
        price = self.limit_price or self.current_price
        return self.quantity * price if price > 0 else Decimal("0")
    
    @property
    def executed_notional_value(self) -> Decimal:
        """Calculate executed notional value"""
        if self.executed_price and self.executed_quantity > 0:
            return self.executed_quantity * self.executed_price
        return Decimal("0")
    
    def update_price(self, price: Decimal) -> None:
        """
        Update current market price
        
        Args:
            price: New market price
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        
        self.current_price = price
        self.last_updated = datetime.now(timezone.utc)
    
    def validate(self) -> bool:
        """
        Validate signal for execution
        
        Returns:
            True if signal is valid, False otherwise
        """
        if not self.is_valid:
            self.status = SignalStatus.REJECTED
            return False
        
        if self.is_expired:
            self.status = SignalStatus.EXPIRED
            return False
        
        self.status = SignalStatus.VALIDATED
        self.last_updated = datetime.now(timezone.utc)
        return True
    
    def execute_partial(
        self,
        executed_qty: Decimal,
        executed_price: Decimal,
        fees: Decimal = Decimal("0")
    ) -> None:
        """
        Record partial execution of signal
        
        Args:
            executed_qty: Quantity executed
            executed_price: Price at which executed
            fees: Execution fees
        """
        if executed_qty <= 0:
            raise ValueError("Executed quantity must be positive")
        
        if executed_price <= 0:
            raise ValueError("Executed price must be positive")
        
        if self.executed_quantity + executed_qty > self.quantity:
            raise ValueError("Cannot execute more than signal quantity")
        
        # Update execution tracking
        prev_total_value = self.executed_quantity * (self.executed_price or Decimal("0"))
        new_total_value = prev_total_value + (executed_qty * executed_price)
        self.executed_quantity += executed_qty
        
        # Calculate weighted average price
        if self.executed_quantity > 0:
            self.executed_price = new_total_value / self.executed_quantity
        
        self.execution_fees += fees
        
        # Calculate slippage
        reference_price = self.limit_price or self.current_price
        if reference_price > 0:
            if self.direction == SignalDirection.BUY:
                self.slippage = (executed_price - reference_price) / reference_price
            else:
                self.slippage = (reference_price - executed_price) / reference_price
        
        # Update status
        if self.is_fully_executed:
            self.status = SignalStatus.EXECUTED
            self.executed_at = datetime.now(timezone.utc)
        else:
            self.status = SignalStatus.PARTIALLY_EXECUTED
        
        self.last_updated = datetime.now(timezone.utc)
    
    def cancel(self, reason: str = "") -> None:
        """
        Cancel the signal
        
        Args:
            reason: Reason for cancellation
        """
        if self.status in [SignalStatus.EXECUTED, SignalStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel signal with status {self.status.value}")
        
        self.status = SignalStatus.CANCELLED
        if reason:
            self.metadata["cancellation_reason"] = reason
        self.last_updated = datetime.now(timezone.utc)
    
    def reject(self, reason: str = "") -> None:
        """
        Reject the signal
        
        Args:
            reason: Reason for rejection
        """
        self.status = SignalStatus.REJECTED
        if reason:
            self.metadata["rejection_reason"] = reason
        self.last_updated = datetime.now(timezone.utc)
    
    def expire(self) -> None:
        """Mark signal as expired"""
        self.status = SignalStatus.EXPIRED
        self.last_updated = datetime.now(timezone.utc)
    
    def calculate_expected_pnl(self, exit_price: Decimal) -> Decimal:
        """
        Calculate expected P&L at given exit price
        
        Args:
            exit_price: Expected exit price
            
        Returns:
            Expected P&L
        """
        entry_price = self.limit_price or self.current_price
        
        if self.direction == SignalDirection.BUY:
            return (exit_price - entry_price) * self.quantity
        elif self.direction == SignalDirection.SELL:
            return (entry_price - exit_price) * self.quantity
        else:
            return Decimal("0")
    
    def create_stop_loss_signal(self, stop_price: Decimal) -> Signal:
        """
        Create a stop loss signal based on this signal
        
        Args:
            stop_price: Stop loss price
            
        Returns:
            Stop loss signal
        """
        opposite_direction = (
            SignalDirection.SELL if self.direction == SignalDirection.BUY
            else SignalDirection.BUY
        )
        
        return Signal(
            strategy_id=self.strategy_id,
            parent_signal_id=self.id,
            signal_type=SignalType.STOP_LOSS,
            direction=opposite_direction,
            symbol=self.symbol,
            exchange=self.exchange,
            quantity=self.quantity,
            order_type="STOP",
            stop_price=stop_price,
            source=f"{self.source}_STOP_LOSS",
            reason=f"Stop loss for signal {self.id}",
            related_positions=self.related_positions.copy()
        )
    
    def create_take_profit_signal(self, target_price: Decimal) -> Signal:
        """
        Create a take profit signal based on this signal
        
        Args:
            target_price: Take profit price
            
        Returns:
            Take profit signal
        """
        opposite_direction = (
            SignalDirection.SELL if self.direction == SignalDirection.BUY
            else SignalDirection.BUY
        )
        
        return Signal(
            strategy_id=self.strategy_id,
            parent_signal_id=self.id,
            signal_type=SignalType.TAKE_PROFIT,
            direction=opposite_direction,
            symbol=self.symbol,
            exchange=self.exchange,
            quantity=self.quantity,
            order_type="LIMIT",
            limit_price=target_price,
            source=f"{self.source}_TAKE_PROFIT",
            reason=f"Take profit for signal {self.id}",
            related_positions=self.related_positions.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for serialization"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "parent_signal_id": str(self.parent_signal_id) if self.parent_signal_id else None,
            "signal_type": self.signal_type.value,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "status": self.status.value,
            "symbol": self.symbol,
            "exchange": self.exchange,
            "current_price": str(self.current_price),
            "target_price": str(self.target_price) if self.target_price else None,
            "quantity": str(self.quantity),
            "order_type": self.order_type,
            "time_in_force": self.time_in_force.value,
            "limit_price": str(self.limit_price) if self.limit_price else None,
            "stop_price": str(self.stop_price) if self.stop_price else None,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "last_updated": self.last_updated.isoformat(),
            "executed_quantity": str(self.executed_quantity),
            "executed_price": str(self.executed_price) if self.executed_price else None,
            "execution_fees": str(self.execution_fees),
            "slippage": str(self.slippage) if self.slippage else None,
            "market_impact": str(self.market_impact) if self.market_impact else None,
            "source": self.source,
            "reason": self.reason,
            "features": self.features,
            "model_outputs": self.model_outputs,
            "tags": self.tags,
            "metadata": self.metadata,
            "related_positions": [str(pos_id) for pos_id in self.related_positions],
            "metrics": {
                "confidence": str(self.metrics.confidence),
                "expected_return": str(self.metrics.expected_return) if self.metrics.expected_return else None,
                "expected_risk": str(self.metrics.expected_risk) if self.metrics.expected_risk else None,
                "sharpe_ratio": str(self.metrics.sharpe_ratio) if self.metrics.sharpe_ratio else None,
                "win_probability": str(self.metrics.win_probability) if self.metrics.win_probability else None,
                "risk_reward_ratio": str(self.metrics.risk_reward_ratio) if self.metrics.risk_reward_ratio else None,
                "volatility_forecast": str(self.metrics.volatility_forecast) if self.metrics.volatility_forecast else None,
                "correlation_score": str(self.metrics.correlation_score) if self.metrics.correlation_score else None,
                "momentum_score": str(self.metrics.momentum_score) if self.metrics.momentum_score else None,
                "mean_reversion_score": str(self.metrics.mean_reversion_score) if self.metrics.mean_reversion_score else None,
            },
            "notional_value": str(self.notional_value),
            "executed_notional_value": str(self.executed_notional_value),
            "fill_ratio": str(self.fill_ratio),
            "remaining_quantity": str(self.remaining_quantity),
            "is_valid": self.is_valid,
            "is_expired": self.is_expired,
            "is_fully_executed": self.is_fully_executed,
            "is_partially_executed": self.is_partially_executed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Signal:
        """Create signal from dictionary"""
        # Reconstruct metrics
        metrics_data = data.get("metrics", {})
        metrics = SignalMetrics(
            confidence=Decimal(metrics_data.get("confidence", "0")),
            expected_return=Decimal(metrics_data["expected_return"]) if metrics_data.get("expected_return") else None,
            expected_risk=Decimal(metrics_data["expected_risk"]) if metrics_data.get("expected_risk") else None,
            sharpe_ratio=Decimal(metrics_data["sharpe_ratio"]) if metrics_data.get("sharpe_ratio") else None,
            win_probability=Decimal(metrics_data["win_probability"]) if metrics_data.get("win_probability") else None,
            risk_reward_ratio=Decimal(metrics_data["risk_reward_ratio"]) if metrics_data.get("risk_reward_ratio") else None,
            volatility_forecast=Decimal(metrics_data["volatility_forecast"]) if metrics_data.get("volatility_forecast") else None,
            correlation_score=Decimal(metrics_data["correlation_score"]) if metrics_data.get("correlation_score") else None,
            momentum_score=Decimal(metrics_data["momentum_score"]) if metrics_data.get("momentum_score") else None,
            mean_reversion_score=Decimal(metrics_data["mean_reversion_score"]) if metrics_data.get("mean_reversion_score") else None,
        )
        
        signal = cls(
            id=UUID(data["id"]),
            strategy_id=UUID(data["strategy_id"]),
            parent_signal_id=UUID(data["parent_signal_id"]) if data.get("parent_signal_id") else None,
            signal_type=SignalType(data["signal_type"]),
            direction=SignalDirection(data["direction"]),
            strength=SignalStrength(data["strength"]),
            status=SignalStatus(data["status"]),
            symbol=data["symbol"],
            exchange=data["exchange"],
            current_price=Decimal(data["current_price"]),
            target_price=Decimal(data["target_price"]) if data.get("target_price") else None,
            quantity=Decimal(data["quantity"]),
            order_type=data["order_type"],
            time_in_force=TimeInForce(data["time_in_force"]),
            limit_price=Decimal(data["limit_price"]) if data.get("limit_price") else None,
            stop_price=Decimal(data["stop_price"]) if data.get("stop_price") else None,
            generated_at=datetime.fromisoformat(data["generated_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            executed_at=datetime.fromisoformat(data["executed_at"]) if data.get("executed_at") else None,
            last_updated=datetime.fromisoformat(data["last_updated"]),
            metrics=metrics,
            executed_quantity=Decimal(data["executed_quantity"]),
            executed_price=Decimal(data["executed_price"]) if data.get("executed_price") else None,
            execution_fees=Decimal(data["execution_fees"]),
            slippage=Decimal(data["slippage"]) if data.get("slippage") else None,
            market_impact=Decimal(data["market_impact"]) if data.get("market_impact") else None,
            source=data["source"],
            reason=data["reason"],
            features=data.get("features", {}),
            model_outputs=data.get("model_outputs", {}),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            related_positions=[UUID(pos_id) for pos_id in data.get("related_positions", [])]
        )
        return signal
    
    def __repr__(self) -> str:
        return (
            f"Signal(id={self.id}, symbol={self.symbol}, "
            f"type={self.signal_type.value}, direction={self.direction.value}, "
            f"quantity={self.quantity}, status={self.status.value})"