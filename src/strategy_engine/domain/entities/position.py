"""
Position Entity - Core domain model for position management

This module implements the Position domain entity with comprehensive 
risk management, position tracking, and P&L calculation capabilities.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4


class PositionType(Enum):
    """Position type enumeration"""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    PENDING = "PENDING"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class Trade:
    """Individual trade within a position"""
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str = ""
    side: str = ""  # BUY or SELL
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    fee_currency: str = ""
    exchange: str = ""
    order_id: Optional[str] = None
    execution_id: Optional[str] = None
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of the trade"""
        return self.quantity * self.price
    
    def __post_init__(self):
        """Validate trade data"""
        if self.quantity <= 0:
            raise ValueError("Trade quantity must be positive")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")


@dataclass
class Position:
    """
    Core Position entity representing a trading position with comprehensive
    risk management and performance tracking capabilities.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    symbol: str = ""
    
    # Position attributes
    position_type: PositionType = PositionType.NEUTRAL
    status: PositionStatus = PositionStatus.PENDING
    
    # Quantities and pricing
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    
    # Timestamps
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Trading history
    trades: List[Trade] = field(default_factory=list)
    
    # Risk management
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    max_loss_amount: Optional[Decimal] = None
    max_gain_amount: Optional[Decimal] = None
    
    # Performance tracking
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    
    # Metadata
    exchange: str = ""
    margin_used: Decimal = Decimal("0")
    leverage: Decimal = Decimal("1")
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation"""
        self._validate_position()
    
    def _validate_position(self) -> None:
        """Validate position integrity"""
        if not self.symbol:
            raise ValueError("Position symbol cannot be empty")
        
        if self.leverage <= 0:
            raise ValueError("Leverage must be positive")
        
        if self.quantity < 0:
            raise ValueError("Position quantity cannot be negative")
    
    def add_trade(self, trade: Trade) -> None:
        """
        Add a trade to the position and update position metrics
        
        Args:
            trade: Trade to add to position
        """
        if trade.symbol != self.symbol:
            raise ValueError(f"Trade symbol {trade.symbol} doesn't match position symbol {self.symbol}")
        
        self.trades.append(trade)
        self._update_position_from_trades()
        self.last_updated = datetime.now(timezone.utc)
    
    def _update_position_from_trades(self) -> None:
        """Update position metrics based on all trades"""
        if not self.trades:
            return
        
        # Calculate net position
        total_quantity = Decimal("0")
        weighted_price_sum = Decimal("0")
        total_fees = Decimal("0")
        
        for trade in self.trades:
            trade_quantity = trade.quantity if trade.side == "BUY" else -trade.quantity
            total_quantity += trade_quantity
            
            if trade.side == "BUY":
                weighted_price_sum += trade.quantity * trade.price
            
            total_fees += trade.fee
        
        # Update position attributes
        self.quantity = abs(total_quantity)
        self.fees_paid = total_fees
        
        # Determine position type
        if total_quantity > 0:
            self.position_type = PositionType.LONG
        elif total_quantity < 0:
            self.position_type = PositionType.SHORT
        else:
            self.position_type = PositionType.NEUTRAL
            self.status = PositionStatus.CLOSED
            self.closed_at = datetime.now(timezone.utc)
        
        # Calculate average entry price for long positions
        if self.position_type == PositionType.LONG and self.quantity > 0:
            buy_quantity = sum(trade.quantity for trade in self.trades if trade.side == "BUY")
            if buy_quantity > 0:
                self.average_entry_price = weighted_price_sum / buy_quantity
        
        # Update status
        if self.quantity > 0 and self.status == PositionStatus.PENDING:
            self.status = PositionStatus.OPEN
    
    def update_current_price(self, price: Decimal) -> None:
        """
        Update current market price and recalculate unrealized P&L
        
        Args:
            price: Current market price
        """
        if price <= 0:
            raise ValueError("Price must be positive")
        
        self.current_price = price
        self.last_updated = datetime.now(timezone.utc)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L based on current price"""
        if self.quantity == 0 or self.current_price == 0:
            return Decimal("0")
        
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.average_entry_price) * self.quantity
        elif self.position_type == PositionType.SHORT:
            return (self.average_entry_price - self.current_price) * self.quantity
        
        return Decimal("0")
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def net_pnl(self) -> Decimal:
        """Calculate net P&L after fees"""
        return self.total_pnl - self.fees_paid
    
    @property
    def return_percentage(self) -> Decimal:
        """Calculate return percentage based on initial investment"""
        if self.average_entry_price == 0 or self.quantity == 0:
            return Decimal("0")
        
        initial_value = self.average_entry_price * self.quantity
        return (self.net_pnl / initial_value) * Decimal("100")
    
    @property
    def notional_value(self) -> Decimal:
        """Calculate current notional value of position"""
        return self.current_price * self.quantity
    
    @property
    def margin_requirement(self) -> Decimal:
        """Calculate margin requirement for leveraged position"""
        if self.leverage <= 0:
            return self.notional_value
        return self.notional_value / self.leverage
    
    def check_stop_loss(self) -> bool:
        """Check if stop loss should be triggered"""
        if not self.stop_loss or self.quantity == 0:
            return False
        
        if self.position_type == PositionType.LONG:
            return self.current_price <= self.stop_loss
        elif self.position_type == PositionType.SHORT:
            return self.current_price >= self.stop_loss
        
        return False
    
    def check_take_profit(self) -> bool:
        """Check if take profit should be triggered"""
        if not self.take_profit or self.quantity == 0:
            return False
        
        if self.position_type == PositionType.LONG:
            return self.current_price >= self.take_profit
        elif self.position_type == PositionType.SHORT:
            return self.current_price <= self.take_profit
        
        return False
    
    def check_max_loss(self) -> bool:
        """Check if maximum loss threshold is breached"""
        if not self.max_loss_amount:
            return False
        
        return self.net_pnl <= -abs(self.max_loss_amount)
    
    def check_max_gain(self) -> bool:
        """Check if maximum gain threshold is reached"""
        if not self.max_gain_amount:
            return False
        
        return self.net_pnl >= self.max_gain_amount
    
    def should_close(self) -> bool:
        """Determine if position should be closed based on risk rules"""
        return (
            self.check_stop_loss() or 
            self.check_take_profit() or 
            self.check_max_loss() or 
            self.check_max_gain()
        )
    
    def close_position(self, close_price: Optional[Decimal] = None) -> None:
        """
        Close the position
        
        Args:
            close_price: Price at which position is closed
        """
        if close_price:
            self.update_current_price(close_price)
        
        # Move unrealized to realized P&L
        self.realized_pnl += self.unrealized_pnl
        self.quantity = Decimal("0")
        self.status = PositionStatus.CLOSED
        self.closed_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "symbol": self.symbol,
            "position_type": self.position_type.value,
            "status": self.status.value,
            "quantity": str(self.quantity),
            "average_entry_price": str(self.average_entry_price),
            "current_price": str(self.current_price),
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "last_updated": self.last_updated.isoformat(),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "max_loss_amount": str(self.max_loss_amount) if self.max_loss_amount else None,
            "max_gain_amount": str(self.max_gain_amount) if self.max_gain_amount else None,
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "total_pnl": str(self.total_pnl),
            "net_pnl": str(self.net_pnl),
            "fees_paid": str(self.fees_paid),
            "return_percentage": str(self.return_percentage),
            "notional_value": str(self.notional_value),
            "exchange": self.exchange,
            "margin_used": str(self.margin_used),
            "leverage": str(self.leverage),
            "notes": self.notes,
            "tags": self.tags,
            "metadata": self.metadata,
            "trades_count": len(self.trades)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Position:
        """Create position from dictionary"""
        position = cls(
            id=UUID(data["id"]),
            strategy_id=UUID(data["strategy_id"]),
            symbol=data["symbol"],
            position_type=PositionType(data["position_type"]),
            status=PositionStatus(data["status"]),
            quantity=Decimal(data["quantity"]),
            average_entry_price=Decimal(data["average_entry_price"]),
            current_price=Decimal(data["current_price"]),
            opened_at=datetime.fromisoformat(data["opened_at"]),
            closed_at=datetime.fromisoformat(data["closed_at"]) if data.get("closed_at") else None,
            last_updated=datetime.fromisoformat(data["last_updated"]),
            stop_loss=Decimal(data["stop_loss"]) if data.get("stop_loss") else None,
            take_profit=Decimal(data["take_profit"]) if data.get("take_profit") else None,
            max_loss_amount=Decimal(data["max_loss_amount"]) if data.get("max_loss_amount") else None,
            max_gain_amount=Decimal(data["max_gain_amount"]) if data.get("max_gain_amount") else None,
            realized_pnl=Decimal(data["realized_pnl"]),
            fees_paid=Decimal(data["fees_paid"]),
            exchange=data["exchange"],
            margin_used=Decimal(data["margin_used"]),
            leverage=Decimal(data["leverage"]),
            notes=data.get("notes", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        return position
    
    def __repr__(self) -> str:
        return (
            f"Position(id={self.id}, symbol={self.symbol}, "
            f"type={self.position_type.value}, status={self.status.value}, "
            f"quantity={self.quantity}, pnl={self.net_pnl})"
        )