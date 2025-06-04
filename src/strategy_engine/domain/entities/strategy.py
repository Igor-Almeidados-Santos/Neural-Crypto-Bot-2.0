"""
Strategy Entity - Core domain model for trading strategies

This module implements the Strategy domain entity with comprehensive
configuration, performance tracking, and risk management capabilities.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Union
from uuid import UUID, uuid4
import asyncio
from concurrent.futures import ThreadPoolExecutor


class StrategyStatus(Enum):
    """Strategy status enumeration"""
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"
    BACKTESTING = "BACKTESTING"
    PAPER_TRADING = "PAPER_TRADING"
    LIVE_TRADING = "LIVE_TRADING"


class StrategyType(Enum):
    """Strategy type enumeration"""
    TREND_FOLLOWING = "TREND_FOLLOWING"
    MEAN_REVERSION = "MEAN_REVERSION"
    STATISTICAL_ARBITRAGE = "STATISTICAL_ARBITRAGE"
    MARKET_MAKING = "MARKET_MAKING"
    MOMENTUM = "MOMENTUM"
    PAIRS_TRADING = "PAIRS_TRADING"
    MACHINE_LEARNING = "MACHINE_LEARNING"
    REINFORCEMENT_LEARNING = "REINFORCEMENT_LEARNING"
    MULTI_STRATEGY = "MULTI_STRATEGY"
    CUSTOM = "CUSTOM"


class RiskTolerance(Enum):
    """Risk tolerance enumeration"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    SPECULATIVE = "SPECULATIVE"


@dataclass(frozen=True)
class StrategyParameters:
    """Configuration parameters for strategy execution"""
    # Trading parameters
    max_position_size: Decimal = Decimal("1000")
    max_positions: int = 10
    position_sizing_method: str = "FIXED"  # FIXED, PERCENT_RISK, KELLY, VOLATILITY_TARGET
    base_position_size: Decimal = Decimal("100")
    
    # Risk management
    max_daily_loss: Optional[Decimal] = None
    max_drawdown: Optional[Decimal] = None
    stop_loss_pct: Optional[Decimal] = None
    take_profit_pct: Optional[Decimal] = None
    max_correlation: Decimal = Decimal("0.7")
    max_leverage: Decimal = Decimal("1.0")
    
    # Timing parameters
    signal_frequency: str = "1m"  # 1s, 1m, 5m, 15m, 1h, 4h, 1d
    execution_delay: int = 0  # seconds
    cooldown_period: int = 300  # seconds between trades on same symbol
    
    # Market conditions
    min_volume: Optional[Decimal] = None
    min_liquidity: Optional[Decimal] = None
    allowed_exchanges: List[str] = field(default_factory=list)
    blacklisted_symbols: Set[str] = field(default_factory=set)
    whitelisted_symbols: Set[str] = field(default_factory=set)
    
    # Advanced parameters
    use_paper_trading: bool = True
    enable_shorting: bool = False
    enable_margin: bool = False
    enable_options: bool = False
    enable_futures: bool = False
    
    # Strategy-specific parameters
    lookback_period: int = 20
    threshold_buy: Optional[Decimal] = None
    threshold_sell: Optional[Decimal] = None
    rebalance_frequency: str = "daily"
    warm_up_period: int = 100
    
    def __post_init__(self):
        """Validate parameters"""
        if self.max_position_size <= 0:
            raise ValueError("Max position size must be positive")
        if self.max_positions <= 0:
            raise ValueError("Max positions must be positive")
        if not (0 <= self.max_leverage <= 10):
            raise ValueError("Max leverage must be between 0 and 10")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance tracking metrics for strategy"""
    # Returns
    total_return: Decimal = Decimal("0")
    annual_return: Decimal = Decimal("0")
    monthly_return: Decimal = Decimal("0")
    daily_return: Decimal = Decimal("0")
    
    # Risk metrics
    volatility: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    var_95: Decimal = Decimal("0")  # Value at Risk 95%
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    
    # Timing metrics
    avg_trade_duration: Optional[float] = None  # hours
    avg_time_in_market: Decimal = Decimal("0")  # percentage
    
    # Advanced metrics
    alpha: Decimal = Decimal("0")
    beta: Decimal = Decimal("0")
    information_ratio: Decimal = Decimal("0")
    tracking_error: Decimal = Decimal("0")
    
    @property
    def expectancy(self) -> Decimal:
        """Calculate expectancy per trade"""
        if self.total_trades == 0:
            return Decimal("0")
        return (self.win_rate * self.avg_win) - ((Decimal("1") - self.win_rate) * abs(self.avg_loss))


@dataclass
class StrategyState:
    """Current state of strategy execution"""
    is_running: bool = False
    is_warmed_up: bool = False
    last_signal_time: Optional[datetime] = None
    last_execution_time: Optional[datetime] = None
    last_error: Optional[str] = None
    error_count: int = 0
    restart_count: int = 0
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    
    # Real-time metrics
    current_positions: int = 0
    current_exposure: Decimal = Decimal("0")
    available_capital: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Defines the interface that all strategies must implement.
    """
    
    @abstractmethod
    async def initialize(self, market_data: Dict[str, Any], **kwargs) -> bool:
        """
        Initialize strategy with market data and configuration
        
        Args:
            market_data: Historical and current market data
            **kwargs: Additional initialization parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Any]:
        """
        Generate trading signals based on current market data
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    async def update_state(self, market_data: Dict[str, Any]) -> None:
        """
        Update internal strategy state with new market data
        
        Args:
            market_data: New market data
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self, 
        symbol: str, 
        price: Decimal, 
        signal_strength: Decimal,
        available_capital: Decimal
    ) -> Decimal:
        """
        Calculate optimal position size for a trade
        
        Args:
            symbol: Trading symbol
            price: Current price
            signal_strength: Strength of the signal (0-1)
            available_capital: Available capital for trading
            
        Returns:
            Position size
        """
        pass
    
    @abstractmethod
    def validate_signal(self, signal: Any, market_data: Dict[str, Any]) -> bool:
        """
        Validate if a signal should be executed
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid, False otherwise
        """
        pass


@dataclass
class Strategy:
    """
    Core Strategy entity representing a trading strategy with comprehensive
    configuration, state management, and performance tracking.
    """
    
    # Identity
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    
    # Classification
    strategy_type: StrategyType = StrategyType.CUSTOM
    status: StrategyStatus = StrategyStatus.INACTIVE
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    
    # Configuration
    parameters: StrategyParameters = field(default_factory=StrategyParameters)
    symbols: Set[str] = field(default_factory=set)
    exchanges: Set[str] = field(default_factory=set)
    
    # State and performance
    state: StrategyState = field(default_factory=StrategyState)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    
    # Implementation
    implementation: Optional[BaseStrategy] = None
    module_path: str = ""
    class_name: str = ""
    
    # Capital allocation
    allocated_capital: Decimal = Decimal("10000")
    used_capital: Decimal = Decimal("0")
    max_capital_usage: Decimal = Decimal("0.8")  # 80% max usage
    
    # Metadata
    author: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Dependencies
    required_indicators: List[str] = field(default_factory=list)
    required_data_sources: List[str] = field(default_factory=list)
    dependencies: List[UUID] = field(default_factory=list)  # Other strategy IDs
    
    def __post_init__(self):
        """Post-initialization validation"""
        self._validate_strategy()
    
    def _validate_strategy(self) -> None:
        """Validate strategy configuration"""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")
        
        if self.allocated_capital <= 0:
            raise ValueError("Allocated capital must be positive")
        
        if not (0 <= self.max_capital_usage <= 1):
            raise ValueError("Max capital usage must be between 0 and 1")
        
        if not self.symbols and self.status != StrategyStatus.INACTIVE:
            raise ValueError("Active strategy must have at least one symbol")
    
    @property
    def available_capital(self) -> Decimal:
        """Calculate available capital for new positions"""
        max_usable = self.allocated_capital * self.max_capital_usage
        return max(Decimal("0"), max_usable - self.used_capital)
    
    @property
    def capital_utilization(self) -> Decimal:
        """Calculate capital utilization percentage"""
        if self.allocated_capital == 0:
            return Decimal("0")
        return (self.used_capital / self.allocated_capital) * Decimal("100")
    
    @property
    def is_active(self) -> bool:
        """Check if strategy is currently active"""
        return self.status in [
            StrategyStatus.ACTIVE,
            StrategyStatus.PAPER_TRADING,
            StrategyStatus.LIVE_TRADING
        ]
    
    @property
    def is_healthy(self) -> bool:
        """Check if strategy is in healthy state"""
        return (
            self.status not in [StrategyStatus.ERROR, StrategyStatus.STOPPED] and
            self.state.error_count < 5 and
            (not self.state.last_error or 
             (datetime.now(timezone.utc) - self.state.last_execution_time).total_seconds() < 3600
             if self.state.last_execution_time else True)
        )
    
    @property
    def uptime_percentage(self) -> Decimal:
        """Calculate strategy uptime percentage"""
        if not self.started_at:
            return Decimal("0")
        
        total_time = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        if total_time == 0:
            return Decimal("100")
        
        # Simple uptime calculation - in real implementation would track downtime
        error_penalty = min(Decimal("50"), self.state.error_count * Decimal("5"))
        return max(Decimal("0"), Decimal("100") - error_penalty)
    
    async def start(self, market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start strategy execution
        
        Args:
            market_data: Initial market data
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.status in [StrategyStatus.ACTIVE, StrategyStatus.LIVE_TRADING]:
            return True
        
        try:
            # Initialize implementation if available
            if self.implementation:
                success = await self.implementation.initialize(market_data or {})
                if not success:
                    self.state.last_error = "Strategy initialization failed"
                    self.status = StrategyStatus.ERROR
                    return False
            
            # Update state
            self.status = StrategyStatus.ACTIVE
            self.started_at = datetime.now(timezone.utc)
            self.state.is_running = True
            self.state.last_error = None
            self.updated_at = datetime.now(timezone.utc)
            
            return True
            
        except Exception as e:
            self.state.last_error = str(e)
            self.state.error_count += 1
            self.status = StrategyStatus.ERROR
            return False
    
    async def stop(self, reason: str = "") -> bool:
        """
        Stop strategy execution
        
        Args:
            reason: Reason for stopping
            
        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            self.status = StrategyStatus.STOPPED
            self.stopped_at = datetime.now(timezone.utc)
            self.state.is_running = False
            
            if reason:
                self.metadata["stop_reason"] = reason
                self.metadata["stopped_at"] = self.stopped_at.isoformat()
            
            self.updated_at = datetime.now(timezone.utc)
            return True
            
        except Exception as e:
            self.state.last_error = str(e)
            self.state.error_count += 1
            return False
    
    async def pause(self) -> bool:
        """
        Pause strategy execution
        
        Returns:
            True if paused successfully, False otherwise
        """
        if self.status not in [StrategyStatus.ACTIVE, StrategyStatus.LIVE_TRADING]:
            return False
        
        self.status = StrategyStatus.PAUSED
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    async def resume(self) -> bool:
        """
        Resume strategy execution
        
        Returns:
            True if resumed successfully, False otherwise
        """
        if self.status != StrategyStatus.PAUSED:
            return False
        
        self.status = StrategyStatus.ACTIVE
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def add_symbol(self, symbol: str) -> None:
        """
        Add symbol to strategy watchlist
        
        Args:
            symbol: Symbol to add
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        self.symbols.add(symbol.upper())
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_symbol(self, symbol: str) -> None:
        """
        Remove symbol from strategy watchlist
        
        Args:
            symbol: Symbol to remove
        """
        self.symbols.discard(symbol.upper())
        self.updated_at = datetime.now(timezone.utc)
    
    def add_exchange(self, exchange: str) -> None:
        """
        Add exchange to strategy
        
        Args:
            exchange: Exchange to add
        """
        if not exchange:
            raise ValueError("Exchange cannot be empty")
        
        self.exchanges.add(exchange.lower())
        self.updated_at = datetime.now(timezone.utc)
    
    def remove_exchange(self, exchange: str) -> None:
        """
        Remove exchange from strategy
        
        Args:
            exchange: Exchange to remove
        """
        self.exchanges.discard(exchange.lower())
        self.updated_at = datetime.now(timezone.utc)
    
    def allocate_capital(self, amount: Decimal) -> bool:
        """
        Allocate additional capital to strategy
        
        Args:
            amount: Amount to allocate
            
        Returns:
            True if allocation successful, False otherwise
        """
        if amount <= 0:
            return False
        
        if self.used_capital + amount > self.allocated_capital * self.max_capital_usage:
            return False
        
        self.used_capital += amount
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def deallocate_capital(self, amount: Decimal) -> bool:
        """
        Deallocate capital from strategy
        
        Args:
            amount: Amount to deallocate
            
        Returns:
            True if deallocation successful, False otherwise
        """
        if amount <= 0:
            return False
        
        if amount > self.used_capital:
            return False
        
        self.used_capital -= amount
        self.updated_at = datetime.now(timezone.utc)
        return True
    
    def update_performance(self, metrics: PerformanceMetrics) -> None:
        """
        Update strategy performance metrics
        
        Args:
            metrics: New performance metrics
        """
        self.performance = metrics
        self.updated_at = datetime.now(timezone.utc)
    
    def record_error(self, error: str) -> None:
        """
        Record an error occurrence
        
        Args:
            error: Error message
        """
        self.state.last_error = error
        self.state.error_count += 1
        self.updated_at = datetime.now(timezone.utc)
        
        # Auto-pause if too many errors
        if self.state.error_count >= 10:
            self.status = StrategyStatus.ERROR
    
    def reset_errors(self) -> None:
        """Reset error count and clear last error"""
        self.state.error_count = 0
        self.state.last_error = None
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary for serialization"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "risk_tolerance": self.risk_tolerance.value,
            "symbols": list(self.symbols),
            "exchanges": list(self.exchanges),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "module_path": self.module_path,
            "class_name": self.class_name,
            "allocated_capital": str(self.allocated_capital),
            "used_capital": str(self.used_capital),
            "available_capital": str(self.available_capital),
            "capital_utilization": str(self.capital_utilization),
            "max_capital_usage": str(self.max_capital_usage),
            "author": self.author,
            "tags": self.tags,
            "metadata": self.metadata,
            "required_indicators": self.required_indicators,
            "required_data_sources": self.required_data_sources,
            "dependencies": [str(dep_id) for dep_id in self.dependencies],
            "is_active": self.is_active,
            "is_healthy": self.is_healthy,
            "uptime_percentage": str(self.uptime_percentage),
            # Include nested objects
            "parameters": {
                "max_position_size": str(self.parameters.max_position_size),
                "max_positions": self.parameters.max_positions,
                "position_sizing_method": self.parameters.position_sizing_method,
                "base_position_size": str(self.parameters.base_position_size),
                "max_daily_loss": str(self.parameters.max_daily_loss) if self.parameters.max_daily_loss else None,
                "max_drawdown": str(self.parameters.max_drawdown) if self.parameters.max_drawdown else None,
                "stop_loss_pct": str(self.parameters.stop_loss_pct) if self.parameters.stop_loss_pct else None,
                "take_profit_pct": str(self.parameters.take_profit_pct) if self.parameters.take_profit_pct else None,
                "max_correlation": str(self.parameters.max_correlation),
                "max_leverage": str(self.parameters.max_leverage),
                "signal_frequency": self.parameters.signal_frequency,
                "execution_delay": self.parameters.execution_delay,
                "cooldown_period": self.parameters.cooldown_period,
                "min_volume": str(self.parameters.min_volume) if self.parameters.min_volume else None,
                "min_liquidity": str(self.parameters.min_liquidity) if self.parameters.min_liquidity else None,
                "allowed_exchanges": self.parameters.allowed_exchanges,
                "blacklisted_symbols": list(self.parameters.blacklisted_symbols),
                "whitelisted_symbols": list(self.parameters.whitelisted_symbols),
                "use_paper_trading": self.parameters.use_paper_trading,
                "enable_shorting": self.parameters.enable_shorting,
                "enable_margin": self.parameters.enable_margin,
                "enable_options": self.parameters.enable_options,
                "enable_futures": self.parameters.enable_futures,
                "lookback_period": self.parameters.lookback_period,
                "threshold_buy": str(self.parameters.threshold_buy) if self.parameters.threshold_buy else None,
                "threshold_sell": str(self.parameters.threshold_sell) if self.parameters.threshold_sell else None,
                "rebalance_frequency": self.parameters.rebalance_frequency,
                "warm_up_period": self.parameters.warm_up_period
            },
            "state": {
                "is_running": self.state.is_running,
                "is_warmed_up": self.state.is_warmed_up,
                "last_signal_time": self.state.last_signal_time.isoformat() if self.state.last_signal_time else None,
                "last_execution_time": self.state.last_execution_time.isoformat() if self.state.last_execution_time else None,
                "last_error": self.state.last_error,
                "error_count": self.state.error_count,
                "restart_count": self.state.restart_count,
                "cpu_usage": self.state.cpu_usage,
                "memory_usage": self.state.memory_usage,
                "network_latency": self.state.network_latency,
                "current_positions": self.state.current_positions,
                "current_exposure": str(self.state.current_exposure),
                "available_capital": str(self.state.available_capital),
                "unrealized_pnl": str(self.state.unrealized_pnl),
                "realized_pnl": str(self.state.realized_pnl)
            },
            "performance": {
                "total_return": str(self.performance.total_return),
                "annual_return": str(self.performance.annual_return),
                "monthly_return": str(self.performance.monthly_return),
                "daily_return": str(self.performance.daily_return),
                "volatility": str(self.performance.volatility),
                "sharpe_ratio": str(self.performance.sharpe_ratio),
                "sortino_ratio": str(self.performance.sortino_ratio),
                "calmar_ratio": str(self.performance.calmar_ratio),
                "max_drawdown": str(self.performance.max_drawdown),
                "var_95": str(self.performance.var_95),
                "total_trades": self.performance.total_trades,
                "winning_trades": self.performance.winning_trades,
                "losing_trades": self.performance.losing_trades,
                "win_rate": str(self.performance.win_rate),
                "avg_win": str(self.performance.avg_win),
                "avg_loss": str(self.performance.avg_loss),
                "profit_factor": str(self.performance.profit_factor),
                "avg_trade_duration": self.performance.avg_trade_duration,
                "avg_time_in_market": str(self.performance.avg_time_in_market),
                "alpha": str(self.performance.alpha),
                "beta": str(self.performance.beta),
                "information_ratio": str(self.performance.information_ratio),
                "tracking_error": str(self.performance.tracking_error),
                "expectancy": str(self.performance.expectancy)
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Strategy:
        """Create strategy from dictionary"""
        # Reconstruct parameters
        params_data = data.get("parameters", {})
        parameters = StrategyParameters(
            max_position_size=Decimal(params_data.get("max_position_size", "1000")),
            max_positions=params_data.get("max_positions", 10),
            position_sizing_method=params_data.get("position_sizing_method", "FIXED"),
            base_position_size=Decimal(params_data.get("base_position_size", "100")),
            max_daily_loss=Decimal(params_data["max_daily_loss"]) if params_data.get("max_daily_loss") else None,
            max_drawdown=Decimal(params_data["max_drawdown"]) if params_data.get("max_drawdown") else None,
            stop_loss_pct=Decimal(params_data["stop_loss_pct"]) if params_data.get("stop_loss_pct") else None,
            take_profit_pct=Decimal(params_data["take_profit_pct"]) if params_data.get("take_profit_pct") else None,
            max_correlation=Decimal(params_data.get("max_correlation", "0.7")),
            max_leverage=Decimal(params_data.get("max_leverage", "1.0")),
            signal_frequency=params_data.get("signal_frequency", "1m"),
            execution_delay=params_data.get("execution_delay", 0),
            cooldown_period=params_data.get("cooldown_period", 300),
            min_volume=Decimal(params_data["min_volume"]) if params_data.get("min_volume") else None,
            min_liquidity=Decimal(params_data["min_liquidity"]) if params_data.get("min_liquidity") else None,
            allowed_exchanges=params_data.get("allowed_exchanges", []),
            blacklisted_symbols=set(params_data.get("blacklisted_symbols", [])),
            whitelisted_symbols=set(params_data.get("whitelisted_symbols", [])),
            use_paper_trading=params_data.get("use_paper_trading", True),
            enable_shorting=params_data.get("enable_shorting", False),
            enable_margin=params_data.get("enable_margin", False),
            enable_options=params_data.get("enable_options", False),
            enable_futures=params_data.get("enable_futures", False),
            lookback_period=params_data.get("lookback_period", 20),
            threshold_buy=Decimal(params_data["threshold_buy"]) if params_data.get("threshold_buy") else None,
            threshold_sell=Decimal(params_data["threshold_sell"]) if params_data.get("threshold_sell") else None,
            rebalance_frequency=params_data.get("rebalance_frequency", "daily"),
            warm_up_period=params_data.get("warm_up_period", 100)
        )
        
        # Reconstruct state
        state_data = data.get("state", {})
        state = StrategyState(
            is_running=state_data.get("is_running", False),
            is_warmed_up=state_data.get("is_warmed_up", False),
            last_signal_time=datetime.fromisoformat(state_data["last_signal_time"]) if state_data.get("last_signal_time") else None,
            last_execution_time=datetime.fromisoformat(state_data["last_execution_time"]) if state_data.get("last_execution_time") else None,
            last_error=state_data.get("last_error"),
            error_count=state_data.get("error_count", 0),
            restart_count=state_data.get("restart_count", 0),
            cpu_usage=state_data.get("cpu_usage", 0.0),
            memory_usage=state_data.get("memory_usage", 0.0),
            network_latency=state_data.get("network_latency", 0.0),
            current_positions=state_data.get("current_positions", 0),
            current_exposure=Decimal(state_data.get("current_exposure", "0")),
            available_capital=Decimal(state_data.get("available_capital", "0")),
            unrealized_pnl=Decimal(state_data.get("unrealized_pnl", "0")),
            realized_pnl=Decimal(state_data.get("realized_pnl", "0"))
        )
        
        # Reconstruct performance
        perf_data = data.get("performance", {})
        performance = PerformanceMetrics(
            total_return=Decimal(perf_data.get("total_return", "0")),
            annual_return=Decimal(perf_data.get("annual_return", "0")),
            monthly_return=Decimal(perf_data.get("monthly_return", "0")),
            daily_return=Decimal(perf_data.get("daily_return", "0")),
            volatility=Decimal(perf_data.get("volatility", "0")),
            sharpe_ratio=Decimal(perf_data.get("sharpe_ratio", "0")),
            sortino_ratio=Decimal(perf_data.get("sortino_ratio", "0")),
            calmar_ratio=Decimal(perf_data.get("calmar_ratio", "0")),
            max_drawdown=Decimal(perf_data.get("max_drawdown", "0")),
            var_95=Decimal(perf_data.get("var_95", "0")),
            total_trades=perf_data.get("total_trades", 0),
            winning_trades=perf_data.get("winning_trades", 0),
            losing_trades=perf_data.get("losing_trades", 0),
            win_rate=Decimal(perf_data.get("win_rate", "0")),
            avg_win=Decimal(perf_data.get("avg_win", "0")),
            avg_loss=Decimal(perf_data.get("avg_loss", "0")),
            profit_factor=Decimal(perf_data.get("profit_factor", "0")),
            avg_trade_duration=perf_data.get("avg_trade_duration"),
            avg_time_in_market=Decimal(perf_data.get("avg_time_in_market", "0")),
            alpha=Decimal(perf_data.get("alpha", "0")),
            beta=Decimal(perf_data.get("beta", "0")),
            information_ratio=Decimal(perf_data.get("information_ratio", "0")),
            tracking_error=Decimal(perf_data.get("tracking_error", "0"))
        )
        
        strategy = cls(
            id=UUID(data["id"]),
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            strategy_type=StrategyType(data["strategy_type"]),
            status=StrategyStatus(data["status"]),
            risk_tolerance=RiskTolerance(data.get("risk_tolerance", "MODERATE")),
            parameters=parameters,
            symbols=set(data.get("symbols", [])),
            exchanges=set(data.get("exchanges", [])),
            state=state,
            performance=performance,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            stopped_at=datetime.fromisoformat(data["stopped_at"]) if data.get("stopped_at") else None,
            module_path=data.get("module_path", ""),
            class_name=data.get("class_name", ""),
            allocated_capital=Decimal(data.get("allocated_capital", "10000")),
            used_capital=Decimal(data.get("used_capital", "0")),
            max_capital_usage=Decimal(data.get("max_capital_usage", "0.8")),
            author=data.get("author", ""),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            required_indicators=data.get("required_indicators", []),
            required_data_sources=data.get("required_data_sources", []),
            dependencies=[UUID(dep_id) for dep_id in data.get("dependencies", [])]
        )
        return strategy
    
    def __repr__(self) -> str:
        return (
            f"Strategy(id={self.id}, name='{self.name}', "
            f"type={self.strategy_type.value}, status={self.status.value}, "
            f"symbols={len(self.symbols)}, capital={self.allocated_capital})"
        )