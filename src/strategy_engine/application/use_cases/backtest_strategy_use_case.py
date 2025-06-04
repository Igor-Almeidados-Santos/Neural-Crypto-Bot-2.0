"""
Backtest Strategy Use Case - Comprehensive backtesting implementation

This module implements advanced backtesting capabilities with realistic
market simulation, comprehensive metrics, and optimization features.
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import asyncio
import logging
from dataclasses import dataclass, field
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Import domain entities and services
from ..domain.entities.strategy import Strategy, StrategyStatus, BaseStrategy
from ..domain.entities.position import Position, PositionStatus, PositionType, Trade
from ..domain.entities.signal import Signal, SignalStatus, SignalDirection, SignalType
from ..application.services.backtest_service import BacktestService, BacktestResult


class BacktestError(Exception):
    """Base exception for backtest errors"""
    pass


class DataError(BacktestError):
    """Raised when there are data quality issues"""
    pass


class ValidationError(BacktestError):
    """Raised when backtest parameters are invalid"""
    pass


@dataclass
class BacktestConfiguration:
    """Advanced configuration for backtesting"""
    # Basic parameters
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("100000")
    
    # Trading costs
    commission_rate: Decimal = Decimal("0.001")  # 0.1%
    slippage_rate: Decimal = Decimal("0.0005")   # 0.05%
    spread_cost: Decimal = Decimal("0.0002")     # 0.02%
    
    # Market simulation
    realistic_execution: bool = True
    market_impact: bool = True
    liquidity_constraints: bool = True
    
    # Risk management
    max_drawdown_stop: Optional[Decimal] = None
    daily_loss_limit: Optional[Decimal] = None
    position_size_limit: Optional[Decimal] = None
    
    # Performance analysis
    benchmark_symbol: Optional[str] = None
    risk_free_rate: Decimal = Decimal("0.02")  # 2% annual
    
    # Data quality
    min_data_quality: float = 0.95  # 95% data coverage required
    handle_missing_data: str = "interpolate"  # interpolate, skip, error
    
    # Optimization
    warm_up_period: int = 100  # bars for strategy warm-up
    parallel_execution: bool = True
    max_workers: int = 4


@dataclass
class MarketDataPoint:
    """Single market data point with comprehensive information"""
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    
    # Additional market data
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread: Optional[Decimal] = None
    
    # Derived indicators (calculated during backtest)
    sma_20: Optional[Decimal] = None
    sma_50: Optional[Decimal] = None
    rsi: Optional[Decimal] = None
    macd: Optional[Decimal] = None
    bb_upper: Optional[Decimal] = None
    bb_lower: Optional[Decimal] = None
    
    # Market microstructure
    order_book_imbalance: Optional[Decimal] = None
    trade_intensity: Optional[Decimal] = None
    volatility: Optional[Decimal] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for strategy consumption"""
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'open': float(self.open),
            'high': float(self.high),
            'low': float(self.low),
            'close': float(self.close),
            'volume': float(self.volume),
            'bid': float(self.bid) if self.bid else None,
            'ask': float(self.ask) if self.ask else None,
            'spread': float(self.spread) if self.spread else None,
            'sma_20': float(self.sma_20) if self.sma_20 else None,
            'sma_50': float(self.sma_50) if self.sma_50 else None,
            'rsi': float(self.rsi) if self.rsi else None,
            'macd': float(self.macd) if self.macd else None,
            'bb_upper': float(self.bb_upper) if self.bb_upper else None,
            'bb_lower': float(self.bb_lower) if self.bb_lower else None,
            'order_book_imbalance': float(self.order_book_imbalance) if self.order_book_imbalance else None,
            'trade_intensity': float(self.trade_intensity) if self.trade_intensity else None,
            'volatility': float(self.volatility) if self.volatility else None
        }


@dataclass
class BacktestPosition:
    """Enhanced position tracking for backtesting"""
    symbol: str
    strategy_id: UUID
    opened_at: datetime
    
    # Position details
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    
    # P&L tracking
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    
    # Trading history
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk tracking
    max_unrealized_profit: Decimal = Decimal("0")
    max_unrealized_loss: Decimal = Decimal("0")
    max_adverse_excursion: Decimal = Decimal("0")  # MAE
    max_favorable_excursion: Decimal = Decimal("0")  # MFE
    
    # Performance metrics
    holding_periods: List[float] = field(default_factory=list)
    entry_efficiency: Decimal = Decimal("0")  # How close to optimal entry
    exit_efficiency: Decimal = Decimal("0")   # How close to optimal exit
    
    # Status
    closed_at: Optional[datetime] = None
    close_reason: str = ""
    
    @property
    def total_pnl(self) -> Decimal:
        """Total P&L including unrealized"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def net_pnl(self) -> Decimal:
        """Net P&L after fees"""
        return self.total_pnl - self.fees_paid
    
    @property
    def is_open(self) -> bool:
        """Check if position is still open"""
        return self.quantity != 0 and not self.closed_at
    
    def update_price(self, price: Decimal) -> None:
        """Update current price and calculate unrealized P&L"""
        self.current_price = price
        
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.average_price) * self.quantity
            
            # Update MAE and MFE
            if self.unrealized_pnl > self.max_favorable_excursion:
                self.max_favorable_excursion = self.unrealized_pnl
            
            if self.unrealized_pnl < self.max_adverse_excursion:
                self.max_adverse_excursion = self.unrealized_pnl
    
    def add_trade(self, side: str, quantity: Decimal, price: Decimal, 
                  fees: Decimal, timestamp: datetime) -> None:
        """Add a trade to the position"""
        trade = {
            'timestamp': timestamp,
            'side': side,
            'quantity': float(quantity),
            'price': float(price),
            'fees': float(fees),
            'pnl': 0.0  # Will be calculated on close
        }
        
        if side == "BUY":
            # Adding to position
            total_cost = self.quantity * self.average_price + quantity * price
            self.quantity += quantity
            self.average_price = total_cost / self.quantity if self.quantity > 0 else price
        else:  # SELL
            # Reducing position
            if quantity <= self.quantity:
                # Calculate realized P&L for this portion
                realized_pnl = (price - self.average_price) * quantity
                self.realized_pnl += realized_pnl
                trade['pnl'] = float(realized_pnl)
                
                self.quantity -= quantity
                
                # Close position if fully sold
                if self.quantity == 0:
                    self.closed_at = timestamp
                    self.close_reason = "Position fully closed"
            else:
                # Trying to sell more than we have - partial close
                realized_pnl = (price - self.average_price) * self.quantity
                self.realized_pnl += realized_pnl
                trade['pnl'] = float(realized_pnl)
                
                self.quantity = Decimal("0")
                self.closed_at = timestamp
                self.close_reason = "Oversized sell order - position closed"
        
        self.fees_paid += fees
        self.trades.append(trade)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics"""
    # Basic returns
    total_return: Decimal = Decimal("0")
    total_return_pct: Decimal = Decimal("0")
    annualized_return: Decimal = Decimal("0")
    
    # Risk metrics
    volatility: Decimal = Decimal("0")
    sharpe_ratio: Decimal = Decimal("0")
    sortino_ratio: Decimal = Decimal("0")
    calmar_ratio: Decimal = Decimal("0")
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_duration: int = 0
    
    # Advanced risk metrics
    var_95: Decimal = Decimal("0")  # Value at Risk 95%
    cvar_95: Decimal = Decimal("0")  # Conditional VaR 95%
    skewness: Decimal = Decimal("0")
    kurtosis: Decimal = Decimal("0")
    downside_deviation: Decimal = Decimal("0")
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: Decimal = Decimal("0")
    avg_win: Decimal = Decimal("0")
    avg_loss: Decimal = Decimal("0")
    profit_factor: Decimal = Decimal("0")
    
    # Efficiency metrics
    avg_trade_duration: float = 0
    avg_time_in_market: Decimal = Decimal("0")
    turnover: Decimal = Decimal("0")
    
    # Advanced trading metrics
    expectancy: Decimal = Decimal("0")
    gain_to_pain_ratio: Decimal = Decimal("0")
    recovery_factor: Decimal = Decimal("0")
    payoff_ratio: Decimal = Decimal("0")
    
    # Market comparison
    beta: Decimal = Decimal("0")
    alpha: Decimal = Decimal("0")
    correlation: Decimal = Decimal("0")
    information_ratio: Decimal = Decimal("0")
    tracking_error: Decimal = Decimal("0")
    
    # Cost analysis
    total_fees: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    market_impact_cost: Decimal = Decimal("0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_return": float(self.total_return),
            "total_return_pct": float(self.total_return_pct),
            "annualized_return": float(self.annualized_return),
            "volatility": float(self.volatility),
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "max_drawdown": float(self.max_drawdown),
            "max_drawdown_duration": self.max_drawdown_duration,
            "var_95": float(self.var_95),
            "cvar_95": float(self.cvar_95),
            "skewness": float(self.skewness),
            "kurtosis": float(self.kurtosis),
            "downside_deviation": float(self.downside_deviation),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(self.win_rate),
            "avg_win": float(self.avg_win),
            "avg_loss": float(self.avg_loss),
            "profit_factor": float(self.profit_factor),
            "avg_trade_duration": self.avg_trade_duration,
            "avg_time_in_market": float(self.avg_time_in_market),
            "turnover": float(self.turnover),
            "expectancy": float(self.expectancy),
            "gain_to_pain_ratio": float(self.gain_to_pain_ratio),
            "recovery_factor": float(self.recovery_factor),
            "payoff_ratio": float(self.payoff_ratio),
            "beta": float(self.beta),
            "alpha": float(self.alpha),
            "correlation": float(self.correlation),
            "information_ratio": float(self.information_ratio),
            "tracking_error": float(self.tracking_error),
            "total_fees": float(self.total_fees),
            "slippage_cost": float(self.slippage_cost),
            "market_impact_cost": float(self.market_impact_cost)
        }


class BacktestStrategyUseCase:
    """
    Advanced backtesting use case with comprehensive market simulation,
    realistic execution modeling, and detailed performance analysis.
    """
    
    def __init__(self, backtest_service: BacktestService):
        self.backtest_service = backtest_service
        self.logger = logging.getLogger(__name__)
    
    async def execute_backtest(self,
                              strategy_id: UUID,
                              config: BacktestConfiguration) -> BacktestResult:
        """
        Execute comprehensive backtest with advanced simulation
        
        Args:
            strategy_id: Strategy to backtest
            config: Backtest configuration
            
        Returns:
            Detailed backtest results
        """
        try:
            self.logger.info(f"Starting advanced backtest for strategy {strategy_id}")
            
            # Validate configuration
            await self._validate_configuration(config)
            
            # Get strategy
            strategy = await self.backtest_service.strategy_repository.get_by_id(strategy_id)
            if not strategy:
                raise ValidationError(f"Strategy {strategy_id} not found")
            
            # Prepare market data
            market_data = await self._prepare_market_data(strategy, config)
            
            # Validate data quality
            await self._validate_data_quality(market_data, config)
            
            # Initialize backtest state
            backtest_state = BacktestState(
                strategy=strategy,
                config=config,
                market_data=market_data
            )
            
            # Run simulation
            await self._run_simulation(backtest_state)
            
            # Calculate comprehensive metrics
            metrics = await self._calculate_metrics(backtest_state)
            
            # Create result
            result = BacktestResult(
                strategy_id=strategy_id,
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                final_capital=backtest_state.current_capital,
                total_return=backtest_state.current_capital - config.initial_capital,
                total_return_pct=(backtest_state.current_capital - config.initial_capital) / config.initial_capital * 100,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                total_trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                avg_trade_return=metrics.expectancy,
                volatility=metrics.volatility,
                trades=backtest_state.all_trades,
                equity_curve=backtest_state.equity_curve,
                performance_metrics=metrics.to_dict()
            )
            
            self.logger.info(f"Backtest completed: {result.total_return_pct:.2f}% return, "
                           f"{result.sharpe_ratio:.2f} Sharpe ratio")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise BacktestError(f"Backtest execution failed: {e}")
    
    async def _validate_configuration(self, config: BacktestConfiguration) -> None:
        """Validate backtest configuration"""
        if config.start_date >= config.end_date:
            raise ValidationError("Start date must be before end date")
        
        if config.initial_capital <= 0:
            raise ValidationError("Initial capital must be positive")
        
        duration = config.end_date - config.start_date
        if duration.days < 1:
            raise ValidationError("Backtest period must be at least 1 day")
        
        if duration.days > 3650:  # 10 years
            raise ValidationError("Backtest period cannot exceed 10 years")
        
        # Validate rates
        if config.commission_rate < 0 or config.commission_rate > Decimal("0.1"):
            raise ValidationError("Commission rate must be between 0% and 10%")
        
        if config.slippage_rate < 0 or config.slippage_rate > Decimal("0.05"):
            raise ValidationError("Slippage rate must be between 0% and 5%")
    
    async def _prepare_market_data(self, 
                                  strategy: Strategy, 
                                  config: BacktestConfiguration) -> Dict[str, List[MarketDataPoint]]:
        """Prepare and enhance market data for backtesting"""
        symbols = list(strategy.symbols)
        raw_data = await self._fetch_historical_data(symbols, config.start_date, config.end_date)
        
        enhanced_data = {}
        for symbol, data_points in raw_data.items():
            enhanced_points = []
            
            for i, point in enumerate(data_points):
                enhanced_point = MarketDataPoint(
                    timestamp=point['timestamp'],
                    symbol=symbol,
                    open=Decimal(str(point['open'])),
                    high=Decimal(str(point['high'])),
                    low=Decimal(str(point['low'])),
                    close=Decimal(str(point['close'])),
                    volume=Decimal(str(point['volume']))
                )
                
                # Add market microstructure simulation
                enhanced_point.bid = enhanced_point.close * Decimal("0.9995")  # 0.05% spread
                enhanced_point.ask = enhanced_point.close * Decimal("1.0005")
                enhanced_point.spread = enhanced_point.ask - enhanced_point.bid
                
                # Calculate technical indicators
                if i >= 20:  # Need 20 periods for SMA
                    recent_closes = [Decimal(str(dp['close'])) for dp in data_points[i-19:i+1]]
                    enhanced_point.sma_20 = sum(recent_closes) / 20
                
                if i >= 50:  # Need 50 periods for SMA
                    recent_closes = [Decimal(str(dp['close'])) for dp in data_points[i-49:i+1]]
                    enhanced_point.sma_50 = sum(recent_closes) / 50
                
                # Calculate RSI (simplified)
                if i >= 14:
                    recent_closes = [Decimal(str(dp['close'])) for dp in data_points[i-13:i+1]]
                    enhanced_point.rsi = self._calculate_rsi(recent_closes)
                
                # Add volatility estimate
                if i >= 20:
                    recent_returns = []
                    for j in range(i-19, i):
                        prev_close = Decimal(str(data_points[j]['close']))
                        curr_close = Decimal(str(data_points[j+1]['close']))
                        if prev_close > 0:
                            returns = (curr_close - prev_close) / prev_close
                            recent_returns.append(float(returns))
                    
                    if recent_returns:
                        enhanced_point.volatility = Decimal(str(statistics.stdev(recent_returns) * (252 ** 0.5)))
                
                enhanced_points.append(enhanced_point)
            
            enhanced_data[symbol] = enhanced_points
        
        return enhanced_data
    
    def _calculate_rsi(self, closes: List[Decimal], period: int = 14) -> Decimal:
        """Calculate RSI indicator"""
        if len(closes) < period + 1:
            return Decimal("50")  # Neutral RSI
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(Decimal("0"))
            else:
                gains.append(Decimal("0"))
                losses.append(-change)
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return Decimal("100")
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    async def _fetch_historical_data(self, 
                                   symbols: List[str], 
                                   start_date: datetime, 
                                   end_date: datetime) -> Dict[str, List[Dict]]:
        """Fetch historical market data"""
        # In production, this would fetch from real data providers
        # For demo, generate realistic synthetic data
        
        historical_data = {}
        
        for symbol in symbols:
            data = []
            current_date = start_date
            base_price = 50000 + (hash(symbol) % 10000)  # Starting price
            
            while current_date <= end_date:
                # Generate realistic price movement
                volatility = 0.02  # 2% daily volatility
                drift = 0.0001     # Slight upward drift
                
                # Random walk with volatility clustering
                random_factor = (hash(str(current_date) + symbol) % 2000 - 1000) / 1000
                price_change = drift + volatility * random_factor
                
                new_price = base_price * (1 + price_change)
                
                # Generate OHLC from close price
                high_factor = abs(hash(str(current_date) + symbol + "high") % 100) / 10000
                low_factor = abs(hash(str(current_date) + symbol + "low") % 100) / 10000
                
                high = new_price * (1 + high_factor)
                low = new_price * (1 - low_factor)
                open_price = base_price
                
                # Generate volume
                base_volume = 1000000
                volume_factor = (hash(str(current_date) + symbol + "vol") % 500 + 500) / 1000
                volume = base_volume * volume_factor
                
                candle = {
                    'timestamp': current_date,
                    'open': float(open_price),
                    'high': float(high),
                    'low': float(low),
                    'close': float(new_price),
                    'volume': float(volume)
                }
                data.append(candle)
                
                base_price = new_price
                current_date += timedelta(hours=1)  # Hourly data
            
            historical_data[symbol] = data
        
        return historical_data
    
    async def _validate_data_quality(self, 
                                   market_data: Dict[str, List[MarketDataPoint]], 
                                   config: BacktestConfiguration) -> None:
        """Validate market data quality"""
        for symbol, data_points in market_data.items():
            if not data_points:
                raise DataError(f"No data available for symbol {symbol}")
            
            # Check data coverage
            expected_points = int((config.end_date - config.start_date).total_seconds() / 3600)
            actual_points = len(data_points)
            coverage = actual_points / expected_points
            
            if coverage < config.min_data_quality:
                if config.handle_missing_data == "error":
                    raise DataError(f"Data quality for {symbol} is {coverage:.2%}, "
                                  f"below required {config.min_data_quality:.2%}")
                else:
                    self.logger.warning(f"Data quality for {symbol} is {coverage:.2%}")
            
            # Check for obvious data errors
            for point in data_points:
                if point.high < point.low or point.high < point.close or point.low > point.close:
                    raise DataError(f"Invalid OHLC data for {symbol} at {point.timestamp}")
                
                if point.volume < 0:
                    raise DataError(f"Negative volume for {symbol} at {point.timestamp}")
    
    async def _run_simulation(self, state: BacktestState) -> None:
        """Run the main backtest simulation"""
        try:
            # Initialize strategy
            if state.strategy.implementation:
                market_data_dict = {}
                for symbol, points in state.market_data.items():
                    if points:
                        market_data_dict[symbol] = points[0].to_dict()
                
                await state.strategy.implementation.initialize(market_data_dict)
            
            # Get all timestamps across all symbols
            all_timestamps = set()
            for symbol_data in state.market_data.values():
                for point in symbol_data:
                    all_timestamps.add(point.timestamp)
            
            sorted_timestamps = sorted(all_timestamps)
            total_timestamps = len(sorted_timestamps)
            
            # Warm-up period
            warm_up_end = min(state.config.warm_up_period, total_timestamps // 4)
            
            # Process each timestamp
            for i, timestamp in enumerate(sorted_timestamps):
                # Update market data for current timestamp
                current_market_data = {}
                for symbol, symbol_data in state.market_data.items():
                    for point in symbol_data:
                        if point.timestamp == timestamp:
                            current_market_data[symbol] = point.to_dict()
                            break
                
                if not current_market_data:
                    continue
                
                # Update position prices
                await self._update_position_prices(state, current_market_data, timestamp)
                
                # Record equity curve
                total_equity = state.current_capital + self._calculate_unrealized_pnl(state)
                state.equity_curve.append({
                    'timestamp': timestamp.isoformat(),
                    'equity': float(total_equity),
                    'capital': float(state.current_capital),
                    'unrealized_pnl': float(self._calculate_unrealized_pnl(state)),
                    'num_positions': len([p for p in state.positions.values() if p.is_open])
                })
                
                # Skip signal generation during warm-up
                if i < warm_up_end:
                    continue
                
                # Generate and process signals
                if state.strategy.implementation:
                    try:
                        await state.strategy.implementation.update_state(current_market_data)
                        signals = await state.strategy.implementation.generate_signals(current_market_data)
                        
                        for signal in signals:
                            await self._process_signal(state, signal, current_market_data, timestamp)
                            
                    except Exception as e:
                        self.logger.warning(f"Error generating signals at {timestamp}: {e}")
                
                # Check risk limits
                await self._check_risk_limits(state, timestamp)
                
                # Progress logging
                if i % 1000 == 0:
                    progress = (i / total_timestamps) * 100
                    self.logger.info(f"Backtest progress: {progress:.1f}%")
            
            # Close all remaining positions
            await self._close_all_positions(state, sorted_timestamps[-1])
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
            raise BacktestError(f"Simulation failed: {e}")
    
    async def _process_signal(self, 
                            state: BacktestState, 
                            signal: Signal, 
                            market_data: Dict[str, Any], 
                            timestamp: datetime) -> None:
        """Process a trading signal with realistic execution"""
        try:
            # Validate signal
            if not self._validate_signal(state, signal, market_data):
                return
            
            # Calculate execution price with slippage
            market_price = Decimal(str(market_data[signal.symbol]['close']))
            execution_price = self._calculate_execution_price(
                signal, market_price, state.config
            )
            
            # Calculate position size
            position_size = self._calculate_position_size(state, signal, execution_price)
            
            if position_size <= 0:
                return
            
            # Calculate fees
            notional_value = position_size * execution_price
            fees = notional_value * state.config.commission_rate
            
            # Execute trade
            await self._execute_trade(
                state, signal.symbol, signal.direction.value, 
                position_size, execution_price, fees, timestamp
            )
            
        except Exception as e:
            self.logger.warning(f"Error processing signal: {e}")
    
    def _calculate_execution_price(self, 
                                 signal: Signal, 
                                 market_price: Decimal, 
                                 config: BacktestConfiguration) -> Decimal:
        """Calculate realistic execution price including slippage"""
        if not config.realistic_execution:
            return market_price
        
        # Apply slippage based on direction
        if signal.direction == SignalDirection.BUY:
            slippage_cost = market_price * config.slippage_rate
            execution_price = market_price + slippage_cost
        else:
            slippage_cost = market_price * config.slippage_rate
            execution_price = market_price - slippage_cost
        
        # Add spread cost
        spread_cost = market_price * config.spread_cost
        if signal.direction == SignalDirection.BUY:
            execution_price += spread_cost
        else:
            execution_price -= spread_cost
        
        return execution_price
    
    def _calculate_position_size(self, 
                               state: BacktestState, 
                               signal: Signal, 
                               price: Decimal) -> Decimal:
        """Calculate appropriate position size based on available capital and risk"""
        # Basic position sizing based on available capital
        available_capital = state.current_capital * Decimal("0.95")  # Keep 5% buffer
        max_position_value = available_capital * Decimal("0.2")  # Max 20% per position
        
        # Strategy-defined position size
        if hasattr(state.strategy.implementation, 'calculate_position_size'):
            try:
                suggested_size = state.strategy.implementation.calculate_position_size(
                    signal.symbol, price, signal.metrics.confidence, available_capital
                )
                max_affordable = max_position_value / price
                return min(Decimal(str(suggested_size)), max_affordable)
            except:
                pass
        
        # Default position sizing
        return min(signal.quantity, max_position_value / price)
    
    async def _execute_trade(self, 
                           state: BacktestState, 
                           symbol: str, 
                           side: str, 
                           quantity: Decimal, 
                           price: Decimal, 
                           fees: Decimal, 
                           timestamp: datetime) -> None:
        """Execute a trade and update positions"""
        position_key = f"{symbol}_{state.strategy.id}"
        
        # Get or create position
        if position_key not in state.positions:
            state.positions[position_key] = BacktestPosition(
                symbol=symbol,
                strategy_id=state.strategy.id,
                opened_at=timestamp
            )
        
        position = state.positions[position_key]
        
        # Execute trade
        position.add_trade(side, quantity, price, fees, timestamp)
        
        # Update capital
        if side == "BUY":
            cost = quantity * price + fees
            state.current_capital -= cost
        else:  # SELL
            proceeds = quantity * price - fees
            state.current_capital += proceeds
        
        # Record trade
        trade_record = {
            'timestamp': timestamp.isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': float(quantity),
            'price': float(price),
            'fees': float(fees),
            'capital_after': float(state.current_capital)
        }
        state.all_trades.append(trade_record)
    
    async def _update_position_prices(self, 
                                    state: BacktestState, 
                                    market_data: Dict[str, Any], 
                                    timestamp: datetime) -> None:
        """Update all position prices with current market data"""
        for position in state.positions.values():
            if position.is_open and position.symbol in market_data:
                current_price = Decimal(str(market_data[position.symbol]['close']))
                position.update_price(current_price)
    
    def _calculate_unrealized_pnl(self, state: BacktestState) -> Decimal:
        """Calculate total unrealized P&L across all positions"""
        return sum(position.unrealized_pnl for position in state.positions.values() if position.is_open)
    
    async def _check_risk_limits(self, state: BacktestState, timestamp: datetime) -> None:
        """Check and enforce risk limits"""
        total_equity = state.current_capital + self._calculate_unrealized_pnl(state)
        
        # Check maximum drawdown
        if state.config.max_drawdown_stop:
            if state.peak_equity > 0:
                current_drawdown = (state.peak_equity - total_equity) / state.peak_equity
                if current_drawdown >= state.config.max_drawdown_stop:
                    await self._close_all_positions(state, timestamp)
                    state.stopped_out = True
                    state.stop_reason = f"Maximum drawdown limit reached: {current_drawdown:.2%}"
        
        # Update peak equity
        if total_equity > state.peak_equity:
            state.peak_equity = total_equity
        
        # Check daily loss limit
        if state.config.daily_loss_limit:
            daily_pnl = total_equity - state.config.initial_capital
            if daily_pnl <= -state.config.daily_loss_limit:
                await self._close_all_positions(state, timestamp)
                state.stopped_out = True
                state.stop_reason = f"Daily loss limit reached: {daily_pnl}"
    
    async def _close_all_positions(self, state: BacktestState, timestamp: datetime) -> None:
        """Close all open positions"""
        for position in state.positions.values():
            if position.is_open:
                # Realize the unrealized P&L
                state.current_capital += position.unrealized_pnl
                position.realized_pnl += position.unrealized_pnl
                position.unrealized_pnl = Decimal("0")
                position.quantity = Decimal("0")
                position.closed_at = timestamp
                position.close_reason = "Backtest end" if not state.stopped_out else state.stop_reason
    
    def _validate_signal(self, 
                        state: BacktestState, 
                        signal: Signal, 
                        market_data: Dict[str, Any]) -> bool:
        """Validate if signal can be executed"""
        # Check if we have market data for the symbol
        if signal.symbol not in market_data:
            return False
        
        # Check capital requirements for buy signals
        if signal.direction == SignalDirection.BUY:
            required_capital = signal.quantity * signal.current_price * Decimal("1.01")  # Include fees
            return state.current_capital >= required_capital
        
        # Check position size for sell signals
        if signal.direction == SignalDirection.SELL:
            position_key = f"{signal.symbol}_{state.strategy.id}"
            if position_key in state.positions:
                position = state.positions[position_key]
                return position.is_open and position.quantity >= signal.quantity
            return False
        
        return True
    
    async def _calculate_metrics(self, state: BacktestState) -> BacktestMetrics:
        """Calculate comprehensive performance metrics"""
        metrics = BacktestMetrics()
        
        if not state.equity_curve:
            return metrics
        
        # Basic return calculations
        initial_capital = state.config.initial_capital
        final_capital = state.current_capital
        
        metrics.total_return = final_capital - initial_capital
        metrics.total_return_pct = (metrics.total_return / initial_capital) * 100
        
        # Time-based calculations
        total_days = (state.config.end_date - state.config.start_date).days
        years = total_days / 365.25
        
        if years > 0:
            metrics.annualized_return = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        
        # Calculate returns series for risk metrics
        equity_values = [point['equity'] for point in state.equity_curve]
        returns = []
        
        for i in range(1, len(equity_values)):
            if equity_values[i-1] > 0:
                daily_return = (equity_values[i] - equity_values[i-1]) / equity_values[i-1]
                returns.append(daily_return)
        
        if returns:
            # Volatility and Sharpe ratio
            metrics.volatility = Decimal(str(statistics.stdev(returns) * (252 ** 0.5)))  # Annualized
            
            if metrics.volatility > 0:
                excess_return = metrics.annualized_return / 100 - state.config.risk_free_rate
                metrics.sharpe_ratio = excess_return / metrics.volatility
            
            # Downside metrics
            negative_returns = [r for r in returns if r < 0]
            if negative_returns:
                metrics.downside_deviation = Decimal(str(statistics.stdev(negative_returns) * (252 ** 0.5)))
                
                if metrics.downside_deviation > 0:
                    target_return = state.config.risk_free_rate / 252  # Daily risk-free rate
                    excess_returns = [r - target_return for r in returns]
                    avg_excess_return = sum(excess_returns) / len(excess_returns) * 252
                    metrics.sortino_ratio = Decimal(str(avg_excess_return)) / metrics.downside_deviation
            
            # VaR calculations
            if len(returns) >= 20:  # Need sufficient data
                sorted_returns = sorted(returns)
                var_index = int(len(sorted_returns) * 0.05)  # 5% VaR
                metrics.var_95 = Decimal(str(abs(sorted_returns[var_index])))
                
                # CVaR (average of worst 5% returns)
                worst_returns = sorted_returns[:var_index+1]
                if worst_returns:
                    metrics.cvar_95 = Decimal(str(abs(sum(worst_returns) / len(worst_returns))))
                
                # Maximum drawdown calculation
        peak = initial_capital
        max_dd = Decimal("0")
        dd_duration = 0
        max_dd_duration = 0
        
        for point in state.equity_curve:
            equity = Decimal(str(point['equity']))
            
            if equity > peak:
                peak = equity
                dd_duration = 0
            else:
                drawdown = (peak - equity) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
                dd_duration += 1
                if dd_duration > max_dd_duration:
                    max_dd_duration = dd_duration
        
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_duration = max_dd_duration
        
        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / (metrics.max_drawdown * 100)
        
        # Trading statistics
        all_trades = []
        for position in state.positions.values():
            all_trades.extend(position.trades)
        
        metrics.total_trades = len(all_trades)
        
        if all_trades:
            winning_trades = [t for t in all_trades if t['pnl'] > 0]
            losing_trades = [t for t in all_trades if t['pnl'] < 0]
            
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            metrics.win_rate = Decimal(str(len(winning_trades) / len(all_trades) * 100))
            
            if winning_trades:
                metrics.avg_win = Decimal(str(sum(t['pnl'] for t in winning_trades) / len(winning_trades)))
            
            if losing_trades:
                metrics.avg_loss = Decimal(str(sum(t['pnl'] for t in losing_trades) / len(losing_trades)))
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            
            if gross_loss > 0:
                metrics.profit_factor = Decimal(str(gross_profit / gross_loss))
            
            # Expectancy
            win_rate_decimal = metrics.win_rate / 100
            loss_rate = Decimal("1") - win_rate_decimal
            
            if metrics.avg_loss != 0:
                metrics.expectancy = (win_rate_decimal * metrics.avg_win) + (loss_rate * metrics.avg_loss)
        
        # Calculate total fees and costs
        metrics.total_fees = sum(Decimal(str(t['fees'])) for t in all_trades)
        
        return metrics


@dataclass
class BacktestState:
    """Internal state for advanced backtesting"""
    strategy: Strategy
    config: BacktestConfiguration
    market_data: Dict[str, List[MarketDataPoint]]
    
    # Financial state
    current_capital: Decimal = None
    peak_equity: Decimal = None
    
    # Positions and trades
    positions: Dict[str, BacktestPosition] = field(default_factory=dict)
    all_trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # Risk management
    stopped_out: bool = False
    stop_reason: str = ""
    
    def __post_init__(self):
        if self.current_capital is None:
            self.current_capital = self.config.initial_capital
        if self.peak_equity is None:
            self.peak_equity = self.config.initial_capital


class ParameterOptimizer:
    """Advanced parameter optimization for strategies"""
    
    def __init__(self, backtest_use_case: BacktestStrategyUseCase):
        self.backtest_use_case = backtest_use_case
        self.logger = logging.getLogger(__name__)
    
    async def optimize_parameters(self,
                                strategy_id: UUID,
                                config: BacktestConfiguration,
                                parameter_ranges: Dict[str, List[Any]],
                                optimization_metric: str = "sharpe_ratio",
                                max_iterations: int = 100,
                                optimization_method: str = "grid_search") -> Dict[str, Any]:
        """
        Optimize strategy parameters using various methods
        
        Args:
            strategy_id: Strategy to optimize
            config: Base backtest configuration
            parameter_ranges: Dict of parameter names to ranges
            optimization_metric: Metric to optimize
            max_iterations: Maximum iterations
            optimization_method: grid_search, random_search, bayesian
            
        Returns:
            Optimization results
        """
        try:
            if optimization_method == "grid_search":
                return await self._grid_search_optimization(
                    strategy_id, config, parameter_ranges, optimization_metric, max_iterations
                )
            elif optimization_method == "random_search":
                return await self._random_search_optimization(
                    strategy_id, config, parameter_ranges, optimization_metric, max_iterations
                )
            elif optimization_method == "bayesian":
                return await self._bayesian_optimization(
                    strategy_id, config, parameter_ranges, optimization_metric, max_iterations
                )
            else:
                raise ValueError(f"Unknown optimization method: {optimization_method}")
                
        except Exception as e:
            self.logger.error(f"Parameter optimization failed: {e}")
            raise
    
    async def _grid_search_optimization(self,
                                      strategy_id: UUID,
                                      config: BacktestConfiguration,
                                      parameter_ranges: Dict[str, List[Any]],
                                      optimization_metric: str,
                                      max_iterations: int) -> Dict[str, Any]:
        """Grid search parameter optimization"""
        import itertools
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*param_values))
        
        if len(all_combinations) > max_iterations:
            # Sample combinations if too many
            import random
            combinations_to_test = random.sample(all_combinations, max_iterations)
        else:
            combinations_to_test = all_combinations
        
        results = []
        best_result = None
        best_metric_value = float('-inf')
        
        total_combinations = len(combinations_to_test)
        
        # Test each combination
        for i, param_combo in enumerate(combinations_to_test):
            try:
                # Create parameter override
                param_override = dict(zip(param_names, param_combo))
                
                # Update strategy parameters
                strategy = await self.backtest_use_case.backtest_service.strategy_repository.get_by_id(strategy_id)
                original_params = {}
                
                for param_name, param_value in param_override.items():
                    if hasattr(strategy.parameters, param_name):
                        original_params[param_name] = getattr(strategy.parameters, param_name)
                        setattr(strategy.parameters, param_name, param_value)
                
                # Run backtest
                result = await self.backtest_use_case.execute_backtest(strategy_id, config)
                
                # Extract metric value
                if hasattr(result, optimization_metric):
                    metric_value = float(getattr(result, optimization_metric))
                else:
                    metric_value = float(result.performance_metrics.get(optimization_metric, 0))
                
                result_entry = {
                    "iteration": i + 1,
                    "parameters": param_override,
                    "metric_value": metric_value,
                    "total_return_pct": float(result.total_return_pct),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "max_drawdown": float(result.max_drawdown),
                    "win_rate": float(result.win_rate),
                    "total_trades": result.total_trades
                }
                
                results.append(result_entry)
                
                # Track best result
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = result_entry
                
                # Restore original parameters
                for param_name, original_value in original_params.items():
                    setattr(strategy.parameters, param_name, original_value)
                
                # Progress logging
                if i % 10 == 0:
                    progress = (i / total_combinations) * 100
                    self.logger.info(f"Optimization progress: {progress:.1f}%")
                
            except Exception as e:
                self.logger.warning(f"Optimization iteration {i+1} failed: {e}")
                continue
        
        # Sort results by metric
        results.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return {
            "method": "grid_search",
            "strategy_id": str(strategy_id),
            "optimization_metric": optimization_metric,
            "parameter_ranges": parameter_ranges,
            "total_iterations": len(results),
            "best_parameters": best_result["parameters"] if best_result else None,
            "best_metric_value": best_metric_value if best_result else None,
            "best_result": best_result,
            "top_10_results": results[:10],
            "all_results": results,
            "summary": {
                "best_return": max(r["total_return_pct"] for r in results) if results else 0,
                "worst_return": min(r["total_return_pct"] for r in results) if results else 0,
                "avg_return": sum(r["total_return_pct"] for r in results) / len(results) if results else 0,
                "success_rate": len(results) / len(combinations_to_test) * 100
            }
        }
    
    async def _random_search_optimization(self,
                                        strategy_id: UUID,
                                        config: BacktestConfiguration,
                                        parameter_ranges: Dict[str, List[Any]],
                                        optimization_metric: str,
                                        max_iterations: int) -> Dict[str, Any]:
        """Random search parameter optimization"""
        import random
        
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        results = []
        best_result = None
        best_metric_value = float('-inf')
        
        # Random sampling
        for i in range(max_iterations):
            try:
                # Generate random combination
                param_combo = []
                for param_vals in param_values:
                    param_combo.append(random.choice(param_vals))
                
                param_override = dict(zip(param_names, param_combo))
                
                # Run backtest with these parameters
                strategy = await self.backtest_use_case.backtest_service.strategy_repository.get_by_id(strategy_id)
                original_params = {}
                
                for param_name, param_value in param_override.items():
                    if hasattr(strategy.parameters, param_name):
                        original_params[param_name] = getattr(strategy.parameters, param_name)
                        setattr(strategy.parameters, param_name, param_value)
                
                result = await self.backtest_use_case.execute_backtest(strategy_id, config)
                
                # Extract metric
                if hasattr(result, optimization_metric):
                    metric_value = float(getattr(result, optimization_metric))
                else:
                    metric_value = float(result.performance_metrics.get(optimization_metric, 0))
                
                result_entry = {
                    "iteration": i + 1,
                    "parameters": param_override,
                    "metric_value": metric_value,
                    "total_return_pct": float(result.total_return_pct),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "max_drawdown": float(result.max_drawdown),
                    "win_rate": float(result.win_rate),
                    "total_trades": result.total_trades
                }
                
                results.append(result_entry)
                
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_result = result_entry
                
                # Restore parameters
                for param_name, original_value in original_params.items():
                    setattr(strategy.parameters, param_name, original_value)
                
                if i % 10 == 0:
                    self.logger.info(f"Random search progress: {i}/{max_iterations}")
                
            except Exception as e:
                self.logger.warning(f"Random search iteration {i+1} failed: {e}")
                continue
        
        results.sort(key=lambda x: x["metric_value"], reverse=True)
        
        return {
            "method": "random_search",
            "strategy_id": str(strategy_id),
            "optimization_metric": optimization_metric,
            "parameter_ranges": parameter_ranges,
            "total_iterations": len(results),
            "best_parameters": best_result["parameters"] if best_result else None,
            "best_metric_value": best_metric_value if best_result else None,
            "best_result": best_result,
            "top_10_results": results[:10],
            "all_results": results
        }
    
    async def _bayesian_optimization(self,
                                   strategy_id: UUID,
                                   config: BacktestConfiguration,
                                   parameter_ranges: Dict[str, List[Any]],
                                   optimization_metric: str,
                                   max_iterations: int) -> Dict[str, Any]:
        """Bayesian optimization (simplified implementation)"""
        # This is a simplified version - in production you'd use libraries like scikit-optimize
        self.logger.info("Bayesian optimization not fully implemented, falling back to random search")
        return await self._random_search_optimization(
            strategy_id, config, parameter_ranges, optimization_metric, max_iterations
        )


class WalkForwardAnalysis:
    """Walk-forward analysis for strategy validation"""
    
    def __init__(self, backtest_use_case: BacktestStrategyUseCase):
        self.backtest_use_case = backtest_use_case
        self.logger = logging.getLogger(__name__)
    
    async def run_walk_forward_analysis(self,
                                      strategy_id: UUID,
                                      start_date: datetime,
                                      end_date: datetime,
                                      optimization_window: int = 252,  # 1 year
                                      test_window: int = 63,  # 3 months
                                      parameter_ranges: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Run walk-forward analysis to validate strategy robustness
        
        Args:
            strategy_id: Strategy to analyze
            start_date: Analysis start date
            end_date: Analysis end date
            optimization_window: Days for optimization period
            test_window: Days for out-of-sample testing
            parameter_ranges: Parameters to optimize
            
        Returns:
            Walk-forward analysis results
        """
        try:
            self.logger.info(f"Starting walk-forward analysis for strategy {strategy_id}")
            
            results = []
            current_date = start_date
            
            while current_date + timedelta(days=optimization_window + test_window) <= end_date:
                # Define optimization and test periods
                opt_start = current_date
                opt_end = current_date + timedelta(days=optimization_window)
                test_start = opt_end
                test_end = opt_end + timedelta(days=test_window)
                
                self.logger.info(f"Walk-forward period: opt {opt_start.date()} to {opt_end.date()}, "
                               f"test {test_start.date()} to {test_end.date()}")
                
                try:
                    # Optimization phase
                    opt_config = BacktestConfiguration(
                        start_date=opt_start,
                        end_date=opt_end,
                        initial_capital=Decimal("100000")
                    )
                    
                    if parameter_ranges:
                        optimizer = ParameterOptimizer(self.backtest_use_case)
                        opt_result = await optimizer.optimize_parameters(
                            strategy_id, opt_config, parameter_ranges, 
                            max_iterations=50
                        )
                        best_params = opt_result.get("best_parameters", {})
                    else:
                        best_params = {}
                    
                    # Out-of-sample testing
                    test_config = BacktestConfiguration(
                        start_date=test_start,
                        end_date=test_end,
                        initial_capital=Decimal("100000")
                    )
                    
                    # Apply optimized parameters
                    strategy = await self.backtest_use_case.backtest_service.strategy_repository.get_by_id(strategy_id)
                    original_params = {}
                    
                    for param_name, param_value in best_params.items():
                        if hasattr(strategy.parameters, param_name):
                            original_params[param_name] = getattr(strategy.parameters, param_name)
                            setattr(strategy.parameters, param_name, param_value)
                    
                    # Run out-of-sample test
                    test_result = await self.backtest_use_case.execute_backtest(strategy_id, test_config)
                    
                    # Restore original parameters
                    for param_name, original_value in original_params.items():
                        setattr(strategy.parameters, param_name, original_value)
                    
                    period_result = {
                        "optimization_period": {
                            "start": opt_start.isoformat(),
                            "end": opt_end.isoformat()
                        },
                        "test_period": {
                            "start": test_start.isoformat(),
                            "end": test_end.isoformat()
                        },
                        "optimized_parameters": best_params,
                        "out_of_sample_results": {
                            "total_return_pct": float(test_result.total_return_pct),
                            "sharpe_ratio": float(test_result.sharpe_ratio),
                            "max_drawdown": float(test_result.max_drawdown),
                            "win_rate": float(test_result.win_rate),
                            "total_trades": test_result.total_trades
                        }
                    }
                    
                    results.append(period_result)
                    
                except Exception as e:
                    self.logger.warning(f"Walk-forward period failed: {e}")
                    continue
                
                # Move to next period
                current_date += timedelta(days=test_window)
            
            # Calculate aggregate statistics
            if results:
                oos_returns = [r["out_of_sample_results"]["total_return_pct"] for r in results]
                oos_sharpe = [r["out_of_sample_results"]["sharpe_ratio"] for r in results]
                
                aggregate_stats = {
                    "total_periods": len(results),
                    "avg_oos_return": sum(oos_returns) / len(oos_returns),
                    "std_oos_return": statistics.stdev(oos_returns) if len(oos_returns) > 1 else 0,
                    "avg_oos_sharpe": sum(oos_sharpe) / len(oos_sharpe),
                    "positive_periods": len([r for r in oos_returns if r > 0]),
                    "negative_periods": len([r for r in oos_returns if r <= 0]),
                    "consistency_ratio": len([r for r in oos_returns if r > 0]) / len(oos_returns) * 100
                }
            else:
                aggregate_stats = {}
            
            return {
                "strategy_id": str(strategy_id),
                "analysis_period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "configuration": {
                    "optimization_window": optimization_window,
                    "test_window": test_window,
                    "parameter_ranges": parameter_ranges
                },
                "results": results,
                "aggregate_statistics": aggregate_stats
            }
            
        except Exception as e:
            self.logger.error(f"Walk-forward analysis failed: {e}")
            raise BacktestError(f"Walk-forward analysis failed: {e}")