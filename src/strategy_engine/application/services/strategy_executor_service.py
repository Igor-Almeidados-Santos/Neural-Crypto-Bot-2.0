"""
Application Services - Strategy Engine Application Layer

This module implements application services for strategy execution, 
backtesting, and signal processing with comprehensive orchestration.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import asyncio
import logging
from dataclasses import dataclass

# Import domain entities and services
from ..domain.entities.strategy import Strategy, StrategyStatus, BaseStrategy
from ..domain.entities.position import Position, PositionStatus, Trade
from ..domain.entities.signal import Signal, SignalStatus, SignalDirection
from ..domain.events.signal_generated_event import SignalGeneratedEvent
from ..domain.events.position_changed_event import PositionChangedEvent
from ..infrastructure.strategy_repository import (
    IStrategyRepository, IPositionRepository, ISignalRepository
)
from ..infrastructure.signal_publisher import ISignalPublisher


class ApplicationServiceError(Exception):
    """Base exception for application service errors"""
    pass


class StrategyNotFoundError(ApplicationServiceError):
    """Raised when strategy is not found"""
    pass


class InvalidStrategyStateError(ApplicationServiceError):
    """Raised when strategy is in invalid state for operation"""
    pass


class RiskLimitExceededError(ApplicationServiceError):
    """Raised when risk limits are exceeded"""
    pass


@dataclass
class BacktestResult:
    """Results from backtesting a strategy"""
    strategy_id: UUID
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: Decimal
    total_return_pct: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_return: Decimal
    volatility: Decimal
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


@dataclass
class ExecutionResult:
    """Results from strategy execution"""
    strategy_id: UUID
    signals_generated: int
    signals_executed: int
    positions_opened: int
    positions_closed: int
    total_pnl: Decimal
    execution_time: float
    errors: List[str]
    warnings: List[str]


class StrategyExecutorService:
    """
    Service for executing trading strategies in real-time or simulation mode.
    Handles signal generation, position management, and risk control.
    """
    
    def __init__(self,
                 strategy_repository: IStrategyRepository,
                 position_repository: IPositionRepository,
                 signal_repository: ISignalRepository,
                 signal_publisher: ISignalPublisher):
        self.strategy_repository = strategy_repository
        self.position_repository = position_repository
        self.signal_repository = signal_repository
        self.signal_publisher = signal_publisher
        self.logger = logging.getLogger(__name__)
        
        # Active strategy execution tasks
        self.execution_tasks: Dict[UUID, asyncio.Task] = {}
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Risk management
        self.max_position_value = Decimal("100000")
        self.max_daily_loss = Decimal("5000")
        self.max_drawdown = Decimal("0.10")  # 10%
    
    async def start_strategy(self, strategy_id: UUID, market_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Start executing a strategy
        
        Args:
            strategy_id: ID of strategy to start
            market_data: Initial market data
            
        Returns:
            True if started successfully
        """
        try:
            # Get strategy from repository
            strategy = await self.strategy_repository.get_by_id(strategy_id)
            if not strategy:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
            
            # Check if strategy is already running
            if strategy_id in self.execution_tasks:
                self.logger.warning(f"Strategy {strategy_id} is already running")
                return True
            
            # Start strategy
            success = await strategy.start(market_data)
            if not success:
                self.logger.error(f"Failed to start strategy {strategy_id}")
                return False
            
            # Save updated strategy state
            await self.strategy_repository.save(strategy)
            
            # Start execution task
            task = asyncio.create_task(self._execute_strategy(strategy))
            self.execution_tasks[strategy_id] = task
            
            self.logger.info(f"Started strategy {strategy.name} ({strategy_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting strategy {strategy_id}: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: UUID, reason: str = "") -> bool:
        """
        Stop executing a strategy
        
        Args:
            strategy_id: ID of strategy to stop
            reason: Reason for stopping
            
        Returns:
            True if stopped successfully
        """
        try:
            # Cancel execution task
            if strategy_id in self.execution_tasks:
                task = self.execution_tasks.pop(strategy_id)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Get and stop strategy
            strategy = await self.strategy_repository.get_by_id(strategy_id)
            if strategy:
                await strategy.stop(reason)
                await self.strategy_repository.save(strategy)
            
            self.logger.info(f"Stopped strategy {strategy_id}: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping strategy {strategy_id}: {e}")
            return False
    
    async def _execute_strategy(self, strategy: Strategy) -> None:
        """
        Main execution loop for a strategy
        
        Args:
            strategy: Strategy to execute
        """
        try:
            while strategy.is_active:
                try:
                    # Get current market data
                    market_data = await self._get_market_data(strategy.symbols)
                    
                    # Update strategy state
                    if strategy.implementation:
                        await strategy.implementation.update_state(market_data)
                    
                    # Generate signals
                    signals = await self._generate_signals(strategy, market_data)
                    
                    # Process and validate signals
                    for signal in signals:
                        await self._process_signal(strategy, signal, market_data)
                    
                    # Update positions with current market prices
                    await self._update_positions(strategy, market_data)
                    
                    # Check risk limits
                    await self._check_risk_limits(strategy)
                    
                    # Sleep based on strategy frequency
                    await self._wait_for_next_cycle(strategy)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error in strategy execution loop: {e}")
                    strategy.record_error(str(e))
                    await self.strategy_repository.save(strategy)
                    
                    # Pause on repeated errors
                    if strategy.state.error_count > 5:
                        await strategy.pause()
                        break
                    
                    await asyncio.sleep(5)  # Brief pause before retry
                    
        except Exception as e:
            self.logger.error(f"Fatal error in strategy execution: {e}")
        finally:
            # Clean up
            self.execution_tasks.pop(strategy.id, None)
    
    async def _generate_signals(self, strategy: Strategy, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate trading signals using strategy implementation"""
        if not strategy.implementation:
            return []
        
        try:
            signals = await strategy.implementation.generate_signals(market_data)
            
            # Update strategy state
            strategy.state.last_signal_time = datetime.now(timezone.utc)
            await self.strategy_repository.save(strategy)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for strategy {strategy.id}: {e}")
            return []
    
    async def _process_signal(self, strategy: Strategy, signal: Signal, market_data: Dict[str, Any]) -> None:
        """Process and potentially execute a trading signal"""
        try:
            # Validate signal
            if not await self._validate_signal(strategy, signal, market_data):
                signal.reject("Signal validation failed")
                await self.signal_repository.save(signal)
                return
            
            # Save validated signal
            signal.validate()
            await self.signal_repository.save(signal)
            
            # Publish signal event
            event = SignalGeneratedEvent(
                aggregate_id=signal.id,
                signal_id=signal.id,
                strategy_id=strategy.id,
                symbol=signal.symbol,
                direction=signal.direction.value,
                signal_type=signal.signal_type.value,
                quantity=signal.quantity,
                current_price=signal.current_price,
                confidence=signal.metrics.confidence,
                reason=signal.reason
            )
            await self.signal_publisher.publish_event(event)
            
            # Execute signal if enabled
            if not strategy.parameters.use_paper_trading:
                await self._execute_signal(strategy, signal)
            
        except Exception as e:
            self.logger.error(f"Error processing signal {signal.id}: {e}")
            signal.reject(f"Processing error: {e}")
            await self.signal_repository.save(signal)
    
    async def _validate_signal(self, strategy: Strategy, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate a trading signal against strategy rules and risk limits"""
        try:
            # Basic validation
            if not signal.is_valid:
                return False
            
            # Strategy-specific validation
            if strategy.implementation:
                if not strategy.implementation.validate_signal(signal, market_data):
                    return False
            
            # Risk validation
            if not await self._validate_signal_risk(strategy, signal):
                return False
            
            # Capital validation
            if not await self._validate_signal_capital(strategy, signal):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    async def _validate_signal_risk(self, strategy: Strategy, signal: Signal) -> bool:
        """Validate signal against risk limits"""
        # Check position size limits
        max_position_size = strategy.parameters.max_position_size
        if signal.quantity * signal.current_price > max_position_size:
            return False
        
        # Check maximum positions
        open_positions = await self.position_repository.get_open_positions(strategy.id)
        if len(open_positions) >= strategy.parameters.max_positions:
            return False
        
        # Check symbol-specific limits
        symbol_positions = [p for p in open_positions if p.symbol == signal.symbol]
        if len(symbol_positions) > 0 and signal.signal_type.value == "ENTRY":
            return False  # Already have position in this symbol
        
        return True
    
    async def _validate_signal_capital(self, strategy: Strategy, signal: Signal) -> bool:
        """Validate signal against available capital"""
        required_capital = signal.quantity * signal.current_price
        
        if signal.direction == SignalDirection.BUY:
            return strategy.available_capital >= required_capital
        
        # For sell signals, check if we have the position
        if signal.direction == SignalDirection.SELL:
            positions = await self.position_repository.get_by_strategy(strategy.id)
            symbol_positions = [p for p in positions if p.symbol == signal.symbol and p.status == PositionStatus.OPEN]
            
            total_quantity = sum(p.quantity for p in symbol_positions)
            return total_quantity >= signal.quantity
        
        return True
    
    async def _execute_signal(self, strategy: Strategy, signal: Signal) -> None:
        """Execute a validated trading signal"""
        try:
            # This would integrate with actual exchange APIs
            # For now, simulate execution
            execution_price = signal.current_price
            execution_quantity = signal.quantity
            execution_fees = execution_quantity * execution_price * Decimal("0.001")  # 0.1% fee
            
            # Record signal execution
            signal.execute_partial(execution_quantity, execution_price, execution_fees)
            await self.signal_repository.save(signal)
            
            # Update or create position
            await self._update_position_from_signal(strategy, signal, execution_quantity, execution_price, execution_fees)
            
            # Update strategy capital usage
            if signal.direction == SignalDirection.BUY:
                notional_value = execution_quantity * execution_price
                strategy.allocate_capital(notional_value)
            elif signal.direction == SignalDirection.SELL:
                notional_value = execution_quantity * execution_price
                strategy.deallocate_capital(notional_value)
            
            await self.strategy_repository.save(strategy)
            
            self.logger.info(f"Executed signal {signal.id}: {execution_quantity} @ {execution_price}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal {signal.id}: {e}")
            raise
    
    async def _update_position_from_signal(self, strategy: Strategy, signal: Signal, 
                                         quantity: Decimal, price: Decimal, fees: Decimal) -> None:
        """Update position based on signal execution"""
        try:
            # Find existing position for this symbol
            positions = await self.position_repository.get_by_strategy(strategy.id)
            existing_position = next((p for p in positions if p.symbol == signal.symbol and p.status == PositionStatus.OPEN), None)
            
            # Create trade record
            trade = Trade(
                timestamp=datetime.now(timezone.utc),
                symbol=signal.symbol,
                side="BUY" if signal.direction == SignalDirection.BUY else "SELL",
                quantity=quantity,
                price=price,
                fee=fees,
                exchange=signal.exchange,
                order_id=str(signal.id)
            )
            
            if existing_position:
                # Add trade to existing position
                existing_position.add_trade(trade)
                await self.position_repository.save(existing_position)
                
                # Publish position changed event
                event = PositionChangedEvent(
                    aggregate_id=existing_position.id,
                    position_id=existing_position.id,
                    strategy_id=strategy.id,
                    symbol=signal.symbol,
                    change_type="TRADE_ADDED",
                    position_type=existing_position.position_type.value,
                    status=existing_position.status.value,
                    quantity=existing_position.quantity,
                    current_price=price,
                    unrealized_pnl=existing_position.unrealized_pnl,
                    total_pnl=existing_position.total_pnl
                )
                await self.signal_publisher.publish_event(event)
                
            else:
                # Create new position
                new_position = Position(
                    strategy_id=strategy.id,
                    symbol=signal.symbol,
                    exchange=signal.exchange,
                    current_price=price
                )
                new_position.add_trade(trade)
                await self.position_repository.save(new_position)
                
                # Publish position opened event
                event = PositionChangedEvent(
                    aggregate_id=new_position.id,
                    position_id=new_position.id,
                    strategy_id=strategy.id,
                    symbol=signal.symbol,
                    change_type="OPENED",
                    position_type=new_position.position_type.value,
                    status=new_position.status.value,
                    quantity=new_position.quantity,
                    current_price=price,
                    unrealized_pnl=new_position.unrealized_pnl,
                    total_pnl=new_position.total_pnl
                )
                await self.signal_publisher.publish_event(event)
            
        except Exception as e:
            self.logger.error(f"Error updating position from signal: {e}")
            raise
    
    async def _update_positions(self, strategy: Strategy, market_data: Dict[str, Any]) -> None:
        """Update all positions with current market prices"""
        try:
            positions = await self.position_repository.get_open_positions(strategy.id)
            
            for position in positions:
                if position.symbol in market_data:
                    current_price = Decimal(str(market_data[position.symbol].get('price', position.current_price)))
                    old_pnl = position.total_pnl
                    
                    position.update_current_price(current_price)
                    await self.position_repository.save(position)
                    
                    # Check if position should be closed due to risk rules
                    if position.should_close():
                        await self._close_position(position, "Risk rule triggered")
                    
                    # Publish position update if significant P&L change
                    pnl_change = position.total_pnl - old_pnl
                    if abs(pnl_change) > Decimal("10"):  # $10 threshold
                        event = PositionChangedEvent(
                            aggregate_id=position.id,
                            position_id=position.id,
                            strategy_id=strategy.id,
                            symbol=position.symbol,
                            change_type="UPDATED",
                            position_type=position.position_type.value,
                            status=position.status.value,
                            quantity=position.quantity,
                            current_price=current_price,
                            unrealized_pnl=position.unrealized_pnl,
                            total_pnl=position.total_pnl,
                            pnl_change=pnl_change
                        )
                        await self.signal_publisher.publish_event(event)
                        
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _close_position(self, position: Position, reason: str) -> None:
        """Close a position"""
        try:
            position.close_position()
            await self.position_repository.save(position)
            
            # Publish position closed event
            event = PositionChangedEvent(
                aggregate_id=position.id,
                position_id=position.id,
                strategy_id=position.strategy_id,
                symbol=position.symbol,
                change_type="CLOSED",
                position_type=position.position_type.value,
                status=position.status.value,
                quantity=Decimal("0"),
                current_price=position.current_price,
                realized_pnl=position.realized_pnl,
                total_pnl=position.total_pnl
            )
            await self.signal_publisher.publish_event(event)
            
            self.logger.info(f"Closed position {position.id}: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error closing position {position.id}: {e}")
    
    async def _check_risk_limits(self, strategy: Strategy) -> None:
        """Check and enforce risk limits"""
        try:
            positions = await self.position_repository.get_open_positions(strategy.id)
            
            # Calculate total unrealized P&L
            total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
            
            # Check maximum loss limit
            if strategy.parameters.max_daily_loss:
                if total_unrealized_pnl <= -strategy.parameters.max_daily_loss:
                    await self.stop_strategy(strategy.id, "Daily loss limit exceeded")
                    return
            
            # Check maximum drawdown
            if strategy.parameters.max_drawdown:
                drawdown_pct = abs(total_unrealized_pnl) / strategy.allocated_capital
                if drawdown_pct >= strategy.parameters.max_drawdown:
                    await self.stop_strategy(strategy.id, "Maximum drawdown exceeded")
                    return
            
            # Check individual position limits
            for position in positions:
                if position.check_max_loss() or position.check_stop_loss():
                    await self._close_position(position, "Position risk limit breached")
                    
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    async def _get_market_data(self, symbols: set) -> Dict[str, Any]:
        """Get current market data for symbols"""
        # In a real implementation, this would fetch from market data providers
        # For now, simulate with cached/mock data
        market_data = {}
        for symbol in symbols:
            if symbol in self.market_data_cache:
                market_data[symbol] = self.market_data_cache[symbol]
            else:
                # Mock data
                market_data[symbol] = {
                    'price': 50000 + (hash(symbol) % 10000),  # Mock price
                    'volume': 1000000,
                    'bid': 49999,
                    'ask': 50001,
                    'timestamp': datetime.now(timezone.utc)
                }
        return market_data
    
    async def _wait_for_next_cycle(self, strategy: Strategy) -> None:
        """Wait for next execution cycle based on strategy frequency"""
        frequency_map = {
            "1s": 1,
            "5s": 5,
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        
        wait_time = frequency_map.get(strategy.parameters.signal_frequency, 60)
        await asyncio.sleep(wait_time)
    
    async def get_execution_status(self, strategy_id: UUID) -> Dict[str, Any]:
        """Get current execution status for a strategy"""
        try:
            strategy = await self.strategy_repository.get_by_id(strategy_id)
            if not strategy:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
            
            positions = await self.position_repository.get_by_strategy(strategy_id)
            recent_signals = await self.signal_repository.get_recent_signals(24, strategy_id)
            
            return {
                "strategy_id": strategy_id,
                "status": strategy.status.value,
                "is_running": strategy_id in self.execution_tasks,
                "positions_count": len([p for p in positions if p.status == PositionStatus.OPEN]),
                "total_pnl": sum(p.total_pnl for p in positions),
                "signals_today": len(recent_signals),
                "last_signal_time": strategy.state.last_signal_time,
                "error_count": strategy.state.error_count,
                "capital_utilization": strategy.capital_utilization
            }
            
        except Exception as e:
            self.logger.error(f"Error getting execution status: {e}")
            raise


class BacktestService:
    """
    Service for backtesting trading strategies against historical data.
    Provides comprehensive performance analysis and optimization capabilities.
    """
    
    def __init__(self,
                 strategy_repository: IStrategyRepository,
                 position_repository: IPositionRepository,
                 signal_repository: ISignalRepository):
        self.strategy_repository = strategy_repository
        self.position_repository = position_repository
        self.signal_repository = signal_repository
        self.logger = logging.getLogger(__name__)
    
    async def run_backtest(self,
                          strategy_id: UUID,
                          start_date: datetime,
                          end_date: datetime,
                          initial_capital: Decimal = Decimal("100000"),
                          historical_data: Optional[Dict[str, List[Dict]]] = None) -> BacktestResult:
        """
        Run a comprehensive backtest of a strategy
        
        Args:
            strategy_id: ID of strategy to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital amount
            historical_data: Historical market data
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        try:
            # Get strategy
            strategy = await self.strategy_repository.get_by_id(strategy_id)
            if not strategy:
                raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
            
            # Initialize backtest state
            backtest_state = BacktestState(
                strategy=strategy,
                initial_capital=initial_capital,
                current_capital=initial_capital,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get or generate historical data
            if not historical_data:
                historical_data = await self._get_historical_data(strategy.symbols, start_date, end_date)
            
            # Run simulation
            await self._simulate_strategy(backtest_state, historical_data)
            
            # Calculate final metrics
            result = self._calculate_backtest_metrics(backtest_state)
            
            self.logger.info(f"Backtest completed for strategy {strategy_id}: {result.total_return_pct:.2f}% return")
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    async def _simulate_strategy(self, state: BacktestState, historical_data: Dict[str, List[Dict]]) -> None:
        """Simulate strategy execution over historical data"""
        try:
            # Initialize strategy implementation
            if state.strategy.implementation:
                await state.strategy.implementation.initialize({})
            
            # Get all timestamps and sort
            all_timestamps = set()
            for symbol_data in historical_data.values():
                for candle in symbol_data:
                    all_timestamps.add(candle['timestamp'])
            
            sorted_timestamps = sorted(all_timestamps)
            
            # Simulate each time step
            for timestamp in sorted_timestamps:
                current_data = {}
                
                # Get market data for this timestamp
                for symbol, symbol_data in historical_data.items():
                    matching_candles = [c for c in symbol_data if c['timestamp'] == timestamp]
                    if matching_candles:
                        current_data[symbol] = matching_candles[0]
                
                if not current_data:
                    continue
                
                # Update positions with current prices
                await self._update_backtest_positions(state, current_data, timestamp)
                
                # Generate signals
                if state.strategy.implementation:
                    await state.strategy.implementation.update_state(current_data)
                    signals = await state.strategy.implementation.generate_signals(current_data)
                    
                    # Process signals
                    for signal in signals:
                        await self._process_backtest_signal(state, signal, current_data, timestamp)
                
                # Record equity curve point
                total_equity = state.current_capital + sum(p.unrealized_pnl for p in state.positions.values())
                state.equity_curve.append({
                    'timestamp': timestamp,
                    'equity': float(total_equity),
                    'capital': float(state.current_capital),
                    'unrealized_pnl': float(sum(p.unrealized_pnl for p in state.positions.values()))
                })
                
        except Exception as e:
            self.logger.error(f"Error in strategy simulation: {e}")
            raise
    
    async def _process_backtest_signal(self, state: BacktestState, signal: Signal, 
                                     market_data: Dict[str, Any], timestamp: datetime) -> None:
        """Process a signal in backtest mode"""
        try:
            # Validate signal
            if not self._validate_backtest_signal(state, signal, market_data):
                return
            
            # Simulate execution
            execution_price = signal.current_price
            execution_quantity = signal.quantity
            execution_fees = execution_quantity * execution_price * Decimal("0.001")
            
            # Create trade record
            trade_record = {
                'timestamp': timestamp,
                'symbol': signal.symbol,
                'side': signal.direction.value,
                'quantity': float(execution_quantity),
                'price': float(execution_price),
                'fees': float(execution_fees),
                'signal_type': signal.signal_type.value,
                'pnl': 0  # Will be calculated when position is closed
            }
            
            # Update position
            position_id = f"{signal.symbol}_{state.strategy.id}"
            
            if position_id in state.positions:
                position = state.positions[position_id]
                self._update_backtest_position(position, signal, execution_price, execution_quantity, execution_fees)
            else:
                # Create new position
                position = BacktestPosition(
                    symbol=signal.symbol,
                    strategy_id=state.strategy.id,
                    opened_at=timestamp
                )
                self._update_backtest_position(position, signal, execution_price, execution_quantity, execution_fees)
                state.positions[position_id] = position
            
            # Update capital
            if signal.direction == SignalDirection.BUY:
                state.current_capital -= (execution_quantity * execution_price + execution_fees)
            else:
                state.current_capital += (execution_quantity * execution_price - execution_fees)
            
            state.trades.append(trade_record)
            
        except Exception as e:
            self.logger.error(f"Error processing backtest signal: {e}")
    
    def _validate_backtest_signal(self, state: BacktestState, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate signal in backtest context"""
        # Check if we have sufficient capital for buy orders
        if signal.direction == SignalDirection.BUY:
            required_capital = signal.quantity * signal.current_price
            if state.current_capital < required_capital:
                return False
        
        # Check if we have sufficient position for sell orders
        if signal.direction == SignalDirection.SELL:
            position_id = f"{signal.symbol}_{state.strategy.id}"
            if position_id not in state.positions:
                return False
            
            position = state.positions[position_id]
            if position.quantity < signal.quantity:
                return False
        
        return True
    
    def _update_backtest_position(self, position: BacktestPosition, signal: Signal, 
                                price: Decimal, quantity: Decimal, fees: Decimal) -> None:
        """Update backtest position with new trade"""
        if signal.direction == SignalDirection.BUY:
            # Add to position
            total_cost = position.quantity * position.average_price + quantity * price
            position.quantity += quantity
            position.average_price = total_cost / position.quantity if position.quantity > 0 else price
            position.fees += fees
        
        elif signal.direction == SignalDirection.SELL:
            # Reduce position
            position.quantity -= quantity
            position.fees += fees
            
            # Calculate realized P&L for the sold portion
            realized_pnl = (price - position.average_price) * quantity - fees
            position.realized_pnl += realized_pnl
    
    async def _update_backtest_positions(self, state: BacktestState, market_data: Dict[str, Any], timestamp: datetime) -> None:
        """Update all positions with current market prices"""
        for position in state.positions.values():
            if position.symbol in market_data:
                position.current_price = Decimal(str(market_data[position.symbol]['close']))
                position.last_updated = timestamp
    
    def _calculate_backtest_metrics(self, state: BacktestState) -> BacktestResult:
        """Calculate comprehensive backtest metrics"""
        final_capital = state.current_capital + sum(p.total_value for p in state.positions.values())
        total_return = final_capital - state.initial_capital
        total_return_pct = (total_return / state.initial_capital) * 100
        
        # Calculate trade-based metrics
        winning_trades = [t for t in state.trades if t['pnl'] > 0]
        losing_trades = [t for t in state.trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / len(state.trades) * 100 if state.trades else 0
        avg_trade_return = sum(t['pnl'] for t in state.trades) / len(state.trades) if state.trades else 0
        
        # Calculate Sharpe ratio
        if state.equity_curve:
            returns = []
            for i in range(1, len(state.equity_curve)):
                prev_equity = state.equity_curve[i-1]['equity']
                curr_equity = state.equity_curve[i]['equity']
                daily_return = (curr_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                returns.append(daily_return)
            
            if returns:
                import statistics
                avg_return = statistics.mean(returns)
                volatility = statistics.stdev(returns) if len(returns) > 1 else 0
                sharpe_ratio = (avg_return / volatility) * (252 ** 0.5) if volatility > 0 else 0  # Annualized
            else:
                sharpe_ratio = 0
                volatility = 0
        else:
            sharpe_ratio = 0
            volatility = 0
        
        # Calculate maximum drawdown
        max_equity = state.initial_capital
        max_drawdown = 0
        
        for point in state.equity_curve:
            equity = point['equity']
            if equity > max_equity:
                max_equity = equity
            
            drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return BacktestResult(
            strategy_id=state.strategy.id,
            start_date=state.start_date,
            end_date=state.end_date,
            initial_capital=state.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=Decimal(str(sharpe_ratio)),
            max_drawdown=Decimal(str(max_drawdown)),
            win_rate=Decimal(str(win_rate)),
            total_trades=len(state.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_return=Decimal(str(avg_trade_return)),
            volatility=Decimal(str(volatility)),
            trades=state.trades,
            equity_curve=state.equity_curve,
            performance_metrics={
                'total_fees': float(sum(p.fees for p in state.positions.values())),
                'max_positions': len(state.positions),
                'avg_position_duration': self._calculate_avg_position_duration(state),
                'profit_factor': self._calculate_profit_factor(state.trades)
            }
        )
    
    def _calculate_avg_position_duration(self, state: BacktestState) -> float:
        """Calculate average position duration in hours"""
        durations = []
        for position in state.positions.values():
            if position.closed_at:
                duration = (position.closed_at - position.opened_at).total_seconds() / 3600
                durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    async def _get_historical_data(self, symbols: set, start_date: datetime, end_date: datetime) -> Dict[str, List[Dict]]:
        """Get historical market data for symbols"""
        # In a real implementation, this would fetch from data providers
        # For now, generate mock data
        historical_data = {}
        
        for symbol in symbols:
            data = []
            current_date = start_date
            base_price = 50000 + (hash(symbol) % 10000)
            
            while current_date <= end_date:
                # Generate mock OHLCV data
                price_change = (hash(str(current_date) + symbol) % 200 - 100) / 100  # -1% to +1%
                base_price *= (1 + price_change / 100)
                
                candle = {
                    'timestamp': current_date,
                    'open': float(base_price),
                    'high': float(base_price * 1.02),
                    'low': float(base_price * 0.98),
                    'close': float(base_price),
                    'volume': 1000000
                }
                data.append(candle)
                
                current_date += timedelta(hours=1)  # Hourly data
            
            historical_data[symbol] = data
        
        return historical_data


@dataclass
class BacktestState:
    """Internal state for backtesting"""
    strategy: Strategy
    initial_capital: Decimal
    current_capital: Decimal
    start_date: datetime
    end_date: datetime
    positions: Dict[str, Any] = None
    trades: List[Dict[str, Any]] = None
    equity_curve: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = {}
        if self.trades is None:
            self.trades = []
        if self.equity_curve is None:
            self.equity_curve = []


@dataclass
class BacktestPosition:
    """Position representation for backtesting"""
    symbol: str
    strategy_id: UUID
    opened_at: datetime
    quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    current_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    closed_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate unrealized P&L"""
        if self.quantity == 0:
            return Decimal("0")
        return (self.current_price - self.average_price) * self.quantity
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L"""
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def total_value(self) -> Decimal:
        """Calculate total position value"""
        return self.current_price * self.quantity