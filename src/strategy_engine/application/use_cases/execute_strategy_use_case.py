"""
Use Cases - Strategy Engine Application Use Cases

This module implements use cases for strategy execution and backtesting
following clean architecture principles with comprehensive error handling.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timezone
from decimal import Decimal
import asyncio
import logging
from dataclasses import dataclass

# Import domain entities and application services
from ..domain.entities.strategy import Strategy, StrategyStatus
from ..domain.entities.position import Position
from ..domain.entities.signal import Signal
from ..application.services.strategy_executor_service import StrategyExecutorService, ExecutionResult
from ..application.services.backtest_service import BacktestService, BacktestResult


class UseCaseError(Exception):
    """Base exception for use case errors"""
    pass


class ValidationError(UseCaseError):
    """Raised when input validation fails"""
    pass


class ExecutionError(UseCaseError):
    """Raised when execution fails"""
    pass


@dataclass
class ExecuteStrategyRequest:
    """Request object for executing a strategy"""
    strategy_id: UUID
    market_data: Optional[Dict[str, Any]] = None
    force_start: bool = False
    paper_trading: bool = True
    risk_override: bool = False


@dataclass
class ExecuteStrategyResponse:
    """Response object for strategy execution"""
    success: bool
    strategy_id: UUID
    execution_result: Optional[ExecutionResult] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


@dataclass
class BacktestStrategyRequest:
    """Request object for backtesting a strategy"""
    strategy_id: UUID
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Decimal("100000")
    symbols: Optional[List[str]] = None
    parameters_override: Optional[Dict[str, Any]] = None
    include_slippage: bool = True
    include_fees: bool = True
    benchmark_symbol: Optional[str] = None


@dataclass
class BacktestStrategyResponse:
    """Response object for strategy backtesting"""
    success: bool
    strategy_id: UUID
    backtest_result: Optional[BacktestResult] = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ExecuteStrategyUseCase:
    """
    Use case for executing a trading strategy in real-time or paper trading mode.
    Handles validation, execution orchestration, and result aggregation.
    """
    
    def __init__(self, strategy_executor: StrategyExecutorService):
        self.strategy_executor = strategy_executor
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, request: ExecuteStrategyRequest) -> ExecuteStrategyResponse:
        """
        Execute a trading strategy
        
        Args:
            request: Strategy execution request
            
        Returns:
            ExecuteStrategyResponse with execution results
        """
        response = ExecuteStrategyResponse(
            success=False,
            strategy_id=request.strategy_id
        )
        
        try:
            # Validate request
            validation_errors = await self._validate_request(request)
            if validation_errors:
                response.errors.extend(validation_errors)
                return response
            
            # Get strategy details for logging
            strategy = await self.strategy_executor.strategy_repository.get_by_id(request.strategy_id)
            if not strategy:
                response.errors.append(f"Strategy {request.strategy_id} not found")
                return response
            
            self.logger.info(f"Starting execution of strategy '{strategy.name}' ({request.strategy_id})")
            
            # Check if strategy is already running
            status = await self.strategy_executor.get_execution_status(request.strategy_id)
            if status.get("is_running", False) and not request.force_start:
                response.warnings.append("Strategy is already running")
                response.success = True
                return response
            
            # Configure paper trading if requested
            if request.paper_trading:
                strategy.parameters.use_paper_trading = True
                await self.strategy_executor.strategy_repository.save(strategy)
                response.warnings.append("Strategy configured for paper trading")
            
            # Start strategy execution
            success = await self.strategy_executor.start_strategy(
                request.strategy_id,
                request.market_data
            )
            
            if not success:
                response.errors.append("Failed to start strategy execution")
                return response
            
            # Wait briefly and get initial execution status
            await asyncio.sleep(2)
            execution_status = await self.strategy_executor.get_execution_status(request.strategy_id)
            
            # Create execution result
            response.execution_result = ExecutionResult(
                strategy_id=request.strategy_id,
                signals_generated=execution_status.get("signals_today", 0),
                signals_executed=0,  # Will be updated by monitoring
                positions_opened=execution_status.get("positions_count", 0),
                positions_closed=0,
                total_pnl=execution_status.get("total_pnl", Decimal("0")),
                execution_time=2.0,  # Initial startup time
                errors=[],
                warnings=response.warnings.copy()
            )
            
            response.success = True
            self.logger.info(f"Successfully started strategy execution: {request.strategy_id}")
            
        except Exception as e:
            self.logger.error(f"Error executing strategy {request.strategy_id}: {e}")
            response.errors.append(f"Execution error: {str(e)}")
        
        return response
    
    async def _validate_request(self, request: ExecuteStrategyRequest) -> List[str]:
        """Validate strategy execution request"""
        errors = []
        
        try:
            # Check if strategy exists
            strategy = await self.strategy_executor.strategy_repository.get_by_id(request.strategy_id)
            if not strategy:
                errors.append(f"Strategy {request.strategy_id} not found")
                return errors
            
            # Validate strategy state
            if strategy.status == StrategyStatus.ERROR and not request.force_start:
                errors.append("Strategy is in error state. Use force_start to override.")
            
            # Validate strategy configuration
            if not strategy.symbols:
                errors.append("Strategy has no symbols configured")
            
            if not strategy.exchanges:
                errors.append("Strategy has no exchanges configured")
            
            # Validate strategy implementation
            if not strategy.implementation and not strategy.module_path:
                errors.append("Strategy has no implementation configured")
            
            # Risk validation
            if not request.risk_override:
                if strategy.allocated_capital <= 0:
                    errors.append("Strategy has no allocated capital")
                
                if strategy.parameters.max_position_size <= 0:
                    errors.append("Invalid position size configuration")
            
            # Market data validation
            if request.market_data:
                required_symbols = strategy.symbols
                provided_symbols = set(request.market_data.keys())
                missing_symbols = required_symbols - provided_symbols
                
                if missing_symbols:
                    errors.append(f"Missing market data for symbols: {missing_symbols}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    async def stop_strategy(self, strategy_id: UUID, reason: str = "Manual stop") -> ExecuteStrategyResponse:
        """
        Stop a running strategy
        
        Args:
            strategy_id: ID of strategy to stop
            reason: Reason for stopping
            
        Returns:
            ExecuteStrategyResponse indicating success/failure
        """
        response = ExecuteStrategyResponse(
            success=False,
            strategy_id=strategy_id
        )
        
        try:
            success = await self.strategy_executor.stop_strategy(strategy_id, reason)
            
            if success:
                response.success = True
                response.warnings.append(f"Strategy stopped: {reason}")
                self.logger.info(f"Successfully stopped strategy {strategy_id}: {reason}")
            else:
                response.errors.append("Failed to stop strategy")
                
        except Exception as e:
            self.logger.error(f"Error stopping strategy {strategy_id}: {e}")
            response.errors.append(f"Stop error: {str(e)}")
        
        return response
    
    async def get_strategy_status(self, strategy_id: UUID) -> Dict[str, Any]:
        """Get current status of a strategy"""
        try:
            return await self.strategy_executor.get_execution_status(strategy_id)
        except Exception as e:
            self.logger.error(f"Error getting strategy status {strategy_id}: {e}")
            return {
                "strategy_id": strategy_id,
                "error": str(e),
                "status": "UNKNOWN"
            }


class BacktestStrategyUseCase:
    """
    Use case for backtesting trading strategies against historical data.
    Provides comprehensive analysis and performance metrics.
    """
    
    def __init__(self, backtest_service: BacktestService):
        self.backtest_service = backtest_service
        self.logger = logging.getLogger(__name__)
    
    async def execute(self, request: BacktestStrategyRequest) -> BacktestStrategyResponse:
        """
        Execute a strategy backtest
        
        Args:
            request: Backtest request parameters
            
        Returns:
            BacktestStrategyResponse with results and metrics
        """
        response = BacktestStrategyResponse(
            success=False,
            strategy_id=request.strategy_id
        )
        
        try:
            # Validate request
            validation_errors = await self._validate_request(request)
            if validation_errors:
                response.errors.extend(validation_errors)
                return response
            
            # Get strategy for logging
            strategy = await self.backtest_service.strategy_repository.get_by_id(request.strategy_id)
            if not strategy:
                response.errors.append(f"Strategy {request.strategy_id} not found")
                return response
            
            self.logger.info(f"Starting backtest of strategy '{strategy.name}' from {request.start_date} to {request.end_date}")
            
            # Apply parameter overrides if provided
            original_parameters = None
            if request.parameters_override:
                original_parameters = strategy.parameters
                strategy = await self._apply_parameter_overrides(strategy, request.parameters_override)
            
            try:
                # Generate historical data if needed
                historical_data = await self._prepare_historical_data(request)
                
                # Run backtest
                backtest_result = await self.backtest_service.run_backtest(
                    strategy_id=request.strategy_id,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    historical_data=historical_data
                )
                
                # Apply trading costs if requested
                if request.include_slippage or request.include_fees:
                    backtest_result = await self._apply_trading_costs(backtest_result, request)
                
                # Calculate benchmark comparison if requested
                if request.benchmark_symbol:
                    backtest_result = await self._add_benchmark_comparison(backtest_result, request)
                
                response.backtest_result = backtest_result
                response.success = True
                
                self.logger.info(f"Backtest completed: {backtest_result.total_return_pct:.2f}% return, "
                               f"{backtest_result.sharpe_ratio:.2f} Sharpe ratio")
                
                # Add performance warnings
                if backtest_result.max_drawdown > Decimal("0.20"):
                    response.warnings.append("High maximum drawdown detected (>20%)")
                
                if backtest_result.win_rate < Decimal("40"):
                    response.warnings.append("Low win rate detected (<40%)")
                
                if backtest_result.sharpe_ratio < Decimal("1.0"):
                    response.warnings.append("Low Sharpe ratio detected (<1.0)")
                
            finally:
                # Restore original parameters if they were overridden
                if original_parameters:
                    strategy.parameters = original_parameters
                    await self.backtest_service.strategy_repository.save(strategy)
        
        except Exception as e:
            self.logger.error(f"Error backtesting strategy {request.strategy_id}: {e}")
            response.errors.append(f"Backtest error: {str(e)}")
        
        return response
    
    async def _validate_request(self, request: BacktestStrategyRequest) -> List[str]:
        """Validate backtest request"""
        errors = []
        
        try:
            # Date validation
            if request.start_date >= request.end_date:
                errors.append("Start date must be before end date")
            
            if request.end_date > datetime.now(timezone.utc):
                errors.append("End date cannot be in the future")
            
            # Duration validation
            duration = request.end_date - request.start_date
            if duration.days < 1:
                errors.append("Backtest period must be at least 1 day")
            
            if duration.days > 3650:  # 10 years
                errors.append("Backtest period cannot exceed 10 years")
            
            # Capital validation
            if request.initial_capital <= 0:
                errors.append("Initial capital must be positive")
            
            if request.initial_capital < Decimal("1000"):
                errors.append("Initial capital should be at least $1,000 for realistic results")
            
            # Strategy validation
            strategy = await self.backtest_service.strategy_repository.get_by_id(request.strategy_id)
            if not strategy:
                errors.append(f"Strategy {request.strategy_id} not found")
                return errors
            
            # Check strategy implementation
            if not strategy.implementation and not strategy.module_path:
                errors.append("Strategy has no implementation for backtesting")
            
            # Symbol validation
            symbols_to_test = request.symbols or list(strategy.symbols)
            if not symbols_to_test:
                errors.append("No symbols specified for backtesting")
            
            # Parameters override validation
            if request.parameters_override:
                param_errors = self._validate_parameter_overrides(request.parameters_override)
                errors.extend(param_errors)
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def _validate_parameter_overrides(self, overrides: Dict[str, Any]) -> List[str]:
        """Validate parameter override values"""
        errors = []
        
        # Validate numeric parameters
        numeric_params = [
            'max_position_size', 'max_positions', 'base_position_size',
            'max_daily_loss', 'max_drawdown', 'stop_loss_pct', 'take_profit_pct',
            'max_correlation', 'max_leverage', 'lookback_period'
        ]
        
        for param in numeric_params:
            if param in overrides:
                try:
                    value = Decimal(str(overrides[param]))
                    if value < 0:
                        errors.append(f"Parameter {param} must be non-negative")
                except (ValueError, TypeError):
                    errors.append(f"Parameter {param} must be numeric")
        
        # Validate specific constraints
        if 'max_positions' in overrides:
            if not isinstance(overrides['max_positions'], int) or overrides['max_positions'] < 1:
                errors.append("max_positions must be a positive integer")
        
        if 'max_leverage' in overrides:
            try:
                leverage = Decimal(str(overrides['max_leverage']))
                if leverage > Decimal("10"):
                    errors.append("max_leverage should not exceed 10x")
            except (ValueError, TypeError):
                pass
        
        return errors
    
    async def _apply_parameter_overrides(self, strategy: Strategy, overrides: Dict[str, Any]) -> Strategy:
        """Apply parameter overrides to strategy"""
        # Create a copy of the strategy with modified parameters
        for param_name, param_value in overrides.items():
            if hasattr(strategy.parameters, param_name):
                setattr(strategy.parameters, param_name, param_value)
                self.logger.info(f"Override parameter {param_name} = {param_value}")
        
        await self.backtest_service.strategy_repository.save(strategy)
        return strategy
    
    async def _prepare_historical_data(self, request: BacktestStrategyRequest) -> Dict[str, List[Dict]]:
        """Prepare historical data for backtesting"""
        # In a real implementation, this would fetch from data providers
        # For now, we'll generate synthetic data or use the service's built-in method
        
        strategy = await self.backtest_service.strategy_repository.get_by_id(request.strategy_id)
        symbols = request.symbols or list(strategy.symbols)
        
        # Use the service's method to get historical data
        return await self.backtest_service._get_historical_data(
            set(symbols),
            request.start_date,
            request.end_date
        )
    
    async def _apply_trading_costs(self, result: BacktestResult, request: BacktestStrategyRequest) -> BacktestResult:
        """Apply realistic trading costs to backtest results"""
        if not (request.include_slippage or request.include_fees):
            return result
        
        # Calculate additional costs
        additional_costs = Decimal("0")
        
        if request.include_slippage:
            # Apply slippage cost (typically 0.01-0.05% per trade)
            slippage_rate = Decimal("0.0002")  # 0.02%
            trade_volume = sum(abs(Decimal(str(trade['quantity'])) * Decimal(str(trade['price']))) 
                             for trade in result.trades)
            slippage_cost = trade_volume * slippage_rate
            additional_costs += slippage_cost
        
        if request.include_fees:
            # Apply additional fees beyond the basic 0.1% already included
            # This could include funding costs, withdrawal fees, etc.
            additional_fee_rate = Decimal("0.0005")  # 0.05%
            trade_volume = sum(abs(Decimal(str(trade['quantity'])) * Decimal(str(trade['price']))) 
                             for trade in result.trades)
            additional_fees = trade_volume * additional_fee_rate
            additional_costs += additional_fees
        
        # Adjust final capital and returns
        result.final_capital -= additional_costs
        result.total_return -= additional_costs
        result.total_return_pct = (result.total_return / result.initial_capital) * 100
        
        # Update performance metrics
        result.performance_metrics['additional_costs'] = float(additional_costs)
        result.performance_metrics['slippage_applied'] = request.include_slippage
        result.performance_metrics['additional_fees_applied'] = request.include_fees
        
        return result
    
    async def _add_benchmark_comparison(self, result: BacktestResult, request: BacktestStrategyRequest) -> BacktestResult:
        """Add benchmark comparison to backtest results"""
        if not request.benchmark_symbol:
            return result
        
        try:
            # Get benchmark data for the same period
            benchmark_data = await self.backtest_service._get_historical_data(
                {request.benchmark_symbol},
                request.start_date,
                request.end_date
            )
            
            if request.benchmark_symbol in benchmark_data:
                symbol_data = benchmark_data[request.benchmark_symbol]
                
                if symbol_data:
                    start_price = Decimal(str(symbol_data[0]['close']))
                    end_price = Decimal(str(symbol_data[-1]['close']))
                    
                    benchmark_return = ((end_price - start_price) / start_price) * 100
                    
                    # Calculate relative performance
                    relative_performance = result.total_return_pct - benchmark_return
                    
                    # Add to performance metrics
                    result.performance_metrics.update({
                        'benchmark_symbol': request.benchmark_symbol,
                        'benchmark_return_pct': float(benchmark_return),
                        'relative_performance_pct': float(relative_performance),
                        'outperformed_benchmark': relative_performance > 0
                    })
        
        except Exception as e:
            self.logger.warning(f"Failed to calculate benchmark comparison: {e}")
            result.performance_metrics['benchmark_error'] = str(e)
        
        return result
    
    async def compare_strategies(self, 
                                strategy_ids: List[UUID], 
                                start_date: datetime, 
                                end_date: datetime,
                                initial_capital: Decimal = Decimal("100000")) -> Dict[str, Any]:
        """
        Compare multiple strategies using backtesting
        
        Args:
            strategy_ids: List of strategy IDs to compare
            start_date: Start date for comparison
            end_date: End date for comparison
            initial_capital: Initial capital for each strategy
            
        Returns:
            Comparison results and rankings
        """
        try:
            results = {}
            
            # Run backtest for each strategy
            for strategy_id in strategy_ids:
                request = BacktestStrategyRequest(
                    strategy_id=strategy_id,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                
                response = await self.execute(request)
                if response.success and response.backtest_result:
                    results[str(strategy_id)] = response.backtest_result
                else:
                    self.logger.warning(f"Backtest failed for strategy {strategy_id}: {response.errors}")
            
            # Compare results
            if not results:
                return {"error": "No successful backtests to compare"}
            
            # Create comparison metrics
            comparison = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "initial_capital": float(initial_capital)
                },
                "strategies": {},
                "rankings": {}
            }
            
            # Add individual strategy results
            for strategy_id, result in results.items():
                strategy = await self.backtest_service.strategy_repository.get_by_id(UUID(strategy_id))
                comparison["strategies"][strategy_id] = {
                    "name": strategy.name if strategy else "Unknown",
                    "total_return_pct": float(result.total_return_pct),
                    "sharpe_ratio": float(result.sharpe_ratio),
                    "max_drawdown": float(result.max_drawdown),
                    "win_rate": float(result.win_rate),
                    "total_trades": result.total_trades,
                    "volatility": float(result.volatility)
                }
            
            # Calculate rankings
            metrics_to_rank = ['total_return_pct', 'sharpe_ratio', 'win_rate']
            for metric in metrics_to_rank:
                sorted_strategies = sorted(
                    comparison["strategies"].items(),
                    key=lambda x: x[1][metric],
                    reverse=True
                )
                comparison["rankings"][metric] = [
                    {"strategy_id": sid, "value": data[metric], "rank": i+1}
                    for i, (sid, data) in enumerate(sorted_strategies)
                ]
            
            # Add summary statistics
            comparison["summary"] = {
                "best_return": max(data["total_return_pct"] for data in comparison["strategies"].values()),
                "best_sharpe": max(data["sharpe_ratio"] for data in comparison["strategies"].values()),
                "lowest_drawdown": min(data["max_drawdown"] for data in comparison["strategies"].values()),
                "avg_return": sum(data["total_return_pct"] for data in comparison["strategies"].values()) / len(comparison["strategies"]),
                "strategies_compared": len(results)
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {"error": str(e)}


class StrategyOptimizationUseCase:
    """Use case for optimizing strategy parameters through backtesting"""
    
    def __init__(self, backtest_use_case: BacktestStrategyUseCase):
        self.backtest_use_case = backtest_use_case
        self.logger = logging.getLogger(__name__)
    
    async def optimize_parameters(self,
                                strategy_id: UUID,
                                start_date: datetime,
                                end_date: datetime,
                                parameter_ranges: Dict[str, List],
                                optimization_metric: str = "sharpe_ratio",
                                max_iterations: int = 100) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search or random search
        
        Args:
            strategy_id: Strategy to optimize
            start_date: Start date for optimization period
            end_date: End date for optimization period
            parameter_ranges: Dict of parameter names to lists of values to test
            optimization_metric: Metric to optimize (sharpe_ratio, total_return_pct, etc.)
            max_iterations: Maximum number of parameter combinations to test
            
        Returns:
            Optimization results with best parameters and performance
        """
        try:
            import itertools
            import random
            
            # Generate parameter combinations
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            # Use grid search for small parameter spaces, random search for large ones
            all_combinations = list(itertools.product(*param_values))
            
            if len(all_combinations) > max_iterations:
                # Random sampling
                combinations_to_test = random.sample(all_combinations, max_iterations)
                self.logger.info(f"Using random search: {max_iterations} out of {len(all_combinations)} combinations")
            else:
                # Full grid search
                combinations_to_test = all_combinations
                self.logger.info(f"Using grid search: {len(combinations_to_test)} combinations")
            
            results = []
            best_result = None
            best_metric_value = float('-inf')
            
            # Test each parameter combination
            for i, param_combo in enumerate(combinations_to_test):
                try:
                    # Create parameter override dict
                    param_override = dict(zip(param_names, param_combo))
                    
                    # Run backtest with these parameters
                    request = BacktestStrategyRequest(
                        strategy_id=strategy_id,
                        start_date=start_date,
                        end_date=end_date,
                        parameters_override=param_override
                    )
                    
                    response = await self.backtest_use_case.execute(request)
                    
                    if response.success and response.backtest_result:
                        result = response.backtest_result
                        metric_value = float(getattr(result, optimization_metric, 0))
                        
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
                        
                        if i % 10 == 0:
                            self.logger.info(f"Completed {i+1}/{len(combinations_to_test)} optimizations")
                    
                except Exception as e:
                    self.logger.warning(f"Optimization iteration {i+1} failed: {e}")
                    continue
            
            # Sort results by optimization metric
            results.sort(key=lambda x: x["metric_value"], reverse=True)
            
            optimization_result = {
                "strategy_id": str(strategy_id),
                "optimization_metric": optimization_metric,
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "parameter_ranges": parameter_ranges,
                "total_combinations_tested": len(results),
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
            
            self.logger.info(f"Optimization completed: {len(results)} successful tests, "
                           f"best {optimization_metric}: {best_metric_value:.4f}")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error in parameter optimization: {e}")
            return {"error": str(e)}


# Factory class for creating use case instances
class UseCaseFactory:
    """Factory for creating use case instances with proper dependencies"""
    
    def __init__(self,
                 strategy_executor: StrategyExecutorService,
                 backtest_service: BacktestService):
        self.strategy_executor = strategy_executor
        self.backtest_service = backtest_service
    
    def create_execute_strategy_use_case(self) -> ExecuteStrategyUseCase:
        """Create strategy execution use case"""
        return ExecuteStrategyUseCase(self.strategy_executor)
    
    def create_backtest_strategy_use_case(self) -> BacktestStrategyUseCase:
        """Create strategy backtesting use case"""
        return BacktestStrategyUseCase(self.backtest_service)
    
    def create_strategy_optimization_use_case(self) -> StrategyOptimizationUseCase:
        """Create strategy optimization use case"""
        backtest_use_case = self.create_backtest_strategy_use_case()
        return StrategyOptimizationUseCase(backtest_use_case)