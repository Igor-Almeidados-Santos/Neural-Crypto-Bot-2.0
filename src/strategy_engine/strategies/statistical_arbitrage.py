"""
Statistical Arbitrage Strategy - Advanced pairs trading and statistical arbitrage

This strategy identifies and exploits temporary mispricings between correlated
instruments using statistical modeling, cointegration analysis, and
mean reversion techniques for market-neutral trading.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple, Set
from uuid import uuid4
import asyncio
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..domain.entities.strategy import BaseStrategy
from ..domain.entities.signal import (
    Signal, SignalType, SignalDirection, SignalStrength, 
    SignalMetrics, RiskParameters, ExecutionConstraints
)


class PairState:
    """Track state for a trading pair"""
    def __init__(self, symbol1: str, symbol2: str):
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.correlation = 0.0
        self.cointegration_pvalue = 1.0
        self.is_cointegrated = False
        self.spread_mean = 0.0
        self.spread_std = 0.0
        self.current_spread = 0.0
        self.z_score = 0.0
        self.half_life = 0.0
        self.last_signal_time = None
        self.position_status = "NONE"  # NONE, LONG_SPREAD, SHORT_SPREAD
        self.entry_spread = 0.0
        self.profit_target = 0.0
        self.stop_loss = 0.0


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Advanced statistical arbitrage strategy implementing pairs trading,
    mean reversion, and market-neutral strategies.
    
    Key Features:
    - Pairs selection using correlation and cointegration
    - Dynamic hedge ratios using Kalman filtering
    - Mean reversion with z-score entry/exit signals
    - Risk-parity position sizing
    - Multi-timeframe analysis
    - Automatic pair monitoring and replacement
    - Market-neutral portfolio construction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Statistical parameters
        self.min_correlation = self.config.get("min_correlation", 0.7)
        self.max_cointegration_pvalue = self.config.get("max_cointegration_pvalue", 0.05)
        self.lookback_period = self.config.get("lookback_period", 252)  # 1 year
        self.formation_period = self.config.get("formation_period", 126)  # 6 months
        self.trading_period = self.config.get("trading_period", 63)  # 3 months
        
        # Signal parameters
        self.entry_z_threshold = self.config.get("entry_z_threshold", 2.0)
        self.exit_z_threshold = self.config.get("exit_z_threshold", 0.5)
        self.stop_loss_z_threshold = self.config.get("stop_loss_z_threshold", 3.5)
        
        # Risk parameters
        self.max_pairs = self.config.get("max_pairs", 10)
        self.position_size_pct = self.config.get("position_size_pct", 0.05)  # 5% per pair
        self.max_sector_exposure = self.config.get("max_sector_exposure", 0.3)
        self.rebalance_frequency = self.config.get("rebalance_frequency", 5)  # days
        
        # State tracking
        self.price_data = {}
        self.return_data = {}
        self.pairs = {}
        self.active_pairs = {}
        self.universe = set()
        self.last_rebalance = None
        self.is_initialized = False
        
        # Performance tracking
        self.total_signals = 0
        self.pairs_traded = 0
        self.successful_pairs = 0
        self.current_exposures = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for statistical arbitrage strategy"""
        return {
            "min_correlation": 0.7,
            "max_cointegration_pvalue": 0.05,
            "lookback_period": 252,
            "formation_period": 126,
            "trading_period": 63,
            "entry_z_threshold": 2.0,
            "exit_z_threshold": 0.5,
            "stop_loss_z_threshold": 3.5,
            "max_pairs": 10,
            "position_size_pct": 0.05,
            "max_sector_exposure": 0.3,
            "rebalance_frequency": 5,
            "min_price_history": 300,
            "signal_cooldown": 3600,  # 1 hour
            "min_dollar_volume": 1000000,  # $1M daily volume
            "max_spread_volatility": 0.1,  # 10%
            "kalman_filtering": True,
            "dynamic_hedging": True
        }
    
    async def initialize(self, market_data: Dict[str, Any], **kwargs) -> bool:
        """
        Initialize the statistical arbitrage strategy
        
        Args:
            market_data: Initial market data
            **kwargs: Additional parameters including universe of symbols
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Statistical Arbitrage Strategy")
            
            # Set universe of tradeable symbols
            self.universe = set(kwargs.get('universe', list(market_data.keys())))
            
            # Initialize data structures
            for symbol in self.universe:
                self.price_data[symbol] = []
                self.return_data[symbol] = []
            
            self.last_rebalance = datetime.now(timezone.utc)
            self.is_initialized = True
            
            self.logger.info(f"Statistical Arbitrage Strategy initialized with {len(self.universe)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Statistical Arbitrage Strategy: {e}")
            return False
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate statistical arbitrage trading signals
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not self.is_initialized:
            return signals
        
        try:
            # Update price data
            await self._update_price_data(market_data)
            
            # Check if we need to rebalance pairs
            if self._should_rebalance():
                await self._rebalance_pairs()
            
            # Check if we have enough data
            if not self._has_sufficient_data():
                return signals
            
            # Update existing pairs and generate signals
            for pair_key, pair_state in self.active_pairs.items():
                pair_signals = await self._generate_pair_signals(pair_state, market_data)
                signals.extend(pair_signals)
            
            self.total_signals += len(signals)
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating statistical arbitrage signals: {e}")
            return []
    
    async def _update_price_data(self, market_data: Dict[str, Any]) -> None:
        """Update price and return data for all symbols"""
        try:
            for symbol in self.universe:
                if symbol in market_data:
                    price = float(market_data[symbol].get('close', market_data[symbol].get('price', 0)))
                    
                    if price > 0:
                        self.price_data[symbol].append(price)
                        
                        # Calculate returns
                        if len(self.price_data[symbol]) > 1:
                            prev_price = self.price_data[symbol][-2]
                            if prev_price > 0:
                                returns = (price - prev_price) / prev_price
                                self.return_data[symbol].append(returns)
                        
                        # Keep only necessary history
                        max_history = self.config["min_price_history"]
                        if len(self.price_data[symbol]) > max_history:
                            self.price_data[symbol] = self.price_data[symbol][-max_history:]
                        if len(self.return_data[symbol]) > max_history:
                            self.return_data[symbol] = self.return_data[symbol][-max_history:]
        
        except Exception as e:
            self.logger.error(f"Error updating price data: {e}")
    
    def _should_rebalance(self) -> bool:
        """Check if pairs should be rebalanced"""
        if not self.last_rebalance:
            return True
        
        days_since_rebalance = (datetime.now(timezone.utc) - self.last_rebalance).days
        return days_since_rebalance >= self.rebalance_frequency
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have sufficient data for analysis"""
        min_symbols_with_data = len(self.universe) // 2
        symbols_with_sufficient_data = 0
        
        for symbol in self.universe:
            if len(self.price_data.get(symbol, [])) >= self.formation_period:
                symbols_with_sufficient_data += 1
        
        return symbols_with_sufficient_data >= min_symbols_with_data
    
    async def _rebalance_pairs(self) -> None:
        """Rebalance and select new trading pairs"""
        try:
            self.logger.info("Rebalancing pairs...")
            
            # Find best pairs
            candidate_pairs = await self._find_cointegrated_pairs()
            
            # Select top pairs based on statistical significance
            selected_pairs = await self._select_best_pairs(candidate_pairs)
            
            # Update active pairs
            self.active_pairs.clear()
            for pair_info in selected_pairs:
                pair_key = f"{pair_info['symbol1']}_{pair_info['symbol2']}"
                self.active_pairs[pair_key] = self._create_pair_state(pair_info)
            
            self.last_rebalance = datetime.now(timezone.utc)
            self.logger.info(f"Rebalanced with {len(self.active_pairs)} active pairs")
            
        except Exception as e:
            self.logger.error(f"Error rebalancing pairs: {e}")
    
    async def _find_cointegrated_pairs(self) -> List[Dict[str, Any]]:
        """Find cointegrated pairs using statistical tests"""
        candidate_pairs = []
        
        try:
            symbols = [s for s in self.universe if len(self.price_data.get(s, [])) >= self.formation_period]
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    
                    # Get price data
                    prices1 = np.array(self.price_data[symbol1][-self.formation_period:])
                    prices2 = np.array(self.price_data[symbol2][-self.formation_period:])
                    
                    if len(prices1) != len(prices2) or len(prices1) < self.formation_period:
                        continue
                    
                    # Calculate correlation
                    correlation = np.corrcoef(prices1, prices2)[0, 1]
                    
                    if abs(correlation) < self.min_correlation:
                        continue
                    
                    # Test for cointegration
                    cointegration_result = self._test_cointegration(prices1, prices2)
                    
                    if cointegration_result['pvalue'] <= self.max_cointegration_pvalue:
                        # Calculate spread statistics
                        spread_stats = self._calculate_spread_statistics(
                            prices1, prices2, cointegration_result['hedge_ratio']
                        )
                        
                        candidate_pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'cointegration_pvalue': cointegration_result['pvalue'],
                            'hedge_ratio': cointegration_result['hedge_ratio'],
                            'half_life': spread_stats['half_life'],
                            'spread_mean': spread_stats['mean'],
                            'spread_std': spread_stats['std'],
                            'score': self._calculate_pair_score(correlation, cointegration_result['pvalue'], spread_stats)
                        })
            
        except Exception as e:
            self.logger.error(f"Error finding cointegrated pairs: {e}")
        
        return candidate_pairs
    
    def _test_cointegration(self, prices1: np.ndarray, prices2: np.ndarray) -> Dict[str, Any]:
        """Test for cointegration between two price series"""
        try:
            # Use Engle-Granger two-step method
            
            # Step 1: Run regression to find hedge ratio
            X = np.column_stack([np.ones(len(prices2)), prices2])
            hedge_ratio = np.linalg.lstsq(X, prices1, rcond=None)[0][1]
            
            # Step 2: Test residuals for stationarity
            spread = prices1 - hedge_ratio * prices2
            
            # Augmented Dickey-Fuller test (simplified)
            # In production, use statsmodels.tsa.stattools.adfuller
            spread_diff = np.diff(spread)
            spread_lag = spread[:-1]
            
            # Simple regression for ADF test
            if len(spread_lag) > 0 and np.std(spread_lag) > 0:
                correlation = np.corrcoef(spread_diff, spread_lag)[0, 1]
                
                # Simplified p-value calculation
                # In production, use proper ADF critical values
                if correlation < -0.1:
                    pvalue = 0.01
                elif correlation < -0.05:
                    pvalue = 0.05
                elif correlation < 0:
                    pvalue = 0.1
                else:
                    pvalue = 0.5
            else:
                pvalue = 1.0
            
            return {
                'hedge_ratio': hedge_ratio,
                'pvalue': pvalue,
                'spread': spread
            }
            
        except Exception as e:
            self.logger.error(f"Error testing cointegration: {e}")
            return {'hedge_ratio': 1.0, 'pvalue': 1.0, 'spread': np.array([])}
    
    def _calculate_spread_statistics(self, prices1: np.ndarray, prices2: np.ndarray, hedge_ratio: float) -> Dict[str, Any]:
        """Calculate spread statistics for a pair"""
        try:
            spread = prices1 - hedge_ratio * prices2
            
            # Calculate half-life of mean reversion
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)
            
            if len(spread_lag) > 0 and np.std(spread_lag) > 0:
                # Simple AR(1) regression
                beta = np.sum(spread_diff * spread_lag) / np.sum(spread_lag ** 2)
                half_life = -np.log(2) / np.log(1 + beta) if beta < 0 else np.inf
            else:
                half_life = np.inf
            
            return {
                'mean': np.mean(spread),
                'std': np.std(spread),
                'half_life': min(half_life, 252),  # Cap at 1 year
                'current': spread[-1] if len(spread) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating spread statistics: {e}")
            return {'mean': 0, 'std': 1, 'half_life': np.inf, 'current': 0}
    
    def _calculate_pair_score(self, correlation: float, pvalue: float, spread_stats: Dict[str, Any]) -> float:
        """Calculate overall score for a trading pair"""
        try:
            # Score components
            correlation_score = abs(correlation)  # Higher correlation is better
            cointegration_score = 1 - pvalue  # Lower p-value is better
            mean_reversion_score = 1 / (1 + spread_stats['half_life'] / 30)  # Faster mean reversion is better
            
            # Weighted average
            score = (
                0.4 * correlation_score +
                0.4 * cointegration_score +
                0.2 * mean_reversion_score
            )
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating pair score: {e}")
            return 0.0
    
    async def _select_best_pairs(self, candidate_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select the best pairs for trading"""
        try:
            # Sort by score
            sorted_pairs = sorted(candidate_pairs, key=lambda x: x['score'], reverse=True)
            
            # Select top pairs up to maximum
            selected_pairs = []
            used_symbols = set()
            
            for pair in sorted_pairs:
                if len(selected_pairs) >= self.max_pairs:
                    break
                
                # Avoid using the same symbol in multiple pairs
                if pair['symbol1'] not in used_symbols and pair['symbol2'] not in used_symbols:
                    selected_pairs.append(pair)
                    used_symbols.add(pair['symbol1'])
                    used_symbols.add(pair['symbol2'])
            
            return selected_pairs
            
        except Exception as e:
            self.logger.error(f"Error selecting best pairs: {e}")
            return []
    
    def _create_pair_state(self, pair_info: Dict[str, Any]) -> PairState:
        """Create a PairState object from pair information"""
        pair_state = PairState(pair_info['symbol1'], pair_info['symbol2'])
        pair_state.correlation = pair_info['correlation']
        pair_state.cointegration_pvalue = pair_info['cointegration_pvalue']
        pair_state.is_cointegrated = True
        pair_state.spread_mean = pair_info['spread_mean']
        pair_state.spread_std = pair_info['spread_std']
        pair_state.half_life = pair_info['half_life']
        
        return pair_state
    
    async def _generate_pair_signals(self, pair_state: PairState, market_data: Dict[str, Any]) -> List[Signal]:
        """Generate signals for a specific pair"""
        signals = []
        
        try:
            # Get current prices
            if (pair_state.symbol1 not in market_data or 
                pair_state.symbol2 not in market_data):
                return signals
            
            price1 = Decimal(str(market_data[pair_state.symbol1].get('close', 0)))
            price2 = Decimal(str(market_data[pair_state.symbol2].get('close', 0)))
            
            if price1 <= 0 or price2 <= 0:
                return signals
            
            # Calculate current spread and z-score
            await self._update_pair_state(pair_state, float(price1), float(price2))
            
            # Check for signal conditions
            current_time = datetime.now(timezone.utc)
            
            # Check cooldown
            if (pair_state.last_signal_time and 
                (current_time - pair_state.last_signal_time).total_seconds() < self.config["signal_cooldown"]):
                return signals
            
            # Generate entry signals
            if pair_state.position_status == "NONE":
                if pair_state.z_score >= self.entry_z_threshold:
                    # Short the spread (short symbol1, long symbol2)
                    signals.extend(await self._create_spread_signals(
                        pair_state, price1, price2, "SHORT_SPREAD", market_data
                    ))
                    
                elif pair_state.z_score <= -self.entry_z_threshold:
                    # Long the spread (long symbol1, short symbol2)
                    signals.extend(await self._create_spread_signals(
                        pair_state, price1, price2, "LONG_SPREAD", market_data
                    ))
            
            # Generate exit signals
            elif abs(pair_state.z_score) <= self.exit_z_threshold:
                signals.extend(await self._create_exit_signals(
                    pair_state, price1, price2, market_data
                ))
            
            # Generate stop-loss signals
            elif abs(pair_state.z_score) >= self.stop_loss_z_threshold:
                signals.extend(await self._create_stop_loss_signals(
                    pair_state, price1, price2, market_data
                ))
            
        except Exception as e:
            self.logger.error(f"Error generating pair signals: {e}")
        
        return signals
    
    async def _update_pair_state(self, pair_state: PairState, price1: float, price2: float) -> None:
        """Update pair state with current prices"""
        try:
            # Get recent price data for hedge ratio calculation
            if (len(self.price_data[pair_state.symbol1]) >= 30 and 
                len(self.price_data[pair_state.symbol2]) >= 30):
                
                recent_prices1 = np.array(self.price_data[pair_state.symbol1][-30:])
                recent_prices2 = np.array(self.price_data[pair_state.symbol2][-30:])
                
                # Update hedge ratio using recent data
                if self.config.get("dynamic_hedging", True):
                    hedge_ratio = self._calculate_dynamic_hedge_ratio(recent_prices1, recent_prices2)
                else:
                    # Use static hedge ratio from cointegration test
                    X = np.column_stack([np.ones(len(recent_prices2)), recent_prices2])
                    hedge_ratio = np.linalg.lstsq(X, recent_prices1, rcond=None)[0][1]
                
                # Calculate current spread
                current_spread = price1 - hedge_ratio * price2
                pair_state.current_spread = current_spread
                
                # Calculate z-score
                if pair_state.spread_std > 0:
                    pair_state.z_score = (current_spread - pair_state.spread_mean) / pair_state.spread_std
                else:
                    pair_state.z_score = 0
            
        except Exception as e:
            self.logger.error(f"Error updating pair state: {e}")
    
    def _calculate_dynamic_hedge_ratio(self, prices1: np.ndarray, prices2: np.ndarray) -> float:
        """Calculate dynamic hedge ratio using rolling regression"""
        try:
            if len(prices1) < 10 or len(prices2) < 10:
                return 1.0
            
            # Use rolling window for hedge ratio
            window = min(20, len(prices1))
            recent_prices1 = prices1[-window:]
            recent_prices2 = prices2[-window:]
            
            # Simple linear regression
            X = np.column_stack([np.ones(len(recent_prices2)), recent_prices2])
            hedge_ratio = np.linalg.lstsq(X, recent_prices1, rcond=None)[0][1]
            
            return hedge_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic hedge ratio: {e}")
            return 1.0
    
    async def _create_spread_signals(self, 
                                   pair_state: PairState, 
                                   price1: Decimal, 
                                   price2: Decimal,
                                   direction: str,
                                   market_data: Dict[str, Any]) -> List[Signal]:
        """Create signals to enter a spread position"""
        signals = []
        
        try:
            # Calculate position sizes
            position_size = self._calculate_pair_position_size(price1, price2)
            
            if direction == "LONG_SPREAD":
                # Long symbol1, short symbol2
                signal1 = await self._create_signal(
                    pair_state.symbol1, SignalDirection.BUY, position_size,
                    price1, market_data, f"Long spread: z-score={pair_state.z_score:.2f}"
                )
                signal2 = await self._create_signal(
                    pair_state.symbol2, SignalDirection.SELL, position_size,
                    price2, market_data, f"Long spread: z-score={pair_state.z_score:.2f}"
                )
                pair_state.position_status = "LONG_SPREAD"
                
            else:  # SHORT_SPREAD
                # Short symbol1, long symbol2
                signal1 = await self._create_signal(
                    pair_state.symbol1, SignalDirection.SELL, position_size,
                    price1, market_data, f"Short spread: z-score={pair_state.z_score:.2f}"
                )
                signal2 = await self._create_signal(
                    pair_state.symbol2, SignalDirection.BUY, position_size,
                    price2, market_data, f"Short spread: z-score={pair_state.z_score:.2f}"
                )
                pair_state.position_status = "SHORT_SPREAD"
            
            if signal1:
                signals.append(signal1)
            if signal2:
                signals.append(signal2)
            
            pair_state.entry_spread = pair_state.current_spread
            pair_state.last_signal_time = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"Error creating spread signals: {e}")
        
        return signals
    
    async def _create_exit_signals(self, 
                                 pair_state: PairState, 
                                 price1: Decimal, 
                                 price2: Decimal,
                                 market_data: Dict[str, Any]) -> List[Signal]:
        """Create signals to exit a spread position"""
        signals = []
        
        try:
            position_size = self._calculate_pair_position_size(price1, price2)
            
            if pair_state.position_status == "LONG_SPREAD":
                # Close long spread: sell symbol1, buy symbol2
                signal1 = await self._create_signal(
                    pair_state.symbol1, SignalDirection.SELL, position_size,
                    price1, market_data, f"Exit long spread: z-score={pair_state.z_score:.2f}"
                )
                signal2 = await self._create_signal(
                    pair_state.symbol2, SignalDirection.BUY, position_size,
                    price2, market_data, f"Exit long spread: z-score={pair_state.z_score:.2f}"
                )
                
            elif pair_state.position_status == "SHORT_SPREAD":
                # Close short spread: buy symbol1, sell symbol2
                signal1 = await self._create_signal(
                    pair_state.symbol1, SignalDirection.BUY, position_size,
                    price1, market_data, f"Exit short spread: z-score={pair_state.z_score:.2f}"
                )
                signal2 = await self._create_signal(
                    pair_state.symbol2, SignalDirection.SELL, position_size,