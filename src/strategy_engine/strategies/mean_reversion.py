"""
Mean Reversion Strategy - Advanced mean reversion implementation

This strategy identifies overbought/oversold conditions and trades the expected
reversion to mean with multiple timeframes and sophisticated risk management.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional
from uuid import uuid4
import asyncio
import logging

from ..domain.entities.strategy import BaseStrategy
from ..domain.entities.signal import (
    Signal, SignalType, SignalDirection, SignalStrength, 
    SignalMetrics, RiskParameters, ExecutionConstraints
)


class MeanReversionStrategy(BaseStrategy):
    """
    Advanced mean reversion strategy using multiple indicators and timeframes.
    
    Key Features:
    - Bollinger Bands for mean reversion signals
    - RSI for momentum confirmation
    - Volume analysis for signal strength
    - Multiple timeframe analysis
    - Dynamic position sizing
    - Adaptive stop-loss and take-profit levels
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Strategy parameters
        self.bb_period = self.config.get("bb_period", 20)
        self.bb_std_dev = self.config.get("bb_std_dev", 2.0)
        self.rsi_period = self.config.get("rsi_period", 14)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.volume_threshold = self.config.get("volume_threshold", 1.5)
        self.min_reversion_distance = self.config.get("min_reversion_distance", 0.02)  # 2%
        
        # Risk parameters
        self.max_position_size = self.config.get("max_position_size", 0.1)  # 10% of capital
        self.stop_loss_pct = self.config.get("stop_loss_pct", 0.03)  # 3%
        self.take_profit_pct = self.config.get("take_profit_pct", 0.02)  # 2%
        self.max_holding_period = self.config.get("max_holding_period", 24)  # hours
        
        # State tracking
        self.price_data = {}
        self.volume_data = {}
        self.indicators = {}
        self.is_initialized = False
        self.last_signals = {}
        
        # Performance tracking
        self.total_signals = 0
        self.successful_signals = 0
        self.current_positions = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for mean reversion strategy"""
        return {
            "bb_period": 20,
            "bb_std_dev": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_threshold": 1.5,
            "min_reversion_distance": 0.02,
            "max_position_size": 0.1,
            "stop_loss_pct": 0.03,
            "take_profit_pct": 0.02,
            "max_holding_period": 24,
            "min_price_history": 50,
            "signal_cooldown": 300  # 5 minutes between signals for same symbol
        }
    
    async def initialize(self, market_data: Dict[str, Any], **kwargs) -> bool:
        """
        Initialize the mean reversion strategy
        
        Args:
            market_data: Initial market data
            **kwargs: Additional parameters
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Mean Reversion Strategy")
            
            # Initialize data structures for each symbol
            for symbol, data in market_data.items():
                self.price_data[symbol] = []
                self.volume_data[symbol] = []
                self.indicators[symbol] = {
                    'bb_upper': [],
                    'bb_middle': [],
                    'bb_lower': [],
                    'rsi': [],
                    'volume_ma': []
                }
                self.last_signals[symbol] = None
                self.current_positions[symbol] = None
            
            self.is_initialized = True
            self.logger.info("Mean Reversion Strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Mean Reversion Strategy: {e}")
            return False
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate mean reversion trading signals
        
        Args:
            market_data: Current market data
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not self.is_initialized:
            return signals
        
        try:
            for symbol, data in market_data.items():
                if symbol not in self.price_data:
                    continue
                
                # Update data and calculate indicators
                await self._update_data(symbol, data)
                await self._calculate_indicators(symbol)
                
                # Check if we have enough data
                if len(self.price_data[symbol]) < self.config["min_price_history"]:
                    continue
                
                # Generate signals for this symbol
                symbol_signals = await self._generate_symbol_signals(symbol, data)
                signals.extend(symbol_signals)
            
            self.total_signals += len(signals)
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    async def _update_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update price and volume data for a symbol"""
        try:
            current_price = Decimal(str(data.get('close', data.get('price', 0))))
            current_volume = Decimal(str(data.get('volume', 0)))
            
            # Add current data
            self.price_data[symbol].append(float(current_price))
            self.volume_data[symbol].append(float(current_volume))
            
            # Keep only necessary history (memory management)
            max_history = self.config["min_price_history"] * 2
            if len(self.price_data[symbol]) > max_history:
                self.price_data[symbol] = self.price_data[symbol][-max_history:]
                self.volume_data[symbol] = self.volume_data[symbol][-max_history:]
                
                # Trim indicators as well
                for indicator in self.indicators[symbol].values():
                    if len(indicator) > max_history:
                        indicator[:] = indicator[-max_history:]
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {e}")
    
    async def _calculate_indicators(self, symbol: str) -> None:
        """Calculate technical indicators for a symbol"""
        try:
            prices = np.array(self.price_data[symbol])
            volumes = np.array(self.volume_data[symbol])
            
            if len(prices) < self.bb_period:
                return
            
            # Bollinger Bands
            bb_middle = self._calculate_sma(prices, self.bb_period)
            bb_std = self._calculate_rolling_std(prices, self.bb_period)
            bb_upper = bb_middle + (bb_std * self.bb_std_dev)
            bb_lower = bb_middle - (bb_std * self.bb_std_dev)
            
            # Update Bollinger Bands
            self.indicators[symbol]['bb_upper'].append(bb_upper[-1] if len(bb_upper) > 0 else 0)
            self.indicators[symbol]['bb_middle'].append(bb_middle[-1] if len(bb_middle) > 0 else 0)
            self.indicators[symbol]['bb_lower'].append(bb_lower[-1] if len(bb_lower) > 0 else 0)
            
            # RSI
            if len(prices) >= self.rsi_period:
                rsi = self._calculate_rsi(prices, self.rsi_period)
                self.indicators[symbol]['rsi'].append(rsi[-1] if len(rsi) > 0 else 50)
            else:
                self.indicators[symbol]['rsi'].append(50)
            
            # Volume Moving Average
            if len(volumes) >= 20:
                volume_ma = self._calculate_sma(volumes, 20)
                self.indicators[symbol]['volume_ma'].append(volume_ma[-1] if len(volume_ma) > 0 else 0)
            else:
                self.indicators[symbol]['volume_ma'].append(volumes[-1] if len(volumes) > 0 else 0)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return np.array([])
        
        sma = np.convolve(data, np.ones(period), 'valid') / period
        return sma
    
    def _calculate_rolling_std(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        if len(data) < period:
            return np.array([])
        
        result = []
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            result.append(np.std(window))
        
        return np.array(result)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return np.array([])
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = []
        
        for i in range(period, len(deltas)):
            # Smoothed averages (Wilder's smoothing)
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return np.array(rsi_values)
    
    async def _generate_symbol_signals(self, symbol: str, data: Dict[str, Any]) -> List[Signal]:
        """Generate signals for a specific symbol"""
        signals = []
        
        try:
            current_price = Decimal(str(data.get('close', data.get('price', 0))))
            current_time = datetime.now(timezone.utc)
            
            # Check signal cooldown
            if self._is_signal_on_cooldown(symbol, current_time):
                return signals
            
            # Get latest indicator values
            if not self._has_sufficient_indicators(symbol):
                return signals
            
            bb_upper = self.indicators[symbol]['bb_upper'][-1]
            bb_middle = self.indicators[symbol]['bb_middle'][-1]
            bb_lower = self.indicators[symbol]['bb_lower'][-1]
            rsi = self.indicators[symbol]['rsi'][-1]
            volume_ma = self.indicators[symbol]['volume_ma'][-1]
            current_volume = float(data.get('volume', 0))
            
            # Calculate signal conditions
            price_above_upper = float(current_price) > bb_upper
            price_below_lower = float(current_price) < bb_lower
            rsi_overbought = rsi > self.rsi_overbought
            rsi_oversold = rsi < self.rsi_oversold
            high_volume = current_volume > (volume_ma * self.volume_threshold)
            
            # Calculate reversion distance
            if bb_middle > 0:
                reversion_distance = abs(float(current_price) - bb_middle) / bb_middle
            else:
                reversion_distance = 0
            
            # Generate SELL signal (overbought condition)
            if (price_above_upper and rsi_overbought and 
                reversion_distance >= self.min_reversion_distance):
                
                signal = await self._create_sell_signal(
                    symbol, current_price, data, rsi, reversion_distance, high_volume
                )
                if signal:
                    signals.append(signal)
            
            # Generate BUY signal (oversold condition)
            elif (price_below_lower and rsi_oversold and 
                  reversion_distance >= self.min_reversion_distance):
                
                signal = await self._create_buy_signal(
                    symbol, current_price, data, rsi, reversion_distance, high_volume
                )
                if signal:
                    signals.append(signal)
            
            # Generate exit signals for existing positions
            exit_signal = await self._check_exit_conditions(symbol, current_price, data)
            if exit_signal:
                signals.append(exit_signal)
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _is_signal_on_cooldown(self, symbol: str, current_time: datetime) -> bool:
        """Check if signal is on cooldown period"""
        if symbol not in self.last_signals or not self.last_signals[symbol]:
            return False
        
        time_diff = (current_time - self.last_signals[symbol]).total_seconds()
        return time_diff < self.config["signal_cooldown"]
    
    def _has_sufficient_indicators(self, symbol: str) -> bool:
        """Check if we have sufficient indicator data"""
        indicators = self.indicators[symbol]
        return (len(indicators['bb_upper']) > 0 and 
                len(indicators['bb_middle']) > 0 and 
                len(indicators['bb_lower']) > 0 and 
                len(indicators['rsi']) > 0 and
                len(indicators['volume_ma']) > 0)
    
    async def _create_sell_signal(self, 
                                symbol: str, 
                                current_price: Decimal, 
                                data: Dict[str, Any],
                                rsi: float, 
                                reversion_distance: float, 
                                high_volume: bool) -> Optional[Signal]:
        """Create a SELL signal for mean reversion"""
        try:
            # Calculate signal strength
            strength = self._calculate_signal_strength(rsi, reversion_distance, high_volume, "SELL")
            
            # Calculate position size
            position_size = self._calculate_position_size(current_price, strength)
            
            if position_size <= 0:
                return None
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(rsi, reversion_distance, high_volume, "SELL")
            
            # Create signal metrics
            metrics = SignalMetrics(
                confidence=Decimal(str(confidence)),
                expected_return=Decimal(str(self.take_profit_pct)),
                expected_risk=Decimal(str(self.stop_loss_pct)),
                win_probability=Decimal(str(0.6)),  # Historical win rate for mean reversion
                risk_reward_ratio=Decimal(str(self.take_profit_pct / self.stop_loss_pct)),
                volatility_forecast=Decimal(str(reversion_distance))
            )
            
            # Risk parameters
            risk_params = RiskParameters(
                stop_loss_pct=Decimal(str(self.stop_loss_pct)),
                take_profit_pct=Decimal(str(self.take_profit_pct)),
                max_position_size=Decimal(str(position_size)),
                max_slippage_pct=Decimal("0.001")
            )
            
            # Execution constraints
            execution_constraints = ExecutionConstraints(
                execution_style="LIMIT",
                urgency="NORMAL" if high_volume else "LOW"
            )
            
            # Create signal
            signal = Signal(
                strategy_id=uuid4(),  # Will be set by the strategy executor
                signal_type=SignalType.ENTRY,
                direction=SignalDirection.SELL,
                strength=strength,
                symbol=symbol,
                exchange=data.get('exchange', 'default'),
                current_price=current_price,
                target_price=current_price * Decimal(str(1 - self.take_profit_pct)),
                quantity=Decimal(str(position_size)),
                order_type="LIMIT",
                limit_price=current_price * Decimal("0.999"),  # Slightly below market for better fill
                metrics=metrics,
                risk_parameters=risk_params,
                execution_constraints=execution_constraints,
                source="MeanReversionStrategy",
                reason=f"Overbought condition: RSI={rsi:.1f}, Price above BB upper by {reversion_distance:.2%}",
                features={
                    "rsi": rsi,
                    "bb_distance": reversion_distance,
                    "high_volume": high_volume,
                    "bb_upper": self.indicators[symbol]['bb_upper'][-1],
                    "bb_middle": self.indicators[symbol]['bb_middle'][-1],
                    "bb_lower": self.indicators[symbol]['bb_lower'][-1]
                }
            )
            
            # Update last signal time
            self.last_signals[symbol] = datetime.now(timezone.utc)
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating sell signal for {symbol}: {e}")
            return None
    
    async def _create_buy_signal(self, 
                               symbol: str, 
                               current_price: Decimal, 
                               data: Dict[str, Any],
                               rsi: float, 
                               reversion_distance: float, 
                               high_volume: bool) -> Optional[Signal]:
        """Create a BUY signal for mean reversion"""
        try:
            # Calculate signal strength
            strength = self._calculate_signal_strength(rsi, reversion_distance, high_volume, "BUY")
            
            # Calculate position size
            position_size = self._calculate_position_size(current_price, strength)
            
            if position_size <= 0:
                return None
            
            # Calculate confidence
            confidence = self._calculate_confidence(rsi, reversion_distance, high_volume, "BUY")
            
            # Create signal metrics
            metrics = SignalMetrics(
                confidence=Decimal(str(confidence)),
                expected_return=Decimal(str(self.take_profit_pct)),
                expected_risk=Decimal(str(self.stop_loss_pct)),
                win_probability=Decimal(str(0.6)),
                risk_reward_ratio=Decimal(str(self.take_profit_pct / self.stop_loss_pct)),
                volatility_forecast=Decimal(str(reversion_distance))
            )
            
            # Risk parameters
            risk_params = RiskParameters(
                stop_loss_pct=Decimal(str(self.stop_loss_pct)),
                take_profit_pct=Decimal(str(self.take_profit_pct)),
                max_position_size=Decimal(str(position_size)),
                max_slippage_pct=Decimal("0.001")
            )
            
            # Execution constraints
            execution_constraints = ExecutionConstraints(
                execution_style="LIMIT",
                urgency="NORMAL" if high_volume else "LOW"
            )
            
            # Create signal
            signal = Signal(
                strategy_id=uuid4(),
                signal_type=SignalType.ENTRY,
                direction=SignalDirection.BUY,
                strength=strength,
                symbol=symbol,
                exchange=data.get('exchange', 'default'),
                current_price=current_price,
                target_price=current_price * Decimal(str(1 + self.take_profit_pct)),
                quantity=Decimal(str(position_size)),
                order_type="LIMIT",
                limit_price=current_price * Decimal("1.001"),  # Slightly above market
                metrics=metrics,
                risk_parameters=risk_params,
                execution_constraints=execution_constraints,
                source="MeanReversionStrategy",
                reason=f"Oversold condition: RSI={rsi:.1f}, Price below BB lower by {reversion_distance:.2%}",
                features={
                    "rsi": rsi,
                    "bb_distance": reversion_distance,
                    "high_volume": high_volume,
                    "bb_upper": self.indicators[symbol]['bb_upper'][-1],
                    "bb_middle": self.indicators[symbol]['bb_middle'][-1],
                    "bb_lower": self.indicators[symbol]['bb_lower'][-1]
                }
            )
            
            self.last_signals[symbol] = datetime.now(timezone.utc)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating buy signal for {symbol}: {e}")
            return None
    
    def _calculate_signal_strength(self, 
                                 rsi: float, 
                                 reversion_distance: float, 
                                 high_volume: bool, 
                                 direction: str) -> SignalStrength:
        """Calculate signal strength based on conditions"""
        strength_score = 0
        
        # RSI contribution
        if direction == "SELL":
            if rsi > 80:
                strength_score += 3
            elif rsi > 75:
                strength_score += 2
            elif rsi > 70:
                strength_score += 1
        else:  # BUY
            if rsi < 20:
                strength_score += 3
            elif rsi < 25:
                strength_score += 2
            elif rsi < 30:
                strength_score += 1
        
        # Reversion distance contribution
        if reversion_distance > 0.05:  # 5%
            strength_score += 2
        elif reversion_distance > 0.03:  # 3%
            strength_score += 1
        
        # Volume contribution
        if high_volume:
            strength_score += 1
        
        # Map score to strength
        if strength_score >= 5:
            return SignalStrength.VERY_STRONG
        elif strength_score >= 3:
            return SignalStrength.STRONG
        elif strength_score >= 2:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def _calculate_confidence(self, 
                            rsi: float, 
                            reversion_distance: float, 
                            high_volume: bool, 
                            direction: str) -> float:
        """Calculate signal confidence (0-100)"""
        base_confidence = 50
        
        # RSI confidence adjustment
        if direction == "SELL":
            rsi_confidence = min((rsi - 50) * 2, 40)  # Max 40 points from RSI
        else:  # BUY
            rsi_confidence = min((50 - rsi) * 2, 40)
        
        # Reversion distance confidence
        distance_confidence = min(reversion_distance * 500, 30)  # Max 30 points
        
        # Volume confidence
        volume_confidence = 10 if high_volume else 0
        
        total_confidence = base_confidence + rsi_confidence + distance_confidence + volume_confidence
        return min(max(total_confidence, 10), 95)  # Clamp between 10-95
    
    def _calculate_position_size(self, current_price: Decimal, strength: SignalStrength) -> float:
        """Calculate position size based on signal strength"""
        base_size = 1000  # Base position size in quote currency
        
        # Adjust based on signal strength
        strength_multipliers = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MODERATE: 0.75,
            SignalStrength.STRONG: 1.0,
            SignalStrength.VERY_STRONG: 1.25
        }
        
        multiplier = strength_multipliers.get(strength, 0.75)
        adjusted_size = base_size * multiplier
        
        # Convert to quantity
        return adjusted_size / float(current_price)
    
    async def _check_exit_conditions(self, 
                                   symbol: str, 
                                   current_price: Decimal, 
                                   data: Dict[str, Any]) -> Optional[Signal]:
        """Check if existing positions should be exited"""
        # This would be implemented to check existing positions and generate exit signals
        # For now, returning None as position management is handled by the position entity
        return None
    
    async def update_state(self, market_data: Dict[str, Any]) -> None:
        """
        Update strategy state with new market data
        
        Args:
            market_data: New market data
        """
        try:
            # Update internal state for each symbol
            for symbol, data in market_data.items():
                await self._update_data(symbol, data)
                await self._calculate_indicators(symbol)
            
        except Exception as e:
            self.logger.error(f"Error updating strategy state: {e}")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              price: Decimal, 
                              signal_strength: Decimal,
                              available_capital: Decimal) -> Decimal:
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
        try:
            # Risk-based position sizing
            risk_per_trade = available_capital * Decimal(str(self.max_position_size))
            
            # Adjust based on signal strength
            strength_adjustment = signal_strength * Decimal("2")  # 0-2 multiplier
            adjusted_risk = risk_per_trade * strength_adjustment
            
            # Calculate position size based on stop loss
            stop_distance = price * Decimal(str(self.stop_loss_pct))
            if stop_distance > 0:
                position_size = adjusted_risk / stop_distance
            else:
                position_size = adjusted_risk / price
            
            # Ensure position size is within limits
            max_position_value = available_capital * Decimal(str(self.max_position_size))
            max_quantity = max_position_value / price
            
            return min(position_size, max_quantity)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return Decimal("0")
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """
        Validate if a signal should be executed
        
        Args:
            signal: Trading signal to validate
            market_data: Current market data
            
        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Basic validation
            if not signal.is_valid:
                return False
            
            # Check if we have current market data
            if signal.symbol not in market_data:
                return False
            
            # Price validation
            current_price = Decimal(str(market_data[signal.symbol].get('close', 0)))
            price_diff = abs(current_price - signal.current_price) / signal.current_price
            
            # Reject if price has moved too much since signal generation
            if price_diff > Decimal("0.01"):  # 1% price movement threshold
                return False
            
            # Volume validation
            current_volume = market_data[signal.symbol].get('volume', 0)
            if current_volume <= 0:
                return False
            
            # Strategy-specific validation
            if signal.symbol in self.indicators:
                # Ensure indicators still support the signal direction
                rsi = self.indicators[signal.symbol]['rsi'][-1] if self.indicators[signal.symbol]['rsi'] else 50
                
                if signal.direction == SignalDirection.SELL and rsi < 60:
                    return False
                elif signal.direction == SignalDirection.BUY and rsi > 40:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current state"""
        return {
            "name": "Mean Reversion Strategy",
            "type": "MEAN_REVERSION",
            "description": "Advanced mean reversion strategy using Bollinger Bands and RSI",
            "parameters": self.config,
            "state": {
                "is_initialized": self.is_initialized,
                "total_signals": self.total_signals,
                "successful_signals": self.successful_signals,
                "success_rate": self.successful_signals / max(self.total_signals, 1) * 100,
                "symbols_tracked": list(self.price_data.keys()),
                "last_update": datetime.now(timezone.utc).isoformat()
            },
            "indicators": {
                symbol: {
                    "bb_upper": indicators['bb_upper'][-1] if indicators['bb_upper'] else None,
                    "bb_middle": indicators['bb_middle'][-1] if indicators['bb_middle'] else None,
                    "bb_lower": indicators['bb_lower'][-1] if indicators['bb_lower'] else None,
                    "rsi": indicators['rsi'][-1] if indicators['rsi'] else None,
                    "volume_ma": indicators['volume_ma'][-1] if indicators['volume_ma'] else None
                }
                for symbol, indicators in self.indicators.items()
            }
        }
    
    def reset_state(self) -> None:
        """Reset strategy state"""
        self.price_data.clear()
        self.volume_data.clear()
        self.indicators.clear()
        self.last_signals.clear()
        self.current_positions.clear()
        self.is_initialized = False
        self.total_signals = 0
        self.successful_signals = 0
        self.logger.info("Mean Reversion Strategy state reset")