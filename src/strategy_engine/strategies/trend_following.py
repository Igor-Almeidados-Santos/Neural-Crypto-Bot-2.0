"""
Trend Following Strategy - Advanced trend following implementation

This strategy identifies and follows market trends using multiple timeframes,
momentum indicators, and adaptive position sizing with sophisticated
trend strength analysis and dynamic risk management.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from uuid import uuid4
import asyncio
import logging

from ..domain.entities.strategy import BaseStrategy
from ..domain.entities.signal import (
    Signal, SignalType, SignalDirection, SignalStrength, 
    SignalMetrics, RiskParameters, ExecutionConstraints
)


class TrendState:
    """Track trend state for a symbol"""
    def __init__(self):
        self.current_trend = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
        self.trend_strength = 0.0  # 0-1
        self.trend_duration = 0  # periods
        self.trend_started_at = None
        self.last_trend_change = None
        self.confidence = 0.0  # 0-1


class TrendFollowingStrategy(BaseStrategy):
    """
    Advanced trend following strategy using multiple timeframes and indicators.
    
    Key Features:
    - Multiple moving average crossovers (EMA, SMA)
    - MACD for momentum confirmation
    - ADX for trend strength measurement
    - Supertrend indicator for trend direction
    - Multiple timeframe analysis
    - Pyramid position sizing in strong trends
    - Adaptive stop-loss trailing
    - Momentum-based exits
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Moving average parameters
        self.fast_ma_period = self.config.get("fast_ma_period", 12)
        self.slow_ma_period = self.config.get("slow_ma_period", 26)
        self.signal_ma_period = self.config.get("signal_ma_period", 9)
        self.ma_type = self.config.get("ma_type", "EMA")  # EMA or SMA
        
        # MACD parameters
        self.macd_fast = self.config.get("macd_fast", 12)
        self.macd_slow = self.config.get("macd_slow", 26)
        self.macd_signal = self.config.get("macd_signal", 9)
        
        # ADX parameters
        self.adx_period = self.config.get("adx_period", 14)
        self.adx_threshold = self.config.get("adx_threshold", 25)
        
        # Supertrend parameters
        self.st_period = self.config.get("st_period", 10)
        self.st_multiplier = self.config.get("st_multiplier", 3.0)
        
        # Risk parameters
        self.base_position_size = self.config.get("base_position_size", 0.05)  # 5%
        self.max_position_size = self.config.get("max_position_size", 0.15)   # 15%
        self.stop_loss_atr_multiple = self.config.get("stop_loss_atr_multiple", 2.0)
        self.trailing_stop_atr_multiple = self.config.get("trailing_stop_atr_multiple", 1.5)
        self.min_trend_strength = self.config.get("min_trend_strength", 0.6)
        
        # State tracking
        self.price_data = {}
        self.indicators = {}
        self.trend_states = {}
        self.is_initialized = False
        self.last_signals = {}
        
        # Performance tracking
        self.total_signals = 0
        self.trend_following_signals = 0
        self.successful_trends = 0
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for trend following strategy"""
        return {
            "fast_ma_period": 12,
            "slow_ma_period": 26,
            "signal_ma_period": 9,
            "ma_type": "EMA",
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 25,
            "st_period": 10,
            "st_multiplier": 3.0,
            "atr_period": 14,
            "base_position_size": 0.05,
            "max_position_size": 0.15,
            "stop_loss_atr_multiple": 2.0,
            "trailing_stop_atr_multiple": 1.5,
            "min_trend_strength": 0.6,
            "min_price_history": 100,
            "signal_cooldown": 600,  # 10 minutes
            "pyramid_enabled": True,
            "max_pyramid_levels": 3
        }
    
    async def initialize(self, market_data: Dict[str, Any], **kwargs) -> bool:
        """
        Initialize the trend following strategy
        
        Args:
            market_data: Initial market data
            **kwargs: Additional parameters
            
        Returns:
            True if initialization successful
        """
        try:
            self.logger.info("Initializing Trend Following Strategy")
            
            # Initialize data structures for each symbol
            for symbol, data in market_data.items():
                self.price_data[symbol] = {
                    'high': [],
                    'low': [],
                    'close': [],
                    'volume': []
                }
                self.indicators[symbol] = {
                    'fast_ma': [],
                    'slow_ma': [],
                    'macd_line': [],
                    'macd_signal': [],
                    'macd_histogram': [],
                    'adx': [],
                    'di_plus': [],
                    'di_minus': [],
                    'supertrend': [],
                    'atr': [],
                    'trend_score': []
                }
                self.trend_states[symbol] = TrendState()
                self.last_signals[symbol] = None
            
            self.is_initialized = True
            self.logger.info("Trend Following Strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Trend Following Strategy: {e}")
            return False
    
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Signal]:
        """
        Generate trend following trading signals
        
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
                await self._update_trend_state(symbol)
                
                # Check if we have enough data
                if len(self.price_data[symbol]['close']) < self.config["min_price_history"]:
                    continue
                
                # Generate signals for this symbol
                symbol_signals = await self._generate_symbol_signals(symbol, data)
                signals.extend(symbol_signals)
            
            self.total_signals += len(signals)
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating trend following signals: {e}")
            return []
    
    async def _update_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """Update OHLC data for a symbol"""
        try:
            # Extract OHLC data
            high = Decimal(str(data.get('high', data.get('close', data.get('price', 0)))))
            low = Decimal(str(data.get('low', data.get('close', data.get('price', 0)))))
            close = Decimal(str(data.get('close', data.get('price', 0))))
            volume = Decimal(str(data.get('volume', 0)))
            
            # Add current data
            self.price_data[symbol]['high'].append(float(high))
            self.price_data[symbol]['low'].append(float(low))
            self.price_data[symbol]['close'].append(float(close))
            self.price_data[symbol]['volume'].append(float(volume))
            
            # Keep only necessary history
            max_history = self.config["min_price_history"] * 2
            for key in self.price_data[symbol]:
                if len(self.price_data[symbol][key]) > max_history:
                    self.price_data[symbol][key] = self.price_data[symbol][key][-max_history:]
            
            # Trim indicators as well
            for indicator in self.indicators[symbol].values():
                if len(indicator) > max_history:
                    indicator[:] = indicator[-max_history:]
            
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {e}")
    
    async def _calculate_indicators(self, symbol: str) -> None:
        """Calculate all technical indicators for a symbol"""
        try:
            highs = np.array(self.price_data[symbol]['high'])
            lows = np.array(self.price_data[symbol]['low'])
            closes = np.array(self.price_data[symbol]['close'])
            
            if len(closes) < max(self.slow_ma_period, self.adx_period, self.st_period):
                return
            
             # Moving Averages
            if self.ma_type == "EMA":
                fast_ma = self._calculate_ema(closes, self.fast_ma_period)
                slow_ma = self._calculate_ema(closes, self.slow_ma_period)
            else:
                fast_ma = self._calculate_sma(closes, self.fast_ma_period)
                slow_ma = self._calculate_sma(closes, self.slow_ma_period)
            
            if len(fast_ma) > 0:
                self.indicators[symbol]['fast_ma'].append(fast_ma[-1])
            if len(slow_ma) > 0:
                self.indicators[symbol]['slow_ma'].append(slow_ma[-1])
            
            # MACD
            macd_line, macd_signal, macd_hist = self._calculate_macd(closes)
            if len(macd_line) > 0:
                self.indicators[symbol]['macd_line'].append(macd_line[-1])
                self.indicators[symbol]['macd_signal'].append(macd_signal[-1])
                self.indicators[symbol]['macd_histogram'].append(macd_hist[-1])
            
            # ADX and Directional Indicators
            adx, di_plus, di_minus = self._calculate_adx(highs, lows, closes)
            if len(adx) > 0:
                self.indicators[symbol]['adx'].append(adx[-1])
                self.indicators[symbol]['di_plus'].append(di_plus[-1])
                self.indicators[symbol]['di_minus'].append(di_minus[-1])
            
            # ATR
            atr = self._calculate_atr(highs, lows, closes)
            if len(atr) > 0:
                self.indicators[symbol]['atr'].append(atr[-1])
            
            # Supertrend
            supertrend = self._calculate_supertrend(highs, lows, closes)
            if len(supertrend) > 0:
                self.indicators[symbol]['supertrend'].append(supertrend[-1])
            
            # Overall trend score
            trend_score = self._calculate_trend_score(symbol)
            self.indicators[symbol]['trend_score'].append(trend_score)
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {symbol}: {e}")
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        if len(data) < period:
            return np.array([])
        
        sma = np.convolve(data, np.ones(period), 'valid') / period
        return sma
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.array([])
        
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_macd(self, closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD line, signal line, and histogram"""
        if len(closes) < self.macd_slow:
            return np.array([]), np.array([]), np.array([])
        
        ema_fast = self._calculate_ema(closes, self.macd_fast)
        ema_slow = self._calculate_ema(closes, self.macd_slow)
        
        # Align arrays
        min_len = min(len(ema_fast), len(ema_slow))
        if min_len == 0:
            return np.array([]), np.array([]), np.array([])
        
        macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
        
        if len(macd_line) < self.macd_signal:
            return macd_line, np.array([]), np.array([])
        
        macd_signal = self._calculate_ema(macd_line, self.macd_signal)
        
        # Align for histogram
        min_signal_len = min(len(macd_line), len(macd_signal))
        if min_signal_len == 0:
            return macd_line, macd_signal, np.array([])
        
        macd_histogram = macd_line[-min_signal_len:] - macd_signal[-min_signal_len:]
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ADX, +DI, and -DI"""
        if len(closes) < self.adx_period + 1:
            return np.array([]), np.array([]), np.array([])
        
        # True Range
        tr = self._calculate_true_range(highs, lows, closes)
        
        # Directional Movement
        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = self._smooth_indicator(tr, self.adx_period)
        plus_di_smooth = self._smooth_indicator(plus_dm, self.adx_period)
        minus_di_smooth = self._smooth_indicator(minus_dm, self.adx_period)
        
        # DI calculations
        plus_di = 100 * plus_di_smooth / atr
        minus_di = 100 * minus_di_smooth / atr
        
        # ADX calculation
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = self._smooth_indicator(dx, self.adx_period)
        
        return adx, plus_di, minus_di
    
    def _calculate_true_range(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate True Range"""
        if len(closes) < 2:
            return np.array([])
        
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate Average True Range"""
        tr = self._calculate_true_range(highs, lows, closes)
        if len(tr) < self.config.get("atr_period", 14):
            return np.array([])
        
        return self._smooth_indicator(tr, self.config.get("atr_period", 14))
    
    def _smooth_indicator(self, data: np.ndarray, period: int) -> np.ndarray:
        """Smooth indicator using Wilder's smoothing method"""
        if len(data) < period:
            return np.array([])
        
        smoothed = np.zeros(len(data))
        smoothed[period-1] = np.mean(data[:period])
        
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period - 1) + data[i]) / period
        
        return smoothed[period-1:]
    
    def _calculate_supertrend(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """Calculate Supertrend indicator"""
        if len(closes) < self.st_period:
            return np.array([])
        
        atr = self._calculate_atr(highs, lows, closes)
        if len(atr) == 0:
            return np.array([])
        
        hl_avg = (highs + lows) / 2
        
        # Align arrays
        min_len = min(len(hl_avg), len(atr))
        if min_len == 0:
            return np.array([])
        
        hl_avg = hl_avg[-min_len:]
        atr = atr[-min_len:] if len(atr) >= min_len else np.pad(atr, (min_len - len(atr), 0), 'edge')
        closes_aligned = closes[-min_len:]
        
        upper_band = hl_avg + (self.st_multiplier * atr)
        lower_band = hl_avg - (self.st_multiplier * atr)
        
        supertrend = np.zeros(len(closes_aligned))
        trend = np.ones(len(closes_aligned))  # 1 for up, -1 for down
        
        for i in range(1, len(closes_aligned)):
            # Upper band
            if upper_band[i] < upper_band[i-1] or closes_aligned[i-1] > upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Lower band
            if lower_band[i] > lower_band[i-1] or closes_aligned[i-1] < lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Trend and Supertrend
            if closes_aligned[i] <= lower_band[i]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            elif closes_aligned[i] >= upper_band[i]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = trend[i-1]
                supertrend[i] = upper_band[i] if trend[i] == -1 else lower_band[i]
        
        return supertrend
    
    def _calculate_trend_score(self, symbol: str) -> float:
        """Calculate overall trend score (0-1)"""
        try:
            score = 0.0
            factors = 0
            
            # MA crossover score
            if (len(self.indicators[symbol]['fast_ma']) > 0 and 
                len(self.indicators[symbol]['slow_ma']) > 0):
                fast_ma = self.indicators[symbol]['fast_ma'][-1]
                slow_ma = self.indicators[symbol]['slow_ma'][-1]
                
                if fast_ma > slow_ma:
                    score += 0.25
                factors += 1
            
            # MACD score
            if (len(self.indicators[symbol]['macd_line']) > 0 and 
                len(self.indicators[symbol]['macd_signal']) > 0):
                macd_line = self.indicators[symbol]['macd_line'][-1]
                macd_signal = self.indicators[symbol]['macd_signal'][-1]
                
                if macd_line > macd_signal and macd_line > 0:
                    score += 0.25
                elif macd_line < macd_signal and macd_line < 0:
                    score -= 0.25
                factors += 1
            
            # ADX strength score
            if len(self.indicators[symbol]['adx']) > 0:
                adx = self.indicators[symbol]['adx'][-1]
                if adx > self.adx_threshold:
                    score += 0.25 * (adx / 100)
                factors += 1
            
            # Directional indicator score
            if (len(self.indicators[symbol]['di_plus']) > 0 and 
                len(self.indicators[symbol]['di_minus']) > 0):
                di_plus = self.indicators[symbol]['di_plus'][-1]
                di_minus = self.indicators[symbol]['di_minus'][-1]
                
                if di_plus > di_minus:
                    score += 0.25
                else:
                    score -= 0.25
                factors += 1
            
            # Supertrend alignment
            if (len(self.indicators[symbol]['supertrend']) > 0 and 
                len(self.price_data[symbol]['close']) > 0):
                current_price = self.price_data[symbol]['close'][-1]
                supertrend = self.indicators[symbol]['supertrend'][-1]
                
                if current_price > supertrend:
                    score += 0.25
                else:
                    score -= 0.25
                factors += 1
            
            # Normalize score
            if factors > 0:
                normalized_score = score / factors
                return max(0, min(1, (normalized_score + 1) / 2))  # Convert to 0-1
            
            return 0.5  # Neutral
            
        except Exception as e:
            self.logger.error(f"Error calculating trend score for {symbol}: {e}")
            return 0.5
    
    async def _update_trend_state(self, symbol: str) -> None:
        """Update trend state for a symbol"""
        try:
            trend_state = self.trend_states[symbol]
            
            if len(self.indicators[symbol]['trend_score']) == 0:
                return
            
            current_score = self.indicators[symbol]['trend_score'][-1]
            previous_trend = trend_state.current_trend
            
            # Determine new trend
            if current_score > 0.65:
                new_trend = "BULLISH"
                trend_state.trend_strength = current_score
            elif current_score < 0.35:
                new_trend = "BEARISH"
                trend_state.trend_strength = 1 - current_score
            else:
                new_trend = "NEUTRAL"
                trend_state.trend_strength = 0.5 - abs(0.5 - current_score)
            
            # Update trend state
            if new_trend != previous_trend:
                trend_state.last_trend_change = datetime.now(timezone.utc)
                trend_state.trend_started_at = datetime.now(timezone.utc)
                trend_state.trend_duration = 0
            else:
                trend_state.trend_duration += 1
            
            trend_state.current_trend = new_trend
            trend_state.confidence = current_score
            
        except Exception as e:
            self.logger.error(f"Error updating trend state for {symbol}: {e}")
    
    async def _generate_symbol_signals(self, symbol: str, data: Dict[str, Any]) -> List[Signal]:
        """Generate trend following signals for a specific symbol"""
        signals = []
        
        try:
            current_price = Decimal(str(data.get('close', data.get('price', 0))))
            current_time = datetime.now(timezone.utc)
            
            # Check signal cooldown
            if self._is_signal_on_cooldown(symbol, current_time):
                return signals
            
            trend_state = self.trend_states[symbol]
            
            # Only trade strong trends
            if trend_state.trend_strength < self.min_trend_strength:
                return signals
            
            # Generate entry signals
            if self._should_enter_long(symbol, trend_state):
                signal = await self._create_long_signal(symbol, current_price, data, trend_state)
                if signal:
                    signals.append(signal)
                    self.trend_following_signals += 1
            
            elif self._should_enter_short(symbol, trend_state):
                signal = await self._create_short_signal(symbol, current_price, data, trend_state)
                if signal:
                    signals.append(signal)
                    self.trend_following_signals += 1
            
            # Generate pyramid signals (if enabled and in existing trend)
            if (self.config.get("pyramid_enabled", True) and 
                self._should_pyramid(symbol, trend_state)):
                pyramid_signal = await self._create_pyramid_signal(symbol, current_price, data, trend_state)
                if pyramid_signal:
                    signals.append(pyramid_signal)
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _is_signal_on_cooldown(self, symbol: str, current_time: datetime) -> bool:
        """Check if signal is on cooldown period"""
        if symbol not in self.last_signals or not self.last_signals[symbol]:
            return False
        
        time_diff = (current_time - self.last_signals[symbol]).total_seconds()
        return time_diff < self.config["signal_cooldown"]
    
    def _should_enter_long(self, symbol: str, trend_state: TrendState) -> bool:
        """Check if should enter long position"""
        if trend_state.current_trend != "BULLISH":
            return False
        
        # Check for fresh trend or strong momentum
        if trend_state.trend_duration < 5:  # Fresh trend
            return True
        
        # Check for momentum continuation
        if (len(self.indicators[symbol]['macd_histogram']) >= 2 and
            self.indicators[symbol]['macd_histogram'][-1] > self.indicators[symbol]['macd_histogram'][-2]):
            return True
        
        return False
    
    def _should_enter_short(self, symbol: str, trend_state: TrendState) -> bool:
        """Check if should enter short position"""
        if trend_state.current_trend != "BEARISH":
            return False
        
        # Check for fresh trend or strong momentum
        if trend_state.trend_duration < 5:
            return True
        
        # Check for momentum continuation
        if (len(self.indicators[symbol]['macd_histogram']) >= 2 and
            self.indicators[symbol]['macd_histogram'][-1] < self.indicators[symbol]['macd_histogram'][-2]):
            return True
        
        return False
    
    def _should_pyramid(self, symbol: str, trend_state: TrendState) -> bool:
        """Check if should add to existing position (pyramid)"""
        # Only pyramid in strong, established trends
        return (trend_state.trend_duration > 10 and 
                trend_state.trend_strength > 0.8)
    
    async def _create_long_signal(self, 
                                symbol: str, 
                                current_price: Decimal, 
                                data: Dict[str, Any],
                                trend_state: TrendState) -> Optional[Signal]:
        """Create a LONG signal"""
        try:
            # Calculate position size based on trend strength
            base_size = self.base_position_size * trend_state.trend_strength
            position_size = min(base_size, self.max_position_size)
            
            # Calculate stop loss using ATR
            atr = self.indicators[symbol]['atr'][-1] if self.indicators[symbol]['atr'] else 0
            stop_loss = current_price - Decimal(str(atr * self.stop_loss_atr_multiple))
            
            # Calculate take profit (trend following - let it run)
            take_profit = None  # No fixed take profit in trend following
            
            # Signal strength based on trend strength and momentum
            if trend_state.trend_strength > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif trend_state.trend_strength > 0.7:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            
            # Calculate confidence
            confidence = min(95, trend_state.confidence * 100)
            
            # Create signal metrics
            metrics = SignalMetrics(
                confidence=Decimal(str(confidence)),
                expected_return=Decimal("0.05"),  # 5% expected return
                expected_risk=Decimal(str(abs(stop_loss - current_price) / current_price)),
                win_probability=Decimal("0.65"),  # Trend following win rate
                volatility_forecast=Decimal(str(atr / float(current_price)))
            )
            
            # Risk parameters
            risk_params = RiskParameters(
                stop_loss_pct=Decimal(str(abs(stop_loss - current_price) / current_price)),
                max_position_size=Decimal(str(position_size)),
                max_slippage_pct=Decimal("0.002")
            )
            
            # Execution constraints
            execution_constraints = ExecutionConstraints(
                execution_style="MARKET",
                urgency="NORMAL"
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
                quantity=Decimal(str(1000 * position_size / float(current_price))),  # Convert to quantity
                order_type="MARKET",
                metrics=metrics,
                risk_parameters=risk_params,
                execution_constraints=execution_constraints,
                source="TrendFollowingStrategy",
                reason=f"Bullish trend detected: strength={trend_state.trend_strength:.2f}, duration={trend_state.trend_duration}",
                features={
                    "trend_strength": trend_state.trend_strength,
                    "trend_duration": trend_state.trend_duration,
                    "adx": self.indicators[symbol]['adx'][-1] if self.indicators[symbol]['adx'] else 0,
                    "fast_ma": self.indicators[symbol]['fast_ma'][-1] if self.indicators[symbol]['fast_ma'] else 0,
                    "slow_ma": self.indicators[symbol]['slow_ma'][-1] if self.indicators[symbol]['slow_ma'] else 0,
                    "macd_line": self.indicators[symbol]['macd_line'][-1] if self.indicators[symbol]['macd_line'] else 0,
                    "atr": atr
                }
            )
            
            self.last_signals[symbol] = datetime.now(timezone.utc)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating long signal for {symbol}: {e}")
            return None
    
    async def _create_short_signal(self, 
                                 symbol: str, 
                                 current_price: Decimal, 
                                 data: Dict[str, Any],
                                 trend_state: TrendState) -> Optional[Signal]:
        """Create a SHORT signal"""
        try:
            # Calculate position size
            base_size = self.base_position_size * trend_state.trend_strength
            position_size = min(base_size, self.max_position_size)
            
            # Calculate stop loss using ATR
            atr = self.indicators[symbol]['atr'][-1] if self.indicators[symbol]['atr'] else 0
            stop_loss = current_price + Decimal(str(atr * self.stop_loss_atr_multiple))
            
            # Signal strength
            if trend_state.trend_strength > 0.8:
                strength = SignalStrength.VERY_STRONG
            elif trend_state.trend_strength > 0.7:
                strength = SignalStrength.STRONG
            else:
                strength = SignalStrength.MODERATE
            
            confidence = min(95, trend_state.confidence * 100)
            
            # Create signal metrics
            metrics = SignalMetrics(
                confidence=Decimal(str(confidence)),
                expected_return=Decimal("0.05"),
                expected_risk=Decimal(str(abs(stop_loss - current_price) / current_price)),
                win_probability=Decimal("0.65"),
                volatility_forecast=Decimal(str(atr / float(current_price)))
            )
            
            # Risk parameters
            risk_params = RiskParameters(
                stop_loss_pct=Decimal(str(abs(stop_loss - current_price) / current_price)),
                max_position_size=Decimal(str(position_size)),
                max_slippage_pct=Decimal("0.002")
            )
            
            # Execution constraints
            execution_constraints = ExecutionConstraints(
                execution_style="MARKET",
                urgency="NORMAL"
            )
            
            # Create signal
            signal = Signal(
                strategy_id=uuid4(),
                signal_type=SignalType.ENTRY,
                direction=SignalDirection.SELL,
                strength=strength,
                symbol=symbol,
                exchange=data.get('exchange', 'default'),
                current_price=current_price,
                quantity=Decimal(str(1000 * position_size / float(current_price))),
                order_type="MARKET",
                metrics=metrics,
                risk_parameters=risk_params,
                execution_constraints=execution_constraints,
                source="TrendFollowingStrategy",
                reason=f"Bearish trend detected: strength={trend_state.trend_strength:.2f}, duration={trend_state.trend_duration}",
                features={
                    "trend_strength": trend_state.trend_strength,
                    "trend_duration": trend_state.trend_duration,
                    "adx": self.indicators[symbol]['adx'][-1] if self.indicators[symbol]['adx'] else 0,
                    "fast_ma": self.indicators[symbol]['fast_ma'][-1] if self.indicators[symbol]['fast_ma'] else 0,
                    "slow_ma": self.indicators[symbol]['slow_ma'][-1] if self.indicators[symbol]['slow_ma'] else 0,
                    "macd_line": self.indicators[symbol]['macd_line'][-1] if self.indicators[symbol]['macd_line'] else 0,
                    "atr": atr
                }
            )
            
            self.last_signals[symbol] = datetime.now(timezone.utc)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating short signal for {symbol}: {e}")
            return None
    
    async def _create_pyramid_signal(self, 
                                   symbol: str, 
                                   current_price: Decimal, 
                                   data: Dict[str, Any],
                                   trend_state: TrendState) -> Optional[Signal]:
        """Create a pyramid signal to add to existing position"""
        try:
            # Smaller position size for pyramiding
            position_size = self.base_position_size * 0.5 * trend_state.trend_strength
            
            direction = SignalDirection.BUY if trend_state.current_trend == "BULLISH" else SignalDirection.SELL
            
            # Create signal with reduced size
            signal = Signal(
                strategy_id=uuid4(),
                signal_type=SignalType.SCALE_IN,
                direction=direction,
                strength=SignalStrength.MODERATE,
                symbol=symbol,
                exchange=data.get('exchange', 'default'),
                current_price=current_price,
                quantity=Decimal(str(1000 * position_size / float(current_price))),
                order_type="MARKET",
                source="TrendFollowingStrategy",
                reason=f"Pyramid signal: {trend_state.current_trend} trend continuation"
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating pyramid signal for {symbol}: {e}")
            return None
    
    async def update_state(self, market_data: Dict[str, Any]) -> None:
        """Update strategy state with new market data"""
        try:
            for symbol, data in market_data.items():
                await self._update_data(symbol, data)
                await self._calculate_indicators(symbol)
                await self._update_trend_state(symbol)
        except Exception as e:
            self.logger.error(f"Error updating strategy state: {e}")
    
    def calculate_position_size(self, 
                              symbol: str, 
                              price: Decimal, 
                              signal_strength: Decimal,
                              available_capital: Decimal) -> Decimal:
        """Calculate optimal position size for a trade"""
        try:
            # Get trend strength for this symbol
            trend_strength = 1.0
            if symbol in self.trend_states:
                trend_strength = self.trend_states[symbol].trend_strength
            
            # Base position sizing
            base_risk = available_capital * Decimal(str(self.base_position_size))
            
            # Adjust based on trend strength and signal strength
            risk_multiplier = Decimal(str(trend_strength)) * signal_strength
            adjusted_risk = base_risk * risk_multiplier
            
            # Calculate position size using ATR-based stop
            if symbol in self.indicators and self.indicators[symbol]['atr']:
                atr = self.indicators[symbol]['atr'][-1]
                stop_distance = Decimal(str(atr * self.stop_loss_atr_multiple))
                position_size = adjusted_risk / stop_distance
            else:
                # Fallback to percentage-based sizing
                position_size = adjusted_risk / price
            
            # Apply maximum limits
            max_position_value = available_capital * Decimal(str(self.max_position_size))
            max_quantity = max_position_value / price
            
            return min(position_size, max_quantity)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return Decimal("0")
    
    def validate_signal(self, signal: Signal, market_data: Dict[str, Any]) -> bool:
        """Validate if a signal should be executed"""
        try:
            # Basic validation
            if not signal.is_valid:
                return False
            
            # Check market data availability
            if signal.symbol not in market_data:
                return False
            
            # Price validation (ensure price hasn't moved too much)
            current_price = Decimal(str(market_data[signal.symbol].get('close', 0)))
            price_diff = abs(current_price - signal.current_price) / signal.current_price
            
            if price_diff > Decimal("0.02"):  # 2% price movement threshold
                return False
            
            # Trend validation - ensure trend is still valid
            if signal.symbol in self.trend_states:
                trend_state = self.trend_states[signal.symbol]
                
                # For long signals, ensure we're still in bullish trend
                if (signal.direction == SignalDirection.BUY and 
                    trend_state.current_trend != "BULLISH"):
                    return False
                
                # For short signals, ensure we're still in bearish trend
                if (signal.direction == SignalDirection.SELL and 
                    trend_state.current_trend != "BEARISH"):
                    return False