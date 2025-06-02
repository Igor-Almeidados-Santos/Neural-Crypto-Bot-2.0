"""
Technical Indicators Module

Contains classes and functions for calculating various technical indicators
used as features in the trading system.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from src.feature_engineering.domain.entities.feature import (
    Feature, FeatureType, FeatureCategory, FeatureScope, FeatureTimeframe, FeatureMetadata
)

logger = logging.getLogger(__name__)


class TechnicalIndicatorCalculator:
    """Base class for technical indicator calculators."""
    
    def __init__(self, symbol: str, timeframe: FeatureTimeframe):
        self.symbol = symbol
        self.timeframe = timeframe
    
    def create_feature_metadata(
        self, 
        name: str, 
        description: str, 
        category: FeatureCategory = FeatureCategory.TREND,
        scope: FeatureScope = FeatureScope.SINGLE_ASSET,
        is_experimental: bool = False,
        tags: List[str] = None,
        properties: Dict[str, any] = None
    ) -> FeatureMetadata:
        """Create standardized metadata for technical indicators."""
        if tags is None:
            tags = []
        if properties is None:
            properties = {}
            
        # Add standard tags for technical indicators
        tags.extend(["technical", category.name.lower()])
        
        return FeatureMetadata(
            description=description,
            category=category,
            scope=scope,
            timeframe=self.timeframe,
            is_experimental=is_experimental,
            tags=tags,
            properties=properties
        )


class MovingAverageCalculator(TechnicalIndicatorCalculator):
    """Calculator for various moving average indicators."""
    
    def calculate_sma(self, prices: pd.Series, period: int) -> Feature:
        """
        Calculate Simple Moving Average (SMA)
        
        Args:
            prices: Series of price data
            period: Number of periods for the moving average
        
        Returns:
            Feature object containing the SMA value
        """
        try:
            # Calculate SMA
            sma_value = prices.rolling(window=period).mean().iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"SMA-{period}",
                description=f"Simple Moving Average over {period} periods",
                category=FeatureCategory.TREND,
                tags=["moving_average", "sma"],
                properties={"period": period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"SMA-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(sma_value),
                metadata=metadata,
                lookback_periods=period,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            raise

    def calculate_ema(self, prices: pd.Series, period: int) -> Feature:
        """
        Calculate Exponential Moving Average (EMA)
        
        Args:
            prices: Series of price data
            period: Number of periods for the moving average
        
        Returns:
            Feature object containing the EMA value
        """
        try:
            # Calculate EMA
            ema_value = prices.ewm(span=period, adjust=False).mean().iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"EMA-{period}",
                description=f"Exponential Moving Average over {period} periods",
                category=FeatureCategory.TREND,
                tags=["moving_average", "ema"],
                properties={"period": period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"EMA-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(ema_value),
                metadata=metadata,
                lookback_periods=period,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            raise

    def calculate_wma(self, prices: pd.Series, period: int) -> Feature:
        """
        Calculate Weighted Moving Average (WMA)
        
        Args:
            prices: Series of price data
            period: Number of periods for the moving average
        
        Returns:
            Feature object containing the WMA value
        """
        try:
            # Create weights (linear, more weight to recent prices)
            weights = np.arange(1, period + 1)
            
            # Calculate WMA
            wma_value = np.sum(prices.iloc[-period:].values * weights) / weights.sum()
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"WMA-{period}",
                description=f"Weighted Moving Average over {period} periods",
                category=FeatureCategory.TREND,
                tags=["moving_average", "wma"],
                properties={"period": period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"WMA-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(wma_value),
                metadata=metadata,
                lookback_periods=period,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating WMA: {str(e)}")
            raise

    def calculate_hull_ma(self, prices: pd.Series, period: int) -> Feature:
        """
        Calculate Hull Moving Average (HMA)
        
        HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        
        Args:
            prices: Series of price data
            period: Number of periods for the moving average
        
        Returns:
            Feature object containing the HMA value
        """
        try:
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            
            # Calculate WMA with period
            weights_full = np.arange(1, period + 1)
            wma_full = np.sum(prices.iloc[-period:].values * weights_full) / weights_full.sum()
            
            # Calculate WMA with half period
            weights_half = np.arange(1, half_period + 1)
            wma_half = np.sum(prices.iloc[-half_period:].values * weights_half) / weights_half.sum()
            
            # Calculate 2*WMA(n/2) - WMA(n)
            raw_hma = 2 * wma_half - wma_full
            
            # Apply final WMA with sqrt(n) period
            # We'll simulate this with the last sqrt_period values
            raw_hma_series = pd.Series([raw_hma] * sqrt_period)
            weights_sqrt = np.arange(1, sqrt_period + 1)
            hma_value = np.sum(raw_hma_series.values * weights_sqrt) / weights_sqrt.sum()
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"HMA-{period}",
                description=f"Hull Moving Average over {period} periods",
                category=FeatureCategory.TREND,
                tags=["moving_average", "hma"],
                properties={"period": period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"HMA-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(hma_value),
                metadata=metadata,
                lookback_periods=period,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating HMA: {str(e)}")
            raise


class OscillatorCalculator(TechnicalIndicatorCalculator):
    """Calculator for oscillator-type indicators."""
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> Feature:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices: Series of price data
            period: RSI period (default: 14)
        
        Returns:
            Feature object containing the RSI value
        """
        try:
            # Calculate price differences
            deltas = prices.diff().dropna()
            
            # Separate gains and losses
            gains = deltas.copy()
            losses = deltas.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean().iloc[-1]
            avg_loss = losses.rolling(window=period).mean().iloc[-1]
            
            # Calculate RS and RSI
            if avg_loss == 0:
                rsi_value = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100.0 - (100.0 / (1.0 + rs))
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"RSI-{period}",
                description=f"Relative Strength Index over {period} periods",
                category=FeatureCategory.MOMENTUM,
                tags=["oscillator", "rsi", "overbought", "oversold"],
                properties={"period": period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"RSI-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(rsi_value),
                metadata=metadata,
                lookback_periods=period + 1,  # Need one extra period for diff
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            raise

    def calculate_stochastic(
        self, 
        high_prices: pd.Series, 
        low_prices: pd.Series, 
        close_prices: pd.Series, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> Tuple[Feature, Feature]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        
        Args:
            high_prices: Series of high price data
            low_prices: Series of low price data
            close_prices: Series of close price data
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D calculation (default: 3)
        
        Returns:
            Tuple of (Feature for %K, Feature for %D)
        """
        try:
            # Calculate %K
            lowest_low = low_prices.rolling(window=k_period).min()
            highest_high = high_prices.rolling(window=k_period).max()
            k_value = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low)).iloc[-1]
            
            # Calculate %D (SMA of %K)
            k_values = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
            d_value = k_values.rolling(window=d_period).mean().iloc[-1]
            
            # Create timestamp
            timestamp = close_prices.index[-1].to_pydatetime() if isinstance(close_prices.index[-1], pd.Timestamp) else close_prices.index[-1]
            
            # Create %K metadata
            k_metadata = self.create_feature_metadata(
                name=f"STOCH_K-{k_period}",
                description=f"Stochastic Oscillator %K over {k_period} periods",
                category=FeatureCategory.MOMENTUM,
                tags=["oscillator", "stochastic", "overbought", "oversold"],
                properties={"k_period": k_period}
            )
            
            # Create %D metadata
            d_metadata = self.create_feature_metadata(
                name=f"STOCH_D-{k_period}-{d_period}",
                description=f"Stochastic Oscillator %D over {k_period} periods with {d_period} smoothing",
                category=FeatureCategory.MOMENTUM,
                tags=["oscillator", "stochastic", "overbought", "oversold"],
                properties={"k_period": k_period, "d_period": d_period}
            )
            
            # Create and return features
            k_feature = Feature.create(
                name=f"STOCH_K-{k_period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(k_value),
                metadata=k_metadata,
                lookback_periods=k_period,
                timestamp=timestamp
            )
            
            d_feature = Feature.create(
                name=f"STOCH_D-{k_period}-{d_period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(d_value),
                metadata=d_metadata,
                lookback_periods=k_period + d_period,
                timestamp=timestamp
            )
            
            return k_feature, d_feature
        except Exception as e:
            logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
            raise

    def calculate_macd(
        self, 
        prices: pd.Series, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Tuple[Feature, Feature, Feature]:
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices: Series of price data
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line period (default: 9)
        
        Returns:
            Tuple of (MACD Line Feature, Signal Line Feature, Histogram Feature)
        """
        try:
            # Calculate EMAs
            fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Get latest values
            macd_value = macd_line.iloc[-1]
            signal_value = signal_line.iloc[-1]
            histogram_value = histogram.iloc[-1]
            
            # Create timestamp
            timestamp = prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            
            # Create MACD line metadata
            macd_metadata = self.create_feature_metadata(
                name=f"MACD-{fast_period}-{slow_period}",
                description=f"MACD Line ({fast_period},{slow_period})",
                category=FeatureCategory.MOMENTUM,
                tags=["macd", "trend_following", "momentum"],
                properties={"fast_period": fast_period, "slow_period": slow_period}
            )
            
            # Create Signal line metadata
            signal_metadata = self.create_feature_metadata(
                name=f"MACD_SIGNAL-{fast_period}-{slow_period}-{signal_period}",
                description=f"MACD Signal Line ({fast_period},{slow_period},{signal_period})",
                category=FeatureCategory.MOMENTUM,
                tags=["macd", "signal_line", "momentum"],
                properties={
                    "fast_period": fast_period, 
                    "slow_period": slow_period,
                    "signal_period": signal_period
                }
            )
            
            # Create Histogram metadata
            histogram_metadata = self.create_feature_metadata(
                name=f"MACD_HIST-{fast_period}-{slow_period}-{signal_period}",
                description=f"MACD Histogram ({fast_period},{slow_period},{signal_period})",
                category=FeatureCategory.MOMENTUM,
                tags=["macd", "histogram", "momentum"],
                properties={
                    "fast_period": fast_period, 
                    "slow_period": slow_period,
                    "signal_period": signal_period
                }
            )
            
            # Create and return features
            macd_feature = Feature.create(
                name=f"MACD-{fast_period}-{slow_period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(macd_value),
                metadata=macd_metadata,
                lookback_periods=slow_period,
                timestamp=timestamp
            )
            
            signal_feature = Feature.create(
                name=f"MACD_SIGNAL-{fast_period}-{slow_period}-{signal_period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(signal_value),
                metadata=signal_metadata,
                lookback_periods=slow_period + signal_period,
                timestamp=timestamp
            )
            
            histogram_feature = Feature.create(
                name=f"MACD_HIST-{fast_period}-{slow_period}-{signal_period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(histogram_value),
                metadata=histogram_metadata,
                lookback_periods=slow_period + signal_period,
                timestamp=timestamp
            )
            
            return macd_feature, signal_feature, histogram_feature
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            raise


class VolatilityCalculator(TechnicalIndicatorCalculator):
    """Calculator for volatility-based indicators."""
    
    def calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        num_std: float = 2.0
    ) -> Tuple[Feature, Feature, Feature]:
        """
        Calculate Bollinger Bands
        
        Args:
            prices: Series of price data
            period: Period for the moving average (default: 20)
            num_std: Number of standard deviations (default: 2.0)
        
        Returns:
            Tuple of (Upper Band Feature, Middle Band Feature, Lower Band Feature)
        """
        try:
            # Calculate middle band (SMA)
            middle_band = prices.rolling(window=period).mean()
            
            # Calculate standard deviation
            std_dev = prices.rolling(window=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_dev * num_std)
            lower_band = middle_band - (std_dev * num_std)
            
            # Get latest values
            middle_value = middle_band.iloc[-1]
            upper_value = upper_band.iloc[-1]
            lower_value = lower_band.iloc[-1]
            
            # Create timestamp
            timestamp = prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            
            # Create metadata
            bb_properties = {"period": period, "num_std": num_std}
            
            upper_metadata = self.create_feature_metadata(
                name=f"BB_UPPER-{period}-{num_std}",
                description=f"Bollinger Band Upper ({period},{num_std})",
                category=FeatureCategory.VOLATILITY,
                tags=["bollinger_bands", "volatility", "upper_band"],
                properties=bb_properties
            )
            
            middle_metadata = self.create_feature_metadata(
                name=f"BB_MIDDLE-{period}",
                description=f"Bollinger Band Middle ({period})",
                category=FeatureCategory.VOLATILITY,
                tags=["bollinger_bands", "volatility", "middle_band"],
                properties=bb_properties
            )
            
            lower_metadata = self.create_feature_metadata(
                name=f"BB_LOWER-{period}-{num_std}",
                description=f"Bollinger Band Lower ({period},{num_std})",
                category=FeatureCategory.VOLATILITY,
                tags=["bollinger_bands", "volatility", "lower_band"],
                properties=bb_properties
            )
            
            # Create and return features
            upper_feature = Feature.create(
                name=f"BB_UPPER-{period}-{num_std}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(upper_value),
                metadata=upper_metadata,
                lookback_periods=period,
                timestamp=timestamp
            )
            
            middle_feature = Feature.create(
                name=f"BB_MIDDLE-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(middle_value),
                metadata=middle_metadata,
                lookback_periods=period,
                timestamp=timestamp
            )
            
            lower_feature = Feature.create(
                name=f"BB_LOWER-{period}-{num_std}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(lower_value),
                metadata=lower_metadata,
                lookback_periods=period,
                timestamp=timestamp
            )
            
            return upper_feature, middle_feature, lower_feature
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            raise

    def calculate_atr(
        self, 
        high_prices: pd.Series, 
        low_prices: pd.Series, 
        close_prices: pd.Series, 
        period: int = 14
    ) -> Feature:
        """
        Calculate Average True Range (ATR)
        
        Args:
            high_prices: Series of high price data
            low_prices: Series of low price data
            close_prices: Series of close price data
            period: ATR period (default: 14)
        
        Returns:
            Feature object containing the ATR value
        """
        try:
            # Get previous close prices
            prev_close = close_prices.shift(1)
            
            # Calculate true range
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - prev_close)
            tr3 = abs(low_prices - prev_close)
            
            # Combine to get true range
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR (Wilder's smoothing method)
            atr = true_range.ewm(alpha=1/period, adjust=False).mean()
            
            # Get latest value
            atr_value = atr.iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"ATR-{period}",
                description=f"Average True Range over {period} periods",
                category=FeatureCategory.VOLATILITY,
                tags=["atr", "volatility", "true_range"],
                properties={"period": period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"ATR-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(atr_value),
                metadata=metadata,
                lookback_periods=period + 1,  # Need one extra period for shift
                timestamp=close_prices.index[-1].to_pydatetime() if isinstance(close_prices.index[-1], pd.Timestamp) else close_prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise


class VolumeCalculator(TechnicalIndicatorCalculator):
    """Calculator for volume-based indicators."""
    
    def calculate_obv(self, close_prices: pd.Series, volumes: pd.Series) -> Feature:
        """
        Calculate On-Balance Volume (OBV)
        
        Args:
            close_prices: Series of close price data
            volumes: Series of volume data
        
        Returns:
            Feature object containing the OBV value
        """
        try:
            # Calculate price changes
            price_changes = close_prices.diff()
            
            # Initialize OBV with first volume
            obv = pd.Series(index=volumes.index)
            obv.iloc[0] = volumes.iloc[0]
            
            # Calculate OBV
            for i in range(1, len(close_prices)):
                if price_changes.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
                elif price_changes.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            # Get latest value
            obv_value = obv.iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name="OBV",
                description="On-Balance Volume",
                category=FeatureCategory.VOLUME,
                tags=["obv", "volume", "price_volume"],
                properties={}
            )
            
            # Create and return feature
            return Feature.create(
                name="OBV",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(obv_value),
                metadata=metadata,
                lookback_periods=len(close_prices),  # OBV requires the entire history
                timestamp=close_prices.index[-1].to_pydatetime() if isinstance(close_prices.index[-1], pd.Timestamp) else close_prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            raise

    def calculate_vwap(
        self, 
        high_prices: pd.Series, 
        low_prices: pd.Series, 
        close_prices: pd.Series, 
        volumes: pd.Series,
        reset_period: Optional[str] = 'D'
    ) -> Feature:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            high_prices: Series of high price data
            low_prices: Series of low price data
            close_prices: Series of close price data
            volumes: Series of volume data
            reset_period: Pandas offset string for VWAP reset (default: 'D' for daily)
                         None for no reset (cumulative VWAP)
        
        Returns:
            Feature object containing the VWAP value
        """
        try:
            # Calculate typical price
            typical_price = (high_prices + low_prices + close_prices) / 3
            
            # Calculate VWAP
            if reset_period:
                # Reset VWAP at specified intervals
                grouper = pd.Grouper(freq=reset_period)
                tp_vol = typical_price * volumes
                
                cumulative_tpv = tp_vol.groupby(grouper).cumsum()
                cumulative_vol = volumes.groupby(grouper).cumsum()
                
                vwap = cumulative_tpv / cumulative_vol
            else:
                # Cumulative VWAP (no reset)
                cumulative_tpv = (typical_price * volumes).cumsum()
                cumulative_vol = volumes.cumsum()
                vwap = cumulative_tpv / cumulative_vol
            
            # Get latest value
            vwap_value = vwap.iloc[-1]
            
            # Create metadata
            reset_str = reset_period if reset_period else "cumulative"
            metadata = self.create_feature_metadata(
                name=f"VWAP-{reset_str}",
                description=f"Volume Weighted Average Price ({reset_str} reset)",
                category=FeatureCategory.VOLUME,
                tags=["vwap", "volume", "price_volume"],
                properties={"reset_period": reset_period}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"VWAP-{reset_str}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(vwap_value),
                metadata=metadata,
                lookback_periods=len(volumes) if not reset_period else None,
                timestamp=close_prices.index[-1].to_pydatetime() if isinstance(close_prices.index[-1], pd.Timestamp) else close_prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            raise


class TrendCalculator(TechnicalIndicatorCalculator):
    """Calculator for trend-based indicators."""
    
    def calculate_adx(
        self, 
        high_prices: pd.Series, 
        low_prices: pd.Series, 
        close_prices: pd.Series, 
        period: int = 14
    ) -> Tuple[Feature, Feature, Feature]:
        """
        Calculate Average Directional Index (ADX), +DI, and -DI
        
        Args:
            high_prices: Series of high price data
            low_prices: Series of low price data
            close_prices: Series of close price data
            period: ADX period (default: 14)
        
        Returns:
            Tuple of (ADX Feature, +DI Feature, -DI Feature)
        """
        try:
            # Calculate True Range
            prev_close = close_prices.shift(1)
            tr1 = high_prices - low_prices
            tr2 = abs(high_prices - prev_close)
            tr3 = abs(low_prices - prev_close)
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            up_move = high_prices - high_prices.shift(1)
            down_move = low_prices.shift(1) - low_prices
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_dm = pd.Series(plus_dm, index=close_prices.index)
            minus_dm = pd.Series(minus_dm, index=close_prices.index)
            
            # Calculate smoothed values using Wilder's smoothing
            tr_smoothed = tr.ewm(alpha=1/period, adjust=False).mean()
            plus_dm_smoothed = plus_dm.ewm(alpha=1/period, adjust=False).mean()
            minus_dm_smoothed = minus_dm.ewm(alpha=1/period, adjust=False).mean()
            
            # Calculate Directional Indicators
            plus_di = 100 * (plus_dm_smoothed / tr_smoothed)
            minus_di = 100 * (minus_dm_smoothed / tr_smoothed)
            
            # Calculate Directional Index
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Calculate ADX
            adx = dx.ewm(alpha=1/period, adjust=False).mean()
            
            # Get latest values
            adx_value = adx.iloc[-1]
            plus_di_value = plus_di.iloc[-1]
            minus_di_value = minus_di.iloc[-1]
            
            # Create timestamp
            timestamp = close_prices.index[-1].to_pydatetime() if isinstance(close_prices.index[-1], pd.Timestamp) else close_prices.index[-1]
            
            # Create metadata
            adx_metadata = self.create_feature_metadata(
                name=f"ADX-{period}",
                description=f"Average Directional Index over {period} periods",
                category=FeatureCategory.TREND,
                tags=["adx", "trend_strength", "directional"],
                properties={"period": period}
            )
            
            plus_di_metadata = self.create_feature_metadata(
                name=f"PLUS_DI-{period}",
                description=f"Plus Directional Indicator over {period} periods",
                category=FeatureCategory.TREND,
                tags=["di", "trend_direction", "directional"],
                properties={"period": period}
            )
            
            minus_di_metadata = self.create_feature_metadata(
                name=f"MINUS_DI-{period}",
                description=f"Minus Directional Indicator over {period} periods",
                category=FeatureCategory.TREND,
                tags=["di", "trend_direction", "directional"],
                properties={"period": period}
            )
            
            # Create and return features
            adx_feature = Feature.create(
                name=f"ADX-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(adx_value),
                metadata=adx_metadata,
                lookback_periods=period * 2,  # ADX requires multiple smoothing operations
                timestamp=timestamp
            )
            
            plus_di_feature = Feature.create(
                name=f"PLUS_DI-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(plus_di_value),
                metadata=plus_di_metadata,
                lookback_periods=period + 1,
                timestamp=timestamp
            )
            
            minus_di_feature = Feature.create(
                name=f"MINUS_DI-{period}",
                symbol=self.symbol,
                type=FeatureType.TECHNICAL,
                value=float(minus_di_value),
                metadata=minus_di_metadata,
                lookback_periods=period + 1,
                timestamp=timestamp
            )
            
            return adx_feature, plus_di_feature, minus_di_feature
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            raise

    def calculate_ichimoku(
        self, 
        high_prices: pd.Series, 
        low_prices: pd.Series, 
        close_prices: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52,
        displacement: int = 26
    ) -> Dict[str, Feature]:
        """
        Calculate Ichimoku Cloud components
        
        Args:
            high_prices: Series of high price data
            low_prices: Series of low price data
            close_prices: Series of close price data
            tenkan_period: Tenkan-sen (Conversion Line) period (default: 9)
            kijun_period: Kijun-sen (Base Line) period (default: 26)
            senkou_b_period: Senkou Span B period (default: 52)
            displacement: Displacement period for Senkou Span (default: 26)
        
        Returns:
            Dictionary of Ichimoku components as Features
        """
        try:
            # Calculate Tenkan-sen (Conversion Line)
            tenkan_high = high_prices.rolling(window=tenkan_period).max()
            tenkan_low = low_prices.rolling(window=tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Calculate Kijun-sen (Base Line)
            kijun_high = high_prices.rolling(window=kijun_period).max()
            kijun_low = low_prices.rolling(window=kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Calculate Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            
            # Calculate Senkou Span B (Leading Span B)
            senkou_high = high_prices.rolling(window=senkou_b_period).max()
            senkou_low = low_prices.rolling(window=senkou_b_period).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
            
            # Calculate Chikou Span (Lagging Span)
            chikou_span = close_prices.shift(-displacement)
            
            # Get latest values (that are not NaN due to shift)
            tenkan_value = tenkan_sen.iloc[-1]
            kijun_value = kijun_sen.iloc[-1]
            senkou_a_value = senkou_span_a.iloc[-displacement]  # Last valid value
            senkou_b_value = senkou_span_b.iloc[-displacement]  # Last valid value
            chikou_value = chikou_span.iloc[-displacement-1] if len(chikou_span) > displacement else None
            
            # Create timestamp
            timestamp = close_prices.index[-1].to_pydatetime() if isinstance(close_prices.index[-1], pd.Timestamp) else close_prices.index[-1]
            
            # Common properties
            ichimoku_props = {
                "tenkan_period": tenkan_period,
                "kijun_period": kijun_period,
                "senkou_b_period": senkou_b_period,
                "displacement": displacement
            }
            
            # Create metadata for each component
            tenkan_metadata = self.create_feature_metadata(
                name="ICHIMOKU_TENKAN",
                description=f"Ichimoku Tenkan-sen (Conversion Line) with period {tenkan_period}",
                category=FeatureCategory.TREND,
                tags=["ichimoku", "tenkan_sen", "conversion_line"],
                properties=ichimoku_props
            )
            
            kijun_metadata = self.create_feature_metadata(
                name="ICHIMOKU_KIJUN",
                description=f"Ichimoku Kijun-sen (Base Line) with period {kijun_period}",
                category=FeatureCategory.TREND,
                tags=["ichimoku", "kijun_sen", "base_line"],
                properties=ichimoku_props
            )
            
            senkou_a_metadata = self.create_feature_metadata(
                name="ICHIMOKU_SENKOU_A",
                description=f"Ichimoku Senkou Span A (Leading Span A)",
                category=FeatureCategory.TREND,
                tags=["ichimoku", "senkou_span_a", "leading_span"],
                properties=ichimoku_props
            )
            
            senkou_b_metadata = self.create_feature_metadata(
                name="ICHIMOKU_SENKOU_B",
                description=f"Ichimoku Senkou Span B (Leading Span B) with period {senkou_b_period}",
                category=FeatureCategory.TREND,
                tags=["ichimoku", "senkou_span_b", "leading_span"],
                properties=ichimoku_props
            )
            
            chikou_metadata = self.create_feature_metadata(
                name="ICHIMOKU_CHIKOU",
                description=f"Ichimoku Chikou Span (Lagging Span) with displacement {displacement}",
                category=FeatureCategory.TREND,
                tags=["ichimoku", "chikou_span", "lagging_span"],
                properties=ichimoku_props
            )
            
            # Create features
            features = {}
            
            # Only add features that have valid values
            if not np.isnan(tenkan_value):
                features["tenkan_sen"] = Feature.create(
                    name="ICHIMOKU_TENKAN",
                    symbol=self.symbol,
                    type=FeatureType.TECHNICAL,
                    value=float(tenkan_value),
                    metadata=tenkan_metadata,
                    lookback_periods=tenkan_period,
                    timestamp=timestamp
                )
                
            if not np.isnan(kijun_value):
                features["kijun_sen"] = Feature.create(
                    name="ICHIMOKU_KIJUN",
                    symbol=self.symbol,
                    type=FeatureType.TECHNICAL,
                    value=float(kijun_value),
                    metadata=kijun_metadata,
                    lookback_periods=kijun_period,
                    timestamp=timestamp
                )
                
            if not np.isnan(senkou_a_value):
                features["senkou_span_a"] = Feature.create(
                    name="ICHIMOKU_SENKOU_A",
                    symbol=self.symbol,
                    type=FeatureType.TECHNICAL,
                    value=float(senkou_a_value),
                    metadata=senkou_a_metadata,
                    lookback_periods=max(tenkan_period, kijun_period) + displacement,
                    timestamp=timestamp
                )
                
            if not np.isnan(senkou_b_value):
                features["senkou_span_b"] = Feature.create(
                    name="ICHIMOKU_SENKOU_B",
                    symbol=self.symbol,
                    type=FeatureType.TECHNICAL,
                    value=float(senkou_b_value),
                    metadata=senkou_b_metadata,
                    lookback_periods=senkou_b_period + displacement,
                    timestamp=timestamp
                )
                
            if chikou_value is not None and not np.isnan(chikou_value):
                features["chikou_span"] = Feature.create(
                    name="ICHIMOKU_CHIKOU",
                    symbol=self.symbol,
                    type=FeatureType.TECHNICAL,
                    value=float(chikou_value),
                    metadata=chikou_metadata,
                    lookback_periods=displacement,
                    timestamp=timestamp
                )
            
            return features
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            raise