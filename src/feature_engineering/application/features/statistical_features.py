"""
Statistical Features Module

Contains classes and functions for calculating various statistical indicators
and measures for financial time series data.
"""
import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from statsmodels.tsa.stattools import adfuller, acf, pacf

from feature_engineering.domain.entities.feature import (
    Feature, FeatureType, FeatureCategory, FeatureScope, FeatureTimeframe, FeatureMetadata
)

logger = logging.getLogger(__name__)


class StatisticalFeatureCalculator:
    """Base class for statistical feature calculators."""
    
    def __init__(self, symbol: str, timeframe: FeatureTimeframe):
        self.symbol = symbol
        self.timeframe = timeframe
    
    def create_feature_metadata(
        self, 
        name: str,
        description: str, 
        category: FeatureCategory = FeatureCategory.STATISTICAL,
        scope: FeatureScope = FeatureScope.SINGLE_ASSET,
        is_experimental: bool = False,
        tags: List[str] = None,
        properties: Dict[str, any] = None
    ) -> FeatureMetadata:
        """Create standardized metadata for statistical features."""
        if tags is None:
            tags = []
        if properties is None:
            properties = {}
            
        # Add standard tags for statistical features
        tags.extend(["statistical", category.name.lower()])
        
        return FeatureMetadata(
            description=description,
            category=category,
            scope=scope,
            timeframe=self.timeframe,
            is_experimental=is_experimental,
            tags=tags,
            properties=properties
        )


class DescriptiveStatisticsCalculator(StatisticalFeatureCalculator):
    """Calculator for descriptive statistics features."""
    
    def calculate_rolling_mean(self, prices: pd.Series, window: int) -> Feature:
        """
        Calculate rolling mean (moving average) for statistical purposes
        
        Args:
            prices: Series of price data
            window: Rolling window size
            
        Returns:
            Feature object containing the rolling mean value
        """
        try:
            # Calculate rolling mean
            rolling_mean = prices.rolling(window=window).mean().iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"ROLLING_MEAN-{window}",
                description=f"Rolling Mean over {window} periods",
                category=FeatureCategory.STATISTICAL,
                tags=["rolling_mean", "central_tendency"],
                properties={"window": window}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"ROLLING_MEAN-{window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(rolling_mean),
                metadata=metadata,
                lookback_periods=window,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating rolling mean: {str(e)}")
            raise
    
    def calculate_rolling_std(self, prices: pd.Series, window: int) -> Feature:
        """
        Calculate rolling standard deviation
        
        Args:
            prices: Series of price data
            window: Rolling window size
            
        Returns:
            Feature object containing the rolling std value
        """
        try:
            # Calculate rolling std
            rolling_std = prices.rolling(window=window).std().iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"ROLLING_STD-{window}",
                description=f"Rolling Standard Deviation over {window} periods",
                category=FeatureCategory.VOLATILITY,
                tags=["rolling_std", "dispersion", "volatility"],
                properties={"window": window}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"ROLLING_STD-{window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(rolling_std),
                metadata=metadata,
                lookback_periods=window,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating rolling standard deviation: {str(e)}")
            raise
            
    def calculate_z_score(self, prices: pd.Series, window: int) -> Feature:
        """
        Calculate z-score (standardized value) of the latest price
        
        Args:
            prices: Series of price data
            window: Window size for mean and std calculation
            
        Returns:
            Feature object containing the z-score value
        """
        try:
            # Calculate mean and std over the window
            window_mean = prices.rolling(window=window).mean().iloc[-1]
            window_std = prices.rolling(window=window).std().iloc[-1]
            
            # Calculate z-score of the latest price
            latest_price = prices.iloc[-1]
            z_score = (latest_price - window_mean) / window_std
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"Z_SCORE-{window}",
                description=f"Z-Score of latest price relative to {window} periods",
                category=FeatureCategory.STATISTICAL,
                tags=["z_score", "standardized", "statistical"],
                properties={"window": window}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"Z_SCORE-{window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(z_score),
                metadata=metadata,
                lookback_periods=window,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating z-score: {str(e)}")
            raise
    
    def calculate_percentile(self, prices: pd.Series, window: int, percentile: float = 95.0) -> Feature:
        """
        Calculate the percentile rank of the latest price within the window
        
        Args:
            prices: Series of price data
            window: Window size for percentile calculation
            percentile: Percentile to calculate (default: 95.0)
            
        Returns:
            Feature object containing the percentile value
        """
        try:
            # Get window data
            window_data = prices.iloc[-window:].values
            
            # Calculate percentile of the window
            percentile_value = np.percentile(window_data, percentile)
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"PERCENTILE-{window}-{percentile}",
                description=f"{percentile}th Percentile over {window} periods",
                category=FeatureCategory.STATISTICAL,
                tags=["percentile", "statistical"],
                properties={"window": window, "percentile": percentile}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"PERCENTILE-{window}-{percentile}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(percentile_value),
                metadata=metadata,
                lookback_periods=window,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating percentile: {str(e)}")
            raise


class TimeSeriesStatisticsCalculator(StatisticalFeatureCalculator):
    """Calculator for time series specific statistical features."""
    
    def calculate_autocorrelation(self, prices: pd.Series, lag: int) -> Feature:
        """
        Calculate autocorrelation at specified lag
        
        Args:
            prices: Series of price data
            lag: Lag period for autocorrelation
            
        Returns:
            Feature object containing the autocorrelation value
        """
        try:
            # Calculate returns first
            returns = prices.pct_change().dropna()
            
            # Calculate autocorrelation
            autocorr = acf(returns, nlags=lag)[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"AUTOCORR-{lag}",
                description=f"Autocorrelation of returns at lag {lag}",
                category=FeatureCategory.STATISTICAL,
                tags=["autocorrelation", "time_series", "statistical"],
                properties={"lag": lag}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"AUTOCORR-{lag}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(autocorr),
                metadata=metadata,
                lookback_periods=len(returns),  # Need full series for ACF
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating autocorrelation: {str(e)}")
            raise
    
    def calculate_stationarity(self, prices: pd.Series) -> Feature:
        """
        Calculate stationarity using Augmented Dickey-Fuller test
        
        Args:
            prices: Series of price data
            
        Returns:
            Feature object containing the ADF p-value
        """
        try:
            # Calculate returns first (prices are rarely stationary)
            returns = prices.pct_change().dropna()
            
            # Perform ADF test
            adf_result = adfuller(returns)
            p_value = adf_result[1]  # p-value
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name="ADF_PVALUE",
                description="Augmented Dickey-Fuller test p-value",
                category=FeatureCategory.STATISTICAL,
                tags=["stationarity", "adf_test", "time_series"],
                properties={"adf_statistic": float(adf_result[0])}
            )
            
            # Create and return feature
            return Feature.create(
                name="ADF_PVALUE",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(p_value),
                metadata=metadata,
                lookback_periods=len(returns),  # Need full series for ADF
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating stationarity: {str(e)}")
            raise


class DistributionalStatisticsCalculator(StatisticalFeatureCalculator):
    """Calculator for distribution-related statistical features."""
    
    def calculate_skewness(self, prices: pd.Series, window: int) -> Feature:
        """
        Calculate rolling skewness of returns
        
        Args:
            prices: Series of price data
            window: Window size for skewness calculation
            
        Returns:
            Feature object containing the skewness value
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate rolling skewness
            rolling_skew = returns.rolling(window=window).skew().iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"SKEWNESS-{window}",
                description=f"Rolling Skewness of returns over {window} periods",
                category=FeatureCategory.STATISTICAL,
                tags=["skewness", "distribution", "statistical"],
                properties={"window": window}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"SKEWNESS-{window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(rolling_skew),
                metadata=metadata,
                lookback_periods=window + 1,  # Need one extra for returns calculation
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating skewness: {str(e)}")
            raise
    
    def calculate_kurtosis(self, prices: pd.Series, window: int) -> Feature:
        """
        Calculate rolling kurtosis of returns
        
        Args:
            prices: Series of price data
            window: Window size for kurtosis calculation
            
        Returns:
            Feature object containing the kurtosis value
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate rolling kurtosis
            rolling_kurt = returns.rolling(window=window).kurt().iloc[-1]
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"KURTOSIS-{window}",
                description=f"Rolling Kurtosis of returns over {window} periods",
                category=FeatureCategory.STATISTICAL,
                tags=["kurtosis", "distribution", "fat_tails", "statistical"],
                properties={"window": window}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"KURTOSIS-{window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(rolling_kurt),
                metadata=metadata,
                lookback_periods=window + 1,  # Need one extra for returns calculation
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating kurtosis: {str(e)}")
            raise
    
    def calculate_jarque_bera(self, prices: pd.Series, window: int) -> Feature:
        """
        Calculate Jarque-Bera test statistic for normality
        
        Args:
            prices: Series of price data
            window: Window size for JB calculation
            
        Returns:
            Feature object containing the JB p-value
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Get window data
            window_returns = returns.iloc[-window:]
            
            # Calculate Jarque-Bera test
            jb_stat, jb_pvalue = stats.jarque_bera(window_returns)
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"JARQUE_BERA_PVALUE-{window}",
                description=f"Jarque-Bera normality test p-value over {window} periods",
                category=FeatureCategory.STATISTICAL,
                tags=["normality", "jarque_bera", "distribution", "statistical"],
                properties={"window": window, "jb_statistic": float(jb_stat)}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"JARQUE_BERA_PVALUE-{window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(jb_pvalue),
                metadata=metadata,
                lookback_periods=window + 1,  # Need one extra for returns calculation
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating Jarque-Bera statistic: {str(e)}")
            raise


class MarketRegimeCalculator(StatisticalFeatureCalculator):
    """Calculator for market regime statistics."""
    
    def calculate_hurst_exponent(self, prices: pd.Series, max_lag: int = 20) -> Feature:
        """
        Calculate Hurst Exponent to detect mean reversion vs trend following
        
        Args:
            prices: Series of price data
            max_lag: Maximum lag for Hurst calculation
            
        Returns:
            Feature object containing the Hurst exponent value
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate Hurst exponent
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(returns.values[lag:], returns.values[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] / 2.0
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"HURST_EXPONENT-{max_lag}",
                description=f"Hurst Exponent with max lag {max_lag}",
                category=FeatureCategory.MARKET_REGIME,
                tags=["hurst", "mean_reversion", "trend_following", "regime"],
                properties={"max_lag": max_lag}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"HURST_EXPONENT-{max_lag}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(hurst),
                metadata=metadata,
                lookback_periods=len(returns),  # Need full series for Hurst
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating Hurst exponent: {str(e)}")
            raise
    
    def calculate_volatility_regime(self, prices: pd.Series, short_window: int = 20, long_window: int = 100) -> Feature:
        """
        Calculate volatility regime by comparing short and long-term volatility
        
        Args:
            prices: Series of price data
            short_window: Short-term window for volatility
            long_window: Long-term window for volatility
            
        Returns:
            Feature object containing the volatility regime ratio
        """
        try:
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate short and long-term volatility
            short_vol = returns.rolling(window=short_window).std().iloc[-1]
            long_vol = returns.rolling(window=long_window).std().iloc[-1]
            
            # Calculate ratio (>1 means increasing volatility)
            vol_ratio = short_vol / long_vol
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"VOL_REGIME-{short_window}-{long_window}",
                description=f"Volatility regime ratio ({short_window}/{long_window})",
                category=FeatureCategory.MARKET_REGIME,
                tags=["volatility", "regime", "statistical"],
                properties={"short_window": short_window, "long_window": long_window}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"VOL_REGIME-{short_window}-{long_window}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(vol_ratio),
                metadata=metadata,
                lookback_periods=long_window + 1,  # Need one extra for returns calculation
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating volatility regime: {str(e)}")
            raise


class AnomalyDetectionCalculator(StatisticalFeatureCalculator):
    """Calculator for statistical anomaly detection features."""
    
    def calculate_price_deviation(self, prices: pd.Series, window: int = 20, std_threshold: float = 2.0) -> Feature:
        """
        Calculate price deviation score based on z-score
        
        Args:
            prices: Series of price data
            window: Window size for mean and std calculation
            std_threshold: Standard deviation threshold for anomaly
            
        Returns:
            Feature object containing the deviation score
        """
        try:
            # Calculate rolling mean and std
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            
            # Calculate z-score
            z_scores = (prices - rolling_mean) / rolling_std
            latest_z = z_scores.iloc[-1]
            
            # Calculate deviation score (0 to 1, where 1 is extreme deviation)
            # Using sigmoid-like function to bound score between 0 and 1
            deviation_score = 2 / (1 + np.exp(-abs(latest_z) / std_threshold)) - 1
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"PRICE_DEVIATION-{window}-{std_threshold}",
                description=f"Price deviation score over {window} periods with {std_threshold} std threshold",
                category=FeatureCategory.ANOMALY,
                tags=["anomaly", "deviation", "z_score", "statistical"],
                properties={"window": window, "std_threshold": std_threshold, "raw_z_score": float(latest_z)}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"PRICE_DEVIATION-{window}-{std_threshold}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(deviation_score),
                metadata=metadata,
                lookback_periods=window,
                timestamp=prices.index[-1].to_pydatetime() if isinstance(prices.index[-1], pd.Timestamp) else prices.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating price deviation: {str(e)}")
            raise
    
    def calculate_volume_anomaly(self, volumes: pd.Series, window: int = 20, std_threshold: float = 2.0) -> Feature:
        """
        Calculate volume anomaly score based on z-score
        
        Args:
            volumes: Series of volume data
            window: Window size for mean and std calculation
            std_threshold: Standard deviation threshold for anomaly
            
        Returns:
            Feature object containing the anomaly score
        """
        try:
            # Calculate rolling mean and std of volumes
            rolling_mean = volumes.rolling(window=window).mean()
            rolling_std = volumes.rolling(window=window).std()
            
            # Calculate z-score
            z_scores = (volumes - rolling_mean) / rolling_std
            latest_z = z_scores.iloc[-1]
            
            # Calculate anomaly score (0 to 1, where 1 is extreme deviation)
            anomaly_score = 2 / (1 + np.exp(-abs(latest_z) / std_threshold)) - 1
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"VOLUME_ANOMALY-{window}-{std_threshold}",
                description=f"Volume anomaly score over {window} periods with {std_threshold} std threshold",
                category=FeatureCategory.ANOMALY,
                tags=["anomaly", "volume", "z_score", "statistical"],
                properties={"window": window, "std_threshold": std_threshold, "raw_z_score": float(latest_z)}
            )
            
            # Create and return feature
            return Feature.create(
                name=f"VOLUME_ANOMALY-{window}-{std_threshold}",
                symbol=self.symbol,
                type=FeatureType.STATISTICAL,
                value=float(anomaly_score),
                metadata=metadata,
                lookback_periods=window,
                timestamp=volumes.index[-1].to_pydatetime() if isinstance(volumes.index[-1], pd.Timestamp) else volumes.index[-1]
            )
        except Exception as e:
            logger.error(f"Error calculating volume anomaly: {str(e)}")
            raise