"""
Feature Engineering Module - Features Package

This package contains the feature calculators for different types of features.
"""

from feature_engineering.application.features.technical_indicators import (
    MovingAverageCalculator, OscillatorCalculator, VolatilityCalculator,
    VolumeCalculator, TrendCalculator
)

from feature_engineering.application.features.statistical_features import (
    DescriptiveStatisticsCalculator, TimeSeriesStatisticsCalculator,
    DistributionalStatisticsCalculator, MarketRegimeCalculator, AnomalyDetectionCalculator
)

from feature_engineering.application.features.orderbook_features import (
    OrderbookFeatureCalculator
)

from feature_engineering.application.features.sentiment_features import (
    NewsSentimentCalculator, SocialMediaSentimentCalculator, 
    OnChainSentimentCalculator, CombinedSentimentCalculator
)

__all__ = [
    # Technical indicators
    'MovingAverageCalculator',
    'OscillatorCalculator',
    'VolatilityCalculator',
    'VolumeCalculator',
    'TrendCalculator',
    
    # Statistical features
    'DescriptiveStatisticsCalculator',
    'TimeSeriesStatisticsCalculator',
    'DistributionalStatisticsCalculator',
    'MarketRegimeCalculator',
    'AnomalyDetectionCalculator',
    
    # Orderbook features
    'OrderbookFeatureCalculator',
    
    # Sentiment features
    'NewsSentimentCalculator',
    'SocialMediaSentimentCalculator',
    'OnChainSentimentCalculator',
    'CombinedSentimentCalculator'
]