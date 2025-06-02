"""
Sentiment Features Module

Contains classes and functions for calculating sentiment-based features
from various data sources including news, social media, and on-chain data.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

from src.feature_engineering.domain.entities.feature import (
    Feature, FeatureType, FeatureCategory, FeatureScope, FeatureTimeframe, FeatureMetadata
)

logger = logging.getLogger(__name__)


class SentimentFeatureCalculator:
    """Base class for sentiment feature calculators."""
    
    def __init__(self, symbol: str, timeframe: FeatureTimeframe):
        self.symbol = symbol
        self.timeframe = timeframe
    
    def create_feature_metadata(
        self, 
        name: str,
        description: str, 
        category: FeatureCategory = FeatureCategory.SENTIMENT,
        scope: FeatureScope = FeatureScope.SINGLE_ASSET,
        is_experimental: bool = False,
        tags: List[str] = None,
        properties: Dict[str, any] = None
    ) -> FeatureMetadata:
        """Create standardized metadata for sentiment features."""
        if tags is None:
            tags = []
        if properties is None:
            properties = {}
            
        # Add standard tags for sentiment features
        tags.extend(["sentiment", category.name.lower()])
        
        return FeatureMetadata(
            description=description,
            category=category,
            scope=scope,
            timeframe=self.timeframe,
            is_experimental=is_experimental,
            tags=tags,
            properties=properties
        )


class NewsSentimentCalculator(SentimentFeatureCalculator):
    """Calculator for news sentiment features."""
    
    def calculate_news_sentiment_score(
        self, 
        news_items: List[Dict[str, Any]], 
        window_hours: int = 24
    ) -> Feature:
        """
        Calculate aggregated sentiment score from news articles
        
        Args:
            news_items: List of news items with sentiment scores and timestamps
                        Each item should have at least: 
                        {'sentiment_score': float, 'timestamp': datetime, 'relevance': float}
            window_hours: Time window in hours for sentiment aggregation
            
        Returns:
            Feature object containing the aggregated sentiment score
        """
        try:
            # Filter news items within the time window
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=window_hours)
            
            recent_news = [
                item for item in news_items 
                if item.get('timestamp') >= cutoff_time
            ]
            
            if not recent_news:
                # No recent news, return neutral sentiment
                sentiment_score = 0.0
                article_count = 0
            else:
                # Calculate weighted sentiment score based on relevance
                weighted_scores = [
                    item.get('sentiment_score', 0) * item.get('relevance', 1.0)
                    for item in recent_news
                ]
                weights = [item.get('relevance', 1.0) for item in recent_news]
                
                if sum(weights) > 0:
                    sentiment_score = sum(weighted_scores) / sum(weights)
                else:
                    sentiment_score = 0.0
                
                article_count = len(recent_news)
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"NEWS_SENTIMENT-{window_hours}h",
                description=f"Aggregated news sentiment score over {window_hours} hours",
                category=FeatureCategory.SENTIMENT,
                tags=["news", "sentiment_score", "media"],
                properties={
                    "window_hours": window_hours,
                    "article_count": article_count,
                    "sources": list(set(item.get('source', 'unknown') for item in recent_news))
                }
            )
            
            # Create and return feature
            return Feature.create(
                name=f"NEWS_SENTIMENT-{window_hours}h",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(sentiment_score),
                metadata=metadata,
                lookback_periods=window_hours,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"Error calculating news sentiment score: {str(e)}")
            raise
    
    def calculate_news_volume(
        self, 
        news_items: List[Dict[str, Any]], 
        window_hours: int = 24,
        baseline_window_hours: int = 168  # 7 days
    ) -> Feature:
        """
        Calculate news volume anomaly relative to baseline
        
        Args:
            news_items: List of news items with timestamps
            window_hours: Current window in hours for volume calculation
            baseline_window_hours: Baseline window in hours for comparison
            
        Returns:
            Feature object containing the news volume anomaly score
        """
        try:
            # Calculate time windows
            current_time = datetime.utcnow()
            current_window_cutoff = current_time - timedelta(hours=window_hours)
            baseline_cutoff = current_time - timedelta(hours=baseline_window_hours)
            
            # Count news in current window
            current_window_news = [
                item for item in news_items 
                if item.get('timestamp') >= current_window_cutoff
            ]
            current_news_count = len(current_window_news)
            
            # Count news in baseline period (excluding current window)
            baseline_news = [
                item for item in news_items 
                if baseline_cutoff <= item.get('timestamp') < current_window_cutoff
            ]
            
            # Calculate baseline daily average
            baseline_days = baseline_window_hours / 24
            if baseline_days > 0:
                baseline_daily_avg = len(baseline_news) / baseline_days
            else:
                baseline_daily_avg = 0
            
            # Calculate current daily rate
            current_days = window_hours / 24
            if current_days > 0:
                current_daily_rate = current_news_count / current_days
            else:
                current_daily_rate = 0
            
            # Calculate anomaly score (percentage above/below baseline)
            if baseline_daily_avg > 0:
                anomaly_score = (current_daily_rate - baseline_daily_avg) / baseline_daily_avg
            else:
                if current_daily_rate > 0:
                    anomaly_score = 1.0  # Maximum anomaly if no baseline but news exists
                else:
                    anomaly_score = 0.0  # No anomaly if no news in either period
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"NEWS_VOLUME_ANOMALY-{window_hours}h",
                description=f"News volume anomaly relative to {baseline_window_hours}h baseline",
                category=FeatureCategory.SENTIMENT,
                tags=["news", "volume", "anomaly", "media_attention"],
                properties={
                    "window_hours": window_hours,
                    "baseline_window_hours": baseline_window_hours,
                    "current_news_count": current_news_count,
                    "baseline_news_count": len(baseline_news),
                    "current_daily_rate": float(current_daily_rate),
                    "baseline_daily_avg": float(baseline_daily_avg)
                }
            )
            
            # Create and return feature
            return Feature.create(
                name=f"NEWS_VOLUME_ANOMALY-{window_hours}h",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(anomaly_score),
                metadata=metadata,
                lookback_periods=baseline_window_hours,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"Error calculating news volume anomaly: {str(e)}")
            raise


class SocialMediaSentimentCalculator(SentimentFeatureCalculator):
    """Calculator for social media sentiment features."""
    
    def calculate_social_sentiment_score(
        self, 
        social_posts: List[Dict[str, Any]], 
        window_hours: int = 24,
        min_engagement: int = 0
    ) -> Feature:
        """
        Calculate aggregated sentiment score from social media posts
        
        Args:
            social_posts: List of social media posts with sentiment scores and timestamps
                          Each post should have at least: 
                          {'sentiment_score': float, 'timestamp': datetime, 
                           'engagement': int, 'platform': str}
            window_hours: Time window in hours for sentiment aggregation
            min_engagement: Minimum engagement threshold to include post
            
        Returns:
            Feature object containing the aggregated social sentiment score
        """
        try:
            # Filter posts within the time window and above engagement threshold
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=window_hours)
            
            recent_posts = [
                post for post in social_posts 
                if post.get('timestamp') >= cutoff_time and 
                post.get('engagement', 0) >= min_engagement
            ]
            
            if not recent_posts:
                # No recent posts meeting criteria, return neutral sentiment
                sentiment_score = 0.0
                post_count = 0
                platforms = []
                total_engagement = 0
            else:
                # Calculate weighted sentiment score based on engagement
                weighted_scores = [
                    post.get('sentiment_score', 0) * post.get('engagement', 1)
                    for post in recent_posts
                ]
                weights = [post.get('engagement', 1) for post in recent_posts]
                
                if sum(weights) > 0:
                    sentiment_score = sum(weighted_scores) / sum(weights)
                else:
                    sentiment_score = 0.0
                
                post_count = len(recent_posts)
                platforms = list(set(post.get('platform', 'unknown') for post in recent_posts))
                total_engagement = sum(weights)
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"SOCIAL_SENTIMENT-{window_hours}h",
                description=f"Aggregated social media sentiment over {window_hours} hours",
                category=FeatureCategory.SENTIMENT,
                tags=["social_media", "sentiment_score", "social"],
                properties={
                    "window_hours": window_hours,
                    "min_engagement": min_engagement,
                    "post_count": post_count,
                    "platforms": platforms,
                    "total_engagement": total_engagement
                }
            )
            
            # Create and return feature
            return Feature.create(
                name=f"SOCIAL_SENTIMENT-{window_hours}h",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(sentiment_score),
                metadata=metadata,
                lookback_periods=window_hours,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"Error calculating social sentiment score: {str(e)}")
            raise
    
    def calculate_social_volume(
        self, 
        social_posts: List[Dict[str, Any]], 
        window_hours: int = 24,
        baseline_window_hours: int = 168,  # 7 days
        min_engagement: int = 0
    ) -> Feature:
        """
        Calculate social media volume anomaly relative to baseline
        
        Args:
            social_posts: List of social media posts with timestamps and engagement
            window_hours: Current window in hours for volume calculation
            baseline_window_hours: Baseline window in hours for comparison
            min_engagement: Minimum engagement threshold to include post
            
        Returns:
            Feature object containing the social volume anomaly score
        """
        try:
            # Calculate time windows
            current_time = datetime.utcnow()
            current_window_cutoff = current_time - timedelta(hours=window_hours)
            baseline_cutoff = current_time - timedelta(hours=baseline_window_hours)
            
            # Filter posts by engagement threshold
            filtered_posts = [
                post for post in social_posts 
                if post.get('engagement', 0) >= min_engagement
            ]
            
            # Count posts in current window
            current_window_posts = [
                post for post in filtered_posts 
                if post.get('timestamp') >= current_window_cutoff
            ]
            current_post_count = len(current_window_posts)
            current_engagement = sum(post.get('engagement', 1) for post in current_window_posts)
            
            # Count posts in baseline period (excluding current window)
            baseline_posts = [
                post for post in filtered_posts 
                if baseline_cutoff <= post.get('timestamp') < current_window_cutoff
            ]
            baseline_engagement = sum(post.get('engagement', 1) for post in baseline_posts)
            
            # Calculate baseline daily average
            baseline_days = baseline_window_hours / 24
            if baseline_days > 0:
                baseline_daily_post_avg = len(baseline_posts) / baseline_days
                baseline_daily_engagement_avg = baseline_engagement / baseline_days
            else:
                baseline_daily_post_avg = 0
                baseline_daily_engagement_avg = 0
            
            # Calculate current daily rate
            current_days = window_hours / 24
            if current_days > 0:
                current_daily_post_rate = current_post_count / current_days
                current_daily_engagement_rate = current_engagement / current_days
            else:
                current_daily_post_rate = 0
                current_daily_engagement_rate = 0
            
            # Calculate post count anomaly score
            if baseline_daily_post_avg > 0:
                post_anomaly_score = (current_daily_post_rate - baseline_daily_post_avg) / baseline_daily_post_avg
            else:
                if current_daily_post_rate > 0:
                    post_anomaly_score = 1.0
                else:
                    post_anomaly_score = 0.0
            
            # Calculate engagement anomaly score
            if baseline_daily_engagement_avg > 0:
                engagement_anomaly_score = (current_daily_engagement_rate - baseline_daily_engagement_avg) / baseline_daily_engagement_avg
            else:
                if current_daily_engagement_rate > 0:
                    engagement_anomaly_score = 1.0
                else:
                    engagement_anomaly_score = 0.0
            
            # Combined anomaly score (weighted more towards engagement)
            combined_anomaly_score = 0.3 * post_anomaly_score + 0.7 * engagement_anomaly_score
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"SOCIAL_VOLUME_ANOMALY-{window_hours}h",
                description=f"Social media volume anomaly relative to {baseline_window_hours}h baseline",
                category=FeatureCategory.SENTIMENT,
                tags=["social_media", "volume", "anomaly", "social_attention"],
                properties={
                    "window_hours": window_hours,
                    "baseline_window_hours": baseline_window_hours,
                    "min_engagement": min_engagement,
                    "current_post_count": current_post_count,
                    "current_engagement": current_engagement,
                    "baseline_post_count": len(baseline_posts),
                    "baseline_engagement": baseline_engagement,
                    "post_anomaly_score": float(post_anomaly_score),
                    "engagement_anomaly_score": float(engagement_anomaly_score)
                }
            )
            
            # Create and return feature
            return Feature.create(
                name=f"SOCIAL_VOLUME_ANOMALY-{window_hours}h",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(combined_anomaly_score),
                metadata=metadata,
                lookback_periods=baseline_window_hours,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"Error calculating social volume anomaly: {str(e)}")
            raise
    
    def calculate_social_sentiment_momentum(
        self, 
        social_posts: List[Dict[str, Any]], 
        short_window_hours: int = 4,
        long_window_hours: int = 24,
        min_engagement: int = 0
    ) -> Feature:
        """
        Calculate social sentiment momentum (short-term vs long-term sentiment)
        
        Args:
            social_posts: List of social media posts with sentiment scores and timestamps
            short_window_hours: Short-term window in hours
            long_window_hours: Long-term window in hours
            min_engagement: Minimum engagement threshold to include post
            
        Returns:
            Feature object containing the sentiment momentum score
        """
        try:
            # Calculate time windows
            current_time = datetime.utcnow()
            short_window_cutoff = current_time - timedelta(hours=short_window_hours)
            long_window_cutoff = current_time - timedelta(hours=long_window_hours)
            
            # Filter posts by engagement threshold
            filtered_posts = [
                post for post in social_posts 
                if post.get('engagement', 0) >= min_engagement
            ]
            
            # Get posts for short-term window
            short_window_posts = [
                post for post in filtered_posts 
                if post.get('timestamp') >= short_window_cutoff
            ]
            
            # Get posts for long-term window (excluding short-term)
            long_window_posts = [
                post for post in filtered_posts 
                if long_window_cutoff <= post.get('timestamp') < short_window_cutoff
            ]
            
            # Calculate short-term sentiment
            if short_window_posts:
                short_weighted_scores = [
                    post.get('sentiment_score', 0) * post.get('engagement', 1)
                    for post in short_window_posts
                ]
                short_weights = [post.get('engagement', 1) for post in short_window_posts]
                
                if sum(short_weights) > 0:
                    short_sentiment = sum(short_weighted_scores) / sum(short_weights)
                else:
                    short_sentiment = 0.0
            else:
                short_sentiment = 0.0
            
            # Calculate long-term sentiment
            if long_window_posts:
                long_weighted_scores = [
                    post.get('sentiment_score', 0) * post.get('engagement', 1)
                    for post in long_window_posts
                ]
                long_weights = [post.get('engagement', 1) for post in long_window_posts]
                
                if sum(long_weights) > 0:
                    long_sentiment = sum(long_weighted_scores) / sum(long_weights)
                else:
                    long_sentiment = 0.0
            else:
                long_sentiment = 0.0
            
            # Calculate momentum score (difference between short and long term)
            momentum_score = short_sentiment - long_sentiment
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"SOCIAL_SENTIMENT_MOMENTUM-{short_window_hours}h-{long_window_hours}h",
                description=f"Social sentiment momentum comparing {short_window_hours}h to {long_window_hours}h",
                category=FeatureCategory.SENTIMENT,
                tags=["social_media", "sentiment_momentum", "trend"],
                properties={
                    "short_window_hours": short_window_hours,
                    "long_window_hours": long_window_hours,
                    "min_engagement": min_engagement,
                    "short_post_count": len(short_window_posts),
                    "long_post_count": len(long_window_posts),
                    "short_sentiment": float(short_sentiment),
                    "long_sentiment": float(long_sentiment)
                }
            )
            
            # Create and return feature
            return Feature.create(
                name=f"SOCIAL_SENTIMENT_MOMENTUM-{short_window_hours}h-{long_window_hours}h",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(momentum_score),
                metadata=metadata,
                lookback_periods=long_window_hours,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"Error calculating social sentiment momentum: {str(e)}")
            raise


class OnChainSentimentCalculator(SentimentFeatureCalculator):
    """Calculator for on-chain sentiment features."""
    
    def calculate_hodl_waves(
        self, 
        utxo_data: List[Dict[str, Any]],
        age_brackets: List[int] = [1, 7, 30, 90, 180, 365, 730]
    ) -> Dict[str, Feature]:
        """
        Calculate HODL waves (coin age distribution) for Bitcoin-like chains
        
        Args:
            utxo_data: List of UTXO age data
                       Each item should have: {'value': float, 'age_days': int}
            age_brackets: List of age thresholds in days
            
        Returns:
            Dictionary of Features for each age bracket
        """
        try:
            # Current timestamp
            timestamp = datetime.utcnow()
            
            # Add infinity as the final bracket
            brackets = age_brackets + [float('inf')]
            
            # Calculate total value
            total_value = sum(utxo.get('value', 0) for utxo in utxo_data)
            
            features = {}
            
            if total_value == 0:
                # Return zero features if no UTXOs
                for i in range(len(brackets) - 1):
                    start_age = brackets[i]
                    end_age = brackets[i+1]
                    
                    # Feature name and description
                    if end_age == float('inf'):
                        name = f"HODL_WAVE_{start_age}d_PLUS"
                        description = f"Percentage of coins held {start_age}+ days"
                    else:
                        name = f"HODL_WAVE_{start_age}d_TO_{end_age}d"
                        description = f"Percentage of coins held {start_age}-{end_age} days"
                    
                    # Create metadata
                    metadata = self.create_feature_metadata(
                        name=name,
                        description=description,
                        category=FeatureCategory.SENTIMENT,
                        tags=["hodl_waves", "coin_age", "on_chain", "holder_behavior"],
                        properties={
                            "start_age_days": start_age,
                            "end_age_days": end_age if end_age != float('inf') else None
                        }
                    )
                    
                    # Create feature with zero value
                    features[name] = Feature.create(
                        name=name,
                        symbol=self.symbol,
                        type=FeatureType.SENTIMENT,
                        value=0.0,
                        metadata=metadata,
                        lookback_periods=int(brackets[-2]) if brackets[-2] != float('inf') else 730,
                        timestamp=timestamp
                    )
                
                return features
            
            # Calculate value in each bracket
            for i in range(len(brackets) - 1):
                start_age = brackets[i]
                end_age = brackets[i+1]
                
                # Sum value in current bracket
                bracket_value = sum(
                    utxo.get('value', 0) for utxo in utxo_data
                    if start_age <= utxo.get('age_days', 0) < end_age
                )
                
                # Calculate percentage
                bracket_percentage = (bracket_value / total_value) * 100
                
                # Feature name and description
                if end_age == float('inf'):
                    name = f"HODL_WAVE_{start_age}d_PLUS"
                    description = f"Percentage of coins held {start_age}+ days"
                else:
                    name = f"HODL_WAVE_{start_age}d_TO_{end_age}d"
                    description = f"Percentage of coins held {start_age}-{end_age} days"
                
                # Create metadata
                metadata = self.create_feature_metadata(
                    name=name,
                    description=description,
                    category=FeatureCategory.SENTIMENT,
                    tags=["hodl_waves", "coin_age", "on_chain", "holder_behavior"],
                    properties={
                        "start_age_days": start_age,
                        "end_age_days": end_age if end_age != float('inf') else None,
                        "bracket_value": float(bracket_value)
                    }
                )
                
                # Create feature
                features[name] = Feature.create(
                    name=name,
                    symbol=self.symbol,
                    type=FeatureType.SENTIMENT,
                    value=float(bracket_percentage),
                    metadata=metadata,
                    lookback_periods=int(brackets[-2]) if brackets[-2] != float('inf') else 730,
                    timestamp=timestamp
                )
            
            return features
        except Exception as e:
            logger.error(f"Error calculating HODL waves: {str(e)}")
            raise
    
    def calculate_exchange_flow_balance(
        self, 
        exchange_flows: List[Dict[str, Any]],
        window_hours: int = 24
    ) -> Feature:
        """
        Calculate net exchange flow balance (inflow - outflow)
        
        Args:
            exchange_flows: List of exchange flow data
                           Each item should have: 
                           {'timestamp': datetime, 'amount': float, 'direction': 'in'|'out'}
            window_hours: Time window in hours
            
        Returns:
            Feature containing the net exchange flow balance
        """
        try:
            # Current timestamp
            current_time = datetime.utcnow()
            cutoff_time = current_time - timedelta(hours=window_hours)
            
            # Filter flows within time window
            recent_flows = [
                flow for flow in exchange_flows
                if flow.get('timestamp') >= cutoff_time
            ]
            
            # Calculate inflows and outflows
            inflows = sum(
                flow.get('amount', 0) for flow in recent_flows
                if flow.get('direction') == 'in'
            )
            
            outflows = sum(
                flow.get('amount', 0) for flow in recent_flows
                if flow.get('direction') == 'out'
            )
            
            # Calculate net flow (positive means more inflow)
            net_flow = inflows - outflows
            
            # Calculate flow ratio
            if outflows == 0:
                flow_ratio = float('inf') if inflows > 0 else 0
            else:
                flow_ratio = inflows / outflows
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name=f"EXCHANGE_FLOW_BALANCE-{window_hours}h",
                description=f"Net exchange flow balance over {window_hours} hours",
                category=FeatureCategory.SENTIMENT,
                tags=["exchange_flows", "on_chain", "whale_activity"],
                properties={
                    "window_hours": window_hours,
                    "inflows": float(inflows),
                    "outflows": float(outflows),
                    "flow_ratio": float(flow_ratio) if not np.isinf(flow_ratio) else None
                }
            )
            
            # Create feature
            return Feature.create(
                name=f"EXCHANGE_FLOW_BALANCE-{window_hours}h",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(net_flow),
                metadata=metadata,
                lookback_periods=window_hours,
                timestamp=current_time
            )
        except Exception as e:
            logger.error(f"Error calculating exchange flow balance: {str(e)}")
            raise
    
    def calculate_stablecoin_supply_ratio(
        self, 
        market_cap: float,
        stablecoin_supply: float
    ) -> Feature:
        """
        Calculate stablecoin supply ratio (indicator of buying power)
        
        Args:
            market_cap: Current market cap of the asset
            stablecoin_supply: Total stablecoin supply in circulation
            
        Returns:
            Feature containing the stablecoin supply ratio
        """
        try:
            # Current timestamp
            timestamp = datetime.utcnow()
            
            # Calculate ratio
            if market_cap == 0:
                ratio = 0.0
            else:
                ratio = stablecoin_supply / market_cap
            
            # Create metadata
            metadata = self.create_feature_metadata(
                name="STABLECOIN_SUPPLY_RATIO",
                description="Ratio of stablecoin supply to market cap",
                category=FeatureCategory.SENTIMENT,
                tags=["stablecoins", "buying_power", "market_structure"],
                properties={
                    "market_cap": float(market_cap),
                    "stablecoin_supply": float(stablecoin_supply)
                }
            )
            
            # Create feature
            return Feature.create(
                name="STABLECOIN_SUPPLY_RATIO",
                symbol=self.symbol,
                type=FeatureType.SENTIMENT,
                value=float(ratio),
                metadata=metadata,
                lookback_periods=0,
                timestamp=timestamp
            )
        except Exception as e:
            logger.error(f"Error calculating stablecoin supply ratio: {str(e)}")
            raise


class CombinedSentimentCalculator(SentimentFeatureCalculator):
    """Calculator for combined sentiment features."""
    
    def calculate_sentiment_index(
        self, 
        sentiment_features: List[Feature], 
        weights: Dict[str, float] = None
    ) -> Feature:
        """
        Calculate a combined sentiment index from multiple sentiment features
        
        Args:
            sentiment_features: List of sentiment Feature objects
            weights: Optional dictionary of feature name to weight mappings
            
        Returns:
            Feature containing the combined sentiment index
        """
        try:
            # Current timestamp
            timestamp = datetime.utcnow()
            
            # Get feature names
            feature_names = [feature.name for feature in sentiment_features]
            
            # Apply default weights if not provided
            if weights is None:
                # Equal weights by default
                weights = {name: 1.0/len(feature_names) for name in feature_names}
            else:
                # Normalize weights to sum to 1
                weight_sum = sum(weights.values())
                if weight_sum > 0:
                    weights = {k: v/weight_sum for k, v in weights.items()}
            
            # Calculate weighted sentiment
            weighted_sum = 0.0
            applied_weights_sum = 0.0
            
            for feature in sentiment_features:
                if feature.name in weights:
                    weight = weights[feature.name]
                    weighted_sum += feature.value * weight
                    applied_weights_sum += weight
            
            # Calculate final sentiment index
            if applied_weights_sum > 0:
                sentiment_index = weighted_sum / applied_weights_sum