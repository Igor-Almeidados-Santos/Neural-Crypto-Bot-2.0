"""
Generate Features Use Case Module

Defines the core application use case for generating features from market data.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime

from src.feature_engineering.domain.entities.feature import (
    Feature, FeatureType, FeatureCategory, FeatureTimeframe
)
from src.feature_engineering.domain.entities.feature_set import (
    FeatureSet, FeatureSetMetadata, FeatureSetType, FeatureSetStatus
)
from src.feature_engineering.application.features.technical_indicators import (
    MovingAverageCalculator, OscillatorCalculator, VolatilityCalculator,
    VolumeCalculator, TrendCalculator
)
from src.feature_engineering.application.features.statistical_features import (
    DescriptiveStatisticsCalculator, TimeSeriesStatisticsCalculator,
    DistributionalStatisticsCalculator, MarketRegimeCalculator, AnomalyDetectionCalculator
)
from src.feature_engineering.application.features.orderbook_features import OrderbookFeatureCalculator
from src.feature_engineering.application.features.sentiment_features import (
    NewsSentimentCalculator, SocialMediaSentimentCalculator, 
    OnChainSentimentCalculator, CombinedSentimentCalculator
)

logger = logging.getLogger(__name__)


class GenerateFeaturesUseCase:
    """
    Application use case for generating trading features from market data.
    
    This class orchestrates the feature generation process by delegating to 
    specialized calculator classes for different types of features.
    """
    
    def __init__(self, feature_store, data_processor):
        """
        Initialize the use case with required dependencies.
        
        Args:
            feature_store: Repository for storing and retrieving features
            data_processor: Service for processing and retrieving market data
        """
        self.feature_store = feature_store
        self.data_processor = data_processor
    
    async def generate_technical_features(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        feature_types: Optional[Set[str]] = None
    ) -> List[Feature]:
        """
        Generate technical indicator features for the specified symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            feature_types: Optional set of feature types to generate
                          (e.g., {"moving_averages", "oscillators"})
        
        Returns:
            List of generated Feature objects
        """
        try:
            # Get OHLCV data from data processor
            ohlcv_data = await self.data_processor.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if ohlcv_data.empty:
                logger.warning(f"No OHLCV data available for {symbol} at {timeframe}")
                return []
            
            # Initialize feature list
            features = []
            
            # Define which feature types to generate
            all_types = {
                "moving_averages", "oscillators", "volatility", 
                "volume", "trend"
            }
            types_to_generate = feature_types or all_types
            
            # Generate moving average features
            if "moving_averages" in types_to_generate:
                ma_calculator = MovingAverageCalculator(symbol, timeframe)
                
                # Simple Moving Averages
                for period in [5, 10, 20, 50, 100, 200]:
                    features.append(ma_calculator.calculate_sma(ohlcv_data['close'], period))
                
                # Exponential Moving Averages
                for period in [5, 10, 20, 50, 100, 200]:
                    features.append(ma_calculator.calculate_ema(ohlcv_data['close'], period))
                
                # Weighted Moving Average
                for period in [20, 50]:
                    features.append(ma_calculator.calculate_wma(ohlcv_data['close'], period))
                
                # Hull Moving Average
                for period in [9, 16]:
                    features.append(ma_calculator.calculate_hull_ma(ohlcv_data['close'], period))
            
            # Generate oscillator features
            if "oscillators" in types_to_generate:
                oscillator_calculator = OscillatorCalculator(symbol, timeframe)
                
                # RSI
                for period in [6, 14, 21]:
                    features.append(oscillator_calculator.calculate_rsi(ohlcv_data['close'], period))
                
                # Stochastic Oscillator
                for k_period, d_period in [(14, 3), (21, 7)]:
                    k_feature, d_feature = oscillator_calculator.calculate_stochastic(
                        ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'],
                        k_period, d_period
                    )
                    features.extend([k_feature, d_feature])
                
                # MACD
                for fast, slow, signal in [(12, 26, 9), (8, 21, 5)]:
                    macd_features = oscillator_calculator.calculate_macd(
                        ohlcv_data['close'], fast, slow, signal
                    )
                    features.extend(macd_features)
            
            # Generate volatility features
            if "volatility" in types_to_generate:
                volatility_calculator = VolatilityCalculator(symbol, timeframe)
                
                # Bollinger Bands
                for period, std in [(20, 2.0), (50, 2.5)]:
                    bb_features = volatility_calculator.calculate_bollinger_bands(
                        ohlcv_data['close'], period, std
                    )
                    features.extend(bb_features)
                
                # ATR
                for period in [14, 21]:
                    features.append(volatility_calculator.calculate_atr(
                        ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'], period
                    ))
            
            # Generate volume features
            if "volume" in types_to_generate and 'volume' in ohlcv_data.columns:
                volume_calculator = VolumeCalculator(symbol, timeframe)
                
                # On-Balance Volume
                features.append(volume_calculator.calculate_obv(
                    ohlcv_data['close'], ohlcv_data['volume']
                ))
                
                # VWAP
                features.append(volume_calculator.calculate_vwap(
                    ohlcv_data['high'], ohlcv_data['low'], 
                    ohlcv_data['close'], ohlcv_data['volume']
                ))
            
            # Generate trend features
            if "trend" in types_to_generate:
                trend_calculator = TrendCalculator(symbol, timeframe)
                
                # ADX
                adx_features = trend_calculator.calculate_adx(
                    ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close'], 14
                )
                features.extend(adx_features)
                
                # Ichimoku Cloud
                ichimoku_features = trend_calculator.calculate_ichimoku(
                    ohlcv_data['high'], ohlcv_data['low'], ohlcv_data['close']
                )
                features.extend(ichimoku_features.values())
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating technical features: {str(e)}")
            raise
    
    async def generate_statistical_features(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        feature_types: Optional[Set[str]] = None
    ) -> List[Feature]:
        """
        Generate statistical features for the specified symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            feature_types: Optional set of feature types to generate
                          (e.g., {"descriptive", "time_series"})
        
        Returns:
            List of generated Feature objects
        """
        try:
            # Get OHLCV data from data processor
            ohlcv_data = await self.data_processor.get_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if ohlcv_data.empty:
                logger.warning(f"No OHLCV data available for {symbol} at {timeframe}")
                return []
            
            # Initialize feature list
            features = []
            
            # Define which feature types to generate
            all_types = {
                "descriptive", "time_series", "distributional", 
                "market_regime", "anomaly"
            }
            types_to_generate = feature_types or all_types
            
            # Generate descriptive statistics features
            if "descriptive" in types_to_generate:
                desc_calculator = DescriptiveStatisticsCalculator(symbol, timeframe)
                
                # Rolling mean and std
                for window in [20, 50, 100]:
                    features.append(desc_calculator.calculate_rolling_mean(ohlcv_data['close'], window))
                    features.append(desc_calculator.calculate_rolling_std(ohlcv_data['close'], window))
                
                # Z-score
                for window in [20, 50]:
                    features.append(desc_calculator.calculate_z_score(ohlcv_data['close'], window))
                
                # Percentiles
                for window, percentile in [(100, 95), (100, 5)]:
                    features.append(desc_calculator.calculate_percentile(ohlcv_data['close'], window, percentile))
            
            # Generate time series statistics features
            if "time_series" in types_to_generate:
                ts_calculator = TimeSeriesStatisticsCalculator(symbol, timeframe)
                
                # Autocorrelation
                for lag in [1, 5, 10]:
                    features.append(ts_calculator.calculate_autocorrelation(ohlcv_data['close'], lag))
                
                # Stationarity
                features.append(ts_calculator.calculate_stationarity(ohlcv_data['close']))
            
            # Generate distributional statistics features
            if "distributional" in types_to_generate:
                dist_calculator = DistributionalStatisticsCalculator(symbol, timeframe)
                
                # Skewness and Kurtosis
                for window in [50, 100]:
                    features.append(dist_calculator.calculate_skewness(ohlcv_data['close'], window))
                    features.append(dist_calculator.calculate_kurtosis(ohlcv_data['close'], window))
                
                # Jarque-Bera test
                features.append(dist_calculator.calculate_jarque_bera(ohlcv_data['close'], 100))
            
            # Generate market regime features
            if "market_regime" in types_to_generate:
                regime_calculator = MarketRegimeCalculator(symbol, timeframe)
                
                # Hurst exponent
                features.append(regime_calculator.calculate_hurst_exponent(ohlcv_data['close']))
                
                # Volatility regime
                features.append(regime_calculator.calculate_volatility_regime(ohlcv_data['close']))
            
            # Generate anomaly detection features
            if "anomaly" in types_to_generate:
                anomaly_calculator = AnomalyDetectionCalculator(symbol, timeframe)
                
                # Price deviation
                features.append(anomaly_calculator.calculate_price_deviation(ohlcv_data['close']))
                
                # Volume anomaly
                if 'volume' in ohlcv_data.columns:
                    features.append(anomaly_calculator.calculate_volume_anomaly(ohlcv_data['volume']))
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating statistical features: {str(e)}")
            raise
    
    async def generate_orderbook_features(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        feature_types: Optional[Set[str]] = None
    ) -> List[Feature]:
        """
        Generate orderbook features for the specified symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            feature_types: Optional set of feature types to generate
                          (e.g., {"liquidity", "imbalance"})
        
        Returns:
            List of generated Feature objects
        """
        try:
            # Get orderbook data from data processor
            orderbook = await self.data_processor.get_orderbook(
                symbol=symbol,
                limit=100  # Get a deep orderbook for better analysis
            )
            
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                logger.warning(f"No orderbook data available for {symbol}")
                return []
            
            # Initialize feature list
            features = []
            
            # Initialize calculator
            ob_calculator = OrderbookFeatureCalculator(symbol, timeframe)
            
            # Define which feature types to generate
            all_types = {
                "basic", "liquidity", "market_impact", "depth"
            }
            types_to_generate = feature_types or all_types
            
            # Generate basic orderbook features
            if "basic" in types_to_generate:
                # Bid-ask spread
                features.append(ob_calculator.calculate_bid_ask_spread(orderbook))
                
                # Order book imbalance
                for depth in [5, 10, 20]:
                    features.append(ob_calculator.calculate_order_book_imbalance(orderbook, depth))
            
            # Generate liquidity features
            if "liquidity" in types_to_generate:
                # Liquidity at price levels
                for price_level in [0.1, 0.5, 1.0, 2.0]:
                    features.append(ob_calculator.calculate_liquidity_at_price(orderbook, price_level))
            
            # Generate market impact features
            if "market_impact" in types_to_generate:
                # Estimate market impact for different trade sizes
                # Use percentage of best bid/ask volume to make it relative to the market
                best_bid_qty = orderbook['bids'][0][1]
                
                for size_multiplier in [1, 5, 10, 50]:
                    trade_size = best_bid_qty * size_multiplier
                    features.append(ob_calculator.calculate_market_impact(orderbook, trade_size))
                    features.append(ob_calculator.calculate_market_impact(orderbook, -trade_size))  # Negative for sell
            
            # Generate depth analysis features
            if "depth" in types_to_generate:
                # Order book pressure
                for levels in [5, 10]:
                    features.append(ob_calculator.calculate_order_book_pressure(orderbook, levels))
                
                # Depth curve slope
                for side in ['bid', 'ask', 'both']:
                    features.append(ob_calculator.calculate_depth_curve_slope(orderbook, side))
                
                # Volume profile
                volume_profile_features = ob_calculator.calculate_orderbook_volume_profile(orderbook)
                features.extend(volume_profile_features.values())
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating orderbook features: {str(e)}")
            raise
    
    async def generate_sentiment_features(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        feature_types: Optional[Set[str]] = None
    ) -> List[Feature]:
        """
        Generate sentiment features for the specified symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            feature_types: Optional set of feature types to generate
                          (e.g., {"news", "social"})
        
        Returns:
            List of generated Feature objects
        """
        try:
            # Initialize feature list
            features = []
            
            # Define which feature types to generate
            all_types = {
                "news", "social", "on_chain", "combined"
            }
            types_to_generate = feature_types or all_types
            
            # Generate news sentiment features
            if "news" in types_to_generate:
                # Get news data from data processor
                news_data = await self.data_processor.get_news_data(
                    symbol=symbol,
                    limit=100  # Get recent news
                )
                
                if news_data:
                    news_calculator = NewsSentimentCalculator(symbol, timeframe)
                    
                    # News sentiment score
                    for window_hours in [6, 24, 72]:
                        features.append(news_calculator.calculate_news_sentiment_score(
                            news_data, window_hours
                        ))
                    
                    # News volume anomaly
                    features.append(news_calculator.calculate_news_volume(news_data))
            
            # Generate social media sentiment features
            if "social" in types_to_generate:
                # Get social media data from data processor
                social_data = await self.data_processor.get_social_data(
                    symbol=symbol,
                    limit=500  # Get more social posts
                )
                
                if social_data:
                    social_calculator = SocialMediaSentimentCalculator(symbol, timeframe)
                    
                    # Social sentiment score
                    for window_hours in [6, 24, 72]:
                        features.append(social_calculator.calculate_social_sentiment_score(
                            social_data, window_hours
                        ))
                    
                    # Social volume anomaly
                    features.append(social_calculator.calculate_social_volume(social_data))
                    
                    # Social sentiment momentum
                    features.append(social_calculator.calculate_social_sentiment_momentum(
                        social_data, 4, 24
                    ))
            
            # Generate on-chain sentiment features
            if "on_chain" in types_to_generate:
                # Only for supported blockchains
                base_asset = symbol.split('/')[0]
                
                if base_asset in ['BTC', 'ETH']:
                    # Get on-chain data from data processor
                    utxo_data = await self.data_processor.get_onchain_data(
                        symbol=symbol,
                        data_type='utxo'
                    )
                    
                    exchange_flow_data = await self.data_processor.get_onchain_data(
                        symbol=symbol,
                        data_type='exchange_flows'
                    )
                    
                    market_data = await self.data_processor.get_market_data(
                        symbol=symbol
                    )
                    
                    onchain_calculator = OnChainSentimentCalculator(symbol, timeframe)
                    
                    # HODL waves
                    if utxo_data:
                        hodl_features = onchain_calculator.calculate_hodl_waves(utxo_data)
                        features.extend(hodl_features.values())
                    
                    # Exchange flow balance
                    if exchange_flow_data:
                        for window_hours in [24, 72, 168]:
                            features.append(onchain_calculator.calculate_exchange_flow_balance(
                                exchange_flow_data, window_hours
                            ))
                    
                    # Stablecoin supply ratio
                    if market_data and 'market_cap' in market_data and 'stablecoin_supply' in market_data:
                        features.append(onchain_calculator.calculate_stablecoin_supply_ratio(
                            market_data['market_cap'], market_data['stablecoin_supply']
                        ))
            
            # Generate combined sentiment features
            if "combined" in types_to_generate and len(features) > 1:
                combined_calculator = CombinedSentimentCalculator(symbol, timeframe)
                
                # Combined sentiment index
                sentiment_features = [f for f in features if f.type == FeatureType.SENTIMENT]
                
                if sentiment_features:
                    features.append(combined_calculator.calculate_sentiment_index(sentiment_features))
                
                # Sentiment divergence (if we have both news and social)
                news_features = [f for f in features if 'NEWS_SENTIMENT' in f.name]
                social_features = [f for f in features if 'SOCIAL_SENTIMENT' in f.name]
                
                if news_features and social_features:
                    # Use 24h features for comparison if available
                    news_feature = next((f for f in news_features if '24h' in f.name), news_features[0])
                    social_feature = next((f for f in social_features if '24h' in f.name), social_features[0])
                    
                    features.append(combined_calculator.calculate_sentiment_divergence(
                        social_feature, news_feature
                    ))
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating sentiment features: {str(e)}")
            raise
    
    async def generate_all_features(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        categories: Optional[Set[str]] = None
    ) -> List[Feature]:
        """
        Generate all available features for the specified symbol and timeframe.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            categories: Optional set of feature categories to generate
                       (e.g., {"technical", "statistical"})
        
        Returns:
            List of generated Feature objects
        """
        try:
            # Define which categories to generate
            all_categories = {"technical", "statistical", "orderbook", "sentiment"}
            categories_to_generate = categories or all_categories
            
            # Initialize feature list
            all_features = []
            
            # Generate technical features
            if "technical" in categories_to_generate:
                technical_features = await self.generate_technical_features(
                    symbol, timeframe, start_time, end_time
                )
                all_features.extend(technical_features)
            
            # Generate statistical features
            if "statistical" in categories_to_generate:
                statistical_features = await self.generate_statistical_features(
                    symbol, timeframe, start_time, end_time
                )
                all_features.extend(statistical_features)
            
            # Generate orderbook features
            if "orderbook" in categories_to_generate:
                orderbook_features = await self.generate_orderbook_features(
                    symbol, timeframe
                )
                all_features.extend(orderbook_features)
            
            # Generate sentiment features
            if "sentiment" in categories_to_generate:
                sentiment_features = await self.generate_sentiment_features(
                    symbol, timeframe
                )
                all_features.extend(sentiment_features)
            
            return all_features
            
        except Exception as e:
            logger.error(f"Error generating all features: {str(e)}")
            raise
    
    async def create_feature_set(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        name: str,
        features: List[Feature],
        feature_set_type: FeatureSetType = FeatureSetType.PRODUCTION,
        description: str = ""
    ) -> FeatureSet:
        """
        Create a feature set from generated features.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            name: Name of the feature set
            features: List of features to include in the set
            feature_set_type: Type of feature set (e.g., PRODUCTION, TRAINING)
            description: Optional description of the feature set
            
        Returns:
            Created FeatureSet object
        """
        try:
            # Create metadata
            metadata = FeatureSetMetadata(
                description=description or f"Feature set for {symbol} at {timeframe} timeframe",
                version="1.0.0",
                author="FeatureEngineeringService"
            )
            
            # Create feature set
            feature_set = FeatureSet.create(
                name=name,
                type=feature_set_type,
                features=features,
                metadata=metadata
            )
            
            # Save to feature store
            await self.feature_store.save_feature_set(feature_set)
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error creating feature set: {str(e)}")
            raise
    
    async def generate_and_store_features(
        self,
        symbol: str,
        timeframe: FeatureTimeframe,
        categories: Optional[Set[str]] = None,
        feature_set_name: Optional[str] = None
    ) -> FeatureSet:
        """
        Generate all features and store them in a feature set.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            categories: Optional set of feature categories to generate
            feature_set_name: Optional name for the feature set
            
        Returns:
            Created FeatureSet object
        """
        try:
            # Generate all features
            features = await self.generate_all_features(
                symbol=symbol,
                timeframe=timeframe,
                categories=categories
            )
            
            if not features:
                logger.warning(f"No features generated for {symbol} at {timeframe}")
                return None
            
            # Default feature set name if not provided
            if not feature_set_name:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                feature_set_name = f"{symbol.replace('/', '_')}_{timeframe.name}_{timestamp}"
            
            # Create and store feature set
            feature_set = await self.create_feature_set(
                symbol=symbol,
                timeframe=timeframe,
                name=feature_set_name,
                features=features,
                feature_set_type=FeatureSetType.PRODUCTION,
                description=f"Automatically generated feature set for {symbol} at {timeframe} timeframe"
            )
            
            logger.info(f"Successfully generated and stored {len(features)} features for {symbol} at {timeframe}")
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error in generate_and_store_features: {str(e)}")
            raise