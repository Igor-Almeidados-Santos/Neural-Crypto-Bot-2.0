"""
Data Processor Module

Provides services for retrieving and processing market data from various sources.
"""
import logging
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta

# For exchange API integration
import ccxt.async_support as ccxt

# For caching
import redis
from arcticdb.arctic import Arctic
from arcticdb.version_store.library import VersionStore

# For async processing
from concurrent.futures import ThreadPoolExecutor

# For data validation
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OHLCVData(BaseModel):
    """Model for OHLCV data validation."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class DataProcessor:
    """Service for retrieving and processing market data."""
    
    def __init__(
        self,
        exchange_config: Dict[str, Any],
        redis_config: Dict[str, Any],
        arctic_config: Dict[str, Any],
        max_workers: int = 4
    ):
        """
        Initialize the data processor.
        
        Args:
            exchange_config: Configuration for exchange API connections
            redis_config: Configuration for Redis cache
            arctic_config: Configuration for ArcticDB time series database
            max_workers: Maximum number of thread workers for parallel processing
        """
        self.exchange_config = exchange_config
        self.redis_config = redis_config
        self.arctic_config = arctic_config
        self.max_workers = max_workers
        
        # Initialize exchange connections
        self.exchanges = {}
        for exchange_id, config in exchange_config.items():
            exchange_class = getattr(ccxt, exchange_id)
            self.exchanges[exchange_id] = exchange_class({
                'apiKey': config.get('api_key'),
                'secret': config.get('api_secret'),
                'enableRateLimit': True,
                'options': config.get('options', {})
            })
        
        # Set primary exchange
        self.primary_exchange_id = exchange_config.get('primary', list(exchange_config.keys())[0])
        
        # Initialize Redis connection
        self.redis = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            decode_responses=True
        )
        
        # Initialize ArcticDB connection
        self.arctic = Arctic(arctic_config.get('uri', 'mongodb://localhost:27017'))
        
        # Create libraries if they don't exist
        for lib_name in ['ohlcv', 'orderbook', 'news', 'social', 'onchain']:
            if not self.arctic.library_exists(lib_name):
                self.arctic.create_library(lib_name)
        
        # Get library references
        self.ohlcv_store: VersionStore = self.arctic.get_library('ohlcv')
        self.orderbook_store: VersionStore = self.arctic.get_library('orderbook')
        self.news_store: VersionStore = self.arctic.get_library('news')
        self.social_store: VersionStore = self.arctic.get_library('social')
        self.onchain_store: VersionStore = self.arctic.get_library('onchain')
        
        # Initialize thread pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def close(self):
        """Close all connections and resources."""
        # Close exchange connections
        for exchange in self.exchanges.values():
            await exchange.close()
        
        # Close Redis connection
        self.redis.close()
        
        # Shutdown thread pool
        self.executor.shutdown()
    
    async def get_ohlcv_data(
        self,
        symbol: str,
        timeframe: Any,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Retrieve OHLCV (candlestick) data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe: Time resolution for the data
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            limit: Maximum number of candles to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame containing OHLCV data with columns:
            [timestamp, open, high, low, close, volume]
        """
        try:
            # Prepare timeframe string for CCXT
            tf_str = self._get_ccxt_timeframe(timeframe)
            
            # Prepare symbol ID for storage
            symbol_id = symbol.replace('/', '_')
            
            # Create unique key for this data
            data_key = f"{symbol_id}_{tf_str}"
            
            # Set default times if not provided
            if end_time is None:
                end_time = datetime.utcnow()
            
            if start_time is None:
                # Calculate start time based on limit and timeframe
                td = self._timeframe_to_timedelta(tf_str)
                start_time = end_time - (td * limit)
            
            # Try to get data from cache
            if use_cache:
                try:
                    cached_data = self.ohlcv_store.read(data_key)
                    
                    # Filter by time range
                    mask = (cached_data.index >= start_time) & (cached_data.index <= end_time)
                    filtered_data = cached_data[mask]
                    
                    if not filtered_data.empty:
                        logger.debug(f"Retrieved OHLCV data for {symbol} from cache")
                        return filtered_data
                except Exception as e:
                    logger.debug(f"Cache miss for OHLCV data: {str(e)}")
            
            # Get data from exchange
            exchange = self.exchanges[self.primary_exchange_id]
            
            # Convert datetime to milliseconds for CCXT
            since = int(start_time.timestamp() * 1000)
            
            # Fetch OHLCV data
            raw_data = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf_str,
                since=since,
                limit=limit
            )
            
            if not raw_data:
                logger.warning(f"No OHLCV data returned for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(
                raw_data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the data
            try:
                self.ohlcv_store.write(data_key, df)
                logger.debug(f"Cached OHLCV data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cache OHLCV data: {str(e)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 50,
        use_cache: bool = False,
        max_age: int = 5  # Max age in seconds for cached orderbook
    ) -> Dict[str, List]:
        """
        Retrieve orderbook data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            limit: Depth of the orderbook to retrieve
            use_cache: Whether to use cached data
            max_age: Maximum age in seconds for cached orderbook
            
        Returns:
            Dictionary with 'bids' and 'asks' lists of [price, amount] pairs
        """
        try:
            # Prepare symbol ID for cache key
            symbol_id = symbol.replace('/', '_')
            cache_key = f"orderbook:{symbol_id}"
            
            # Try to get from cache
            if use_cache:
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    # Check timestamp
                    cache_timestamp = self.redis.get(f"{cache_key}:timestamp")
                    if cache_timestamp:
                        timestamp = float(cache_timestamp)
                        age = datetime.utcnow().timestamp() - timestamp
                        
                        if age < max_age:
                            # Deserialize the orderbook
                            import json
                            orderbook = json.loads(cached_data)
                            logger.debug(f"Retrieved orderbook for {symbol} from cache (age: {age:.2f}s)")
                            return orderbook
            
            # Get orderbook from exchange
            exchange = self.exchanges[self.primary_exchange_id]
            orderbook = await exchange.fetch_order_book(symbol, limit)
            
            if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
                logger.warning(f"Invalid orderbook data returned for {symbol}")
                return {'bids': [], 'asks': []}
            
            # Cache the orderbook
            try:
                import json
                self.redis.set(cache_key, json.dumps(orderbook))
                self.redis.set(f"{cache_key}:timestamp", datetime.utcnow().timestamp())
                # Set expiration to avoid stale data
                self.redis.expire(cache_key, 60)  # 1 minute TTL
                self.redis.expire(f"{cache_key}:timestamp", 60)
                logger.debug(f"Cached orderbook for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cache orderbook: {str(e)}")
            
            return orderbook
            
        except Exception as e:
            logger.error(f"Error retrieving orderbook: {str(e)}")
            return {'bids': [], 'asks': []}
    
    async def get_news_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve news data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            limit: Maximum number of news items to retrieve
            
        Returns:
            List of news items with sentiment scores and timestamps
        """
        try:
            # Extract base asset (e.g., BTC from BTC/USD)
            base_asset = symbol.split('/')[0]
            
            # Set default times if not provided
            if end_time is None:
                end_time = datetime.utcnow()
            
            if start_time is None:
                # Default to 7 days of news
                start_time = end_time - timedelta(days=7)
            
            # Try to get from cache
            try:
                # Read from news store
                news_key = f"{base_asset}_news"
                cached_data = self.news_store.read(news_key)
                
                if not cached_data.empty:
                    # Convert DataFrame to list of dicts
                    news_list = cached_data.to_dict('records')
                    
                    # Filter by time range
                    filtered_news = [
                        news for news in news_list
                        if start_time <= news['timestamp'] <= end_time
                    ]
                    
                    # Limit the number of items
                    return filtered_news[:limit]
            except Exception as e:
                logger.debug(f"Cache miss for news data: {str(e)}")
            
            # For demonstration purposes, generate sample news data
            # In a real system, this would be fetched from news APIs
            
            # Generate random news items
            import random
            from faker import Faker
            fake = Faker()
            
            # Generate dates between start and end time
            date_range = (end_time - start_time).total_seconds()
            
            news_data = []
            for _ in range(min(limit, 50)):  # Cap at 50 items for demo
                seconds_offset = random.uniform(0, date_range)
                news_timestamp = start_time + timedelta(seconds=seconds_offset)
                
                # Create news item
                news_item = {
                    'title': fake.sentence(),
                    'summary': fake.paragraph(),
                    'source': random.choice(['CoinDesk', 'CoinTelegraph', 'Bloomberg', 'Reuters', 'Forbes']),
                    'url': fake.uri(),
                    'timestamp': news_timestamp,
                    'sentiment_score': random.uniform(-1.0, 1.0),
                    'relevance': random.uniform(0.5, 1.0)
                }
                news_data.append(news_item)
            
            # Sort by timestamp (newest first)
            news_data.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Cache the data
            try:
                df = pd.DataFrame(news_data)
                self.news_store.write(f"{base_asset}_news", df)
                logger.debug(f"Cached news data for {base_asset}")
            except Exception as e:
                logger.warning(f"Failed to cache news data: {str(e)}")
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error retrieving news data: {str(e)}")
            return []
    
    async def get_social_data(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Retrieve social media data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            limit: Maximum number of social media posts to retrieve
            
        Returns:
            List of social media posts with sentiment scores and timestamps
        """
        try:
            # Extract base asset (e.g., BTC from BTC/USD)
            base_asset = symbol.split('/')[0]
            
            # Set default times if not provided
            if end_time is None:
                end_time = datetime.utcnow()
            
            if start_time is None:
                # Default to 3 days of social data
                start_time = end_time - timedelta(days=3)
            
            # Try to get from cache
            try:
                # Read from social store
                social_key = f"{base_asset}_social"
                cached_data = self.social_store.read(social_key)
                
                if not cached_data.empty:
                    # Convert DataFrame to list of dicts
                    social_list = cached_data.to_dict('records')
                    
                    # Filter by time range
                    filtered_social = [
                        post for post in social_list
                        if start_time <= post['timestamp'] <= end_time
                    ]
                    
                    # Limit the number of items
                    return filtered_social[:limit]
            except Exception as e:
                logger.debug(f"Cache miss for social data: {str(e)}")
            
            # For demonstration purposes, generate sample social data
            # In a real system, this would be fetched from social media APIs
            
            # Generate random social posts
            import random
            from faker import Faker
            fake = Faker()
            
            # Generate dates between start and end time
            date_range = (end_time - start_time).total_seconds()
            
            social_data = []
            for _ in range(min(limit, 200)):  # Cap at 200 items for demo
                seconds_offset = random.uniform(0, date_range)
                post_timestamp = start_time + timedelta(seconds=seconds_offset)
                
                # Create social post
                post_item = {
                    'content': fake.text(max_nb_chars=280),
                    'platform': random.choice(['Twitter', 'Reddit', 'Telegram', 'Discord']),
                    'username': fake.user_name(),
                    'timestamp': post_timestamp,
                    'sentiment_score': random.uniform(-1.0, 1.0),
                    'engagement': int(random.expovariate(1/100))  # Exponential distribution for engagement
                }
                social_data.append(post_item)
            
            # Sort by timestamp (newest first)
            social_data.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Cache the data
            try:
                df = pd.DataFrame(social_data)
                self.social_store.write(f"{base_asset}_social", df)
                logger.debug(f"Cached social data for {base_asset}")
            except Exception as e:
                logger.warning(f"Failed to cache social data: {str(e)}")
            
            return social_data
            
        except Exception as e:
            logger.error(f"Error retrieving social data: {str(e)}")
            return []
    
    async def get_onchain_data(
        self,
        symbol: str,
        data_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve on-chain data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            data_type: Type of on-chain data to retrieve (e.g., 'utxo', 'exchange_flows')
            start_time: Optional start time for data retrieval
            end_time: Optional end time for data retrieval
            
        Returns:
            List of on-chain data items
        """
        try:
            # Extract base asset (e.g., BTC from BTC/USD)
            base_asset = symbol.split('/')[0]
            
            # Set default times if not provided
            if end_time is None:
                end_time = datetime.utcnow()
            
            if start_time is None:
                # Default to 30 days of on-chain data
                start_time = end_time - timedelta(days=30)
            
            # Try to get from cache
            try:
                # Read from onchain store
                onchain_key = f"{base_asset}_{data_type}"
                cached_data = self.onchain_store.read(onchain_key)
                
                if not cached_data.empty:
                    # Convert DataFrame to list of dicts
                    onchain_list = cached_data.to_dict('records')
                    
                    # Filter by time range if applicable
                    if 'timestamp' in cached_data.columns:
                        filtered_onchain = [
                            item for item in onchain_list
                            if start_time <= item['timestamp'] <= end_time
                        ]
                        return filtered_onchain
                    else:
                        return onchain_list
            except Exception as e:
                logger.debug(f"Cache miss for on-chain data: {str(e)}")
            
            # For demonstration purposes, generate sample on-chain data
            import random
            
            if data_type == 'utxo':
                # Generate UTXO age distribution data
                utxo_data = []
                
                # Create distribution across different age brackets
                age_brackets = [
                    (0, 7),      # 0-7 days
                    (7, 30),     # 7-30 days
                    (30, 90),    # 30-90 days
                    (90, 180),   # 90-180 days
                    (180, 365),  # 180-365 days
                    (365, 730),  # 1-2 years
                    (730, 1460)  # 2-4 years
                ]
                
                # Total coin supply distribution
                total_supply = 1000000  # Arbitrary total supply
                
                # Distribution weights (more weight to younger UTXOs)
                weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05]
                
                for i, ((min_age, max_age), weight) in enumerate(zip(age_brackets, weights)):
                    # Number of UTXOs in this bracket
                    num_utxos = int(random.normalvariate(100, 20))
                    
                    # Total value in this bracket
                    bracket_value = total_supply * weight
                    
                    # Generate individual UTXOs
                    for _ in range(num_utxos):
                        age = random.uniform(min_age, max_age)
                        value = bracket_value / num_utxos * random.uniform(0.5, 1.5)
                        
                        utxo_data.append({
                            'age_days': age,
                            'value': value
                        })
                
                # Cache the data
                try:
                    df = pd.DataFrame(utxo_data)
                    self.onchain_store.write(f"{base_asset}_utxo", df)
                    logger.debug(f"Cached UTXO data for {base_asset}")
                except Exception as e:
                    logger.warning(f"Failed to cache UTXO data: {str(e)}")
                
                return utxo_data
                
            elif data_type == 'exchange_flows':
                # Generate exchange flow data
                flow_data = []
                
                # Generate dates between start and end time
                date_range = (end_time - start_time).total_seconds()
                
                # Generate flow events
                num_events = 100
                for _ in range(num_events):
                    seconds_offset = random.uniform(0, date_range)
                    flow_timestamp = start_time + timedelta(seconds=seconds_offset)
                    
                    # More outflows than inflows on average (65% outflows)
                    direction = random.choice(['out', 'out', 'out', 'in', 'in'])
                    
                    # Generate flow amount (log-normal distribution)
                    amount = np.random.lognormal(mean=1, sigma=2)
                    
                    flow_data.append({
                        'timestamp': flow_timestamp,
                        'amount': amount,
                        'direction': direction
                    })
                
                # Sort by timestamp
                flow_data.sort(key=lambda x: x['timestamp'])
                
                # Cache the data
                try:
                    df = pd.DataFrame(flow_data)
                    self.onchain_store.write(f"{base_asset}_exchange_flows", df)
                    logger.debug(f"Cached exchange flow data for {base_asset}")
                except Exception as e:
                    logger.warning(f"Failed to cache exchange flow data: {str(e)}")
                
                return flow_data
            
            else:
                logger.warning(f"Unsupported on-chain data type: {data_type}")
                return []
            
        except Exception as e:
            logger.error(f"Error retrieving on-chain data: {str(e)}")
            return []
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve general market data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            
        Returns:
            Dictionary containing market data like market cap, supply, etc.
        """
        try:
            # Extract base asset (e.g., BTC from BTC/USD)
            base_asset = symbol.split('/')[0]
            
            # Try to get from cache
            cache_key = f"market:{base_asset}"
            cached_data = self.redis.get(cache_key)
            
            if cached_data:
                import json
                market_data = json.loads(cached_data)
                
                # Check if data is fresh (less than 1 hour old)
                cache_timestamp = self.redis.get(f"{cache_key}:timestamp")
                if cache_timestamp:
                    timestamp = float(cache_timestamp)
                    age = datetime.utcnow().timestamp() - timestamp
                    
                    if age < 3600:  # 1 hour in seconds
                        logger.debug(f"Retrieved market data for {base_asset} from cache")
                        return market_data
            
            # For demonstration purposes, generate sample market data
            # In a real system, this would be fetched from market data APIs
            
            # Generate random market data
            import random
            
            # Get current price from exchange
            exchange = self.exchanges[self.primary_exchange_id]
            ticker = await exchange.fetch_ticker(symbol)
            current_price = ticker['last'] if ticker and 'last' in ticker else 10000.0  # Default fallback
            
            # Generate market cap and supply
            if base_asset == 'BTC':
                circulating_supply = 19_000_000 + random.uniform(-100000, 100000)
                max_supply = 21_000_000
            elif base_asset == 'ETH':
                circulating_supply = 120_000_000 + random.uniform(-1000000, 1000000)
                max_supply = None  # ETH has no hard cap
            else:
                circulating_supply = 100_000_000 + random.uniform(-5000000, 5000000)
                max_supply = 1000_000_000
            
            market_cap = circulating_supply * current_price
            
            # Stablecoin supply (for stablecoin ratio calculations)
            stablecoin_supply = 100_000_000_000 + random.uniform(-5000000000, 5000000000)
            
            # Volume data
            daily_volume = market_cap * random.uniform(0.01, 0.1)  # 1-10% of market cap
            
            # Create market data dictionary
            market_data = {
                'symbol': symbol,
                'price': current_price,
                'market_cap': market_cap,
                'circulating_supply': circulating_supply,
                'max_supply': max_supply,
                'daily_volume': daily_volume,
                'stablecoin_supply': stablecoin_supply,
                'updated_at': datetime.utcnow().isoformat()
            }
            
            # Cache the data
            try:
                import json
                self.redis.set(cache_key, json.dumps(market_data))
                self.redis.set(f"{cache_key}:timestamp", datetime.utcnow().timestamp())
                # Set expiration
                self.redis.expire(cache_key, 3600)  # 1 hour TTL
                self.redis.expire(f"{cache_key}:timestamp", 3600)
                logger.debug(f"Cached market data for {base_asset}")
            except Exception as e:
                logger.warning(f"Failed to cache market data: {str(e)}")
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error retrieving market data: {str(e)}")
            return {}
    
    def _get_ccxt_timeframe(self, timeframe) -> str:
        """Convert internal timeframe enum to CCXT timeframe string."""
        # Handle case where timeframe is already a string
        if isinstance(timeframe, str):
            return timeframe
        
        # Mapping from FeatureTimeframe enum to CCXT timeframe strings
        tf_mapping = {
            'TICK': '1m',  # No tick level in CCXT, use 1m as minimum
            'SECOND_1': '1m',  # No second level in CCXT, use 1m as minimum
            'SECOND_5': '1m',
            'SECOND_15': '1m',
            'SECOND_30': '1m',
            'MINUTE_1': '1m',
            'MINUTE_3': '3m',
            'MINUTE_5': '5m',
            'MINUTE_15': '15m',
            'MINUTE_30': '30m',
            'HOUR_1': '1h',
            'HOUR_2': '2h',
            'HOUR_4': '4h',
            'HOUR_6': '6h',
            'HOUR_8': '8h',
            'HOUR_12': '12h',
            'DAY_1': '1d',
            'DAY_3': '3d',
            'WEEK_1': '1w',
            'MONTH_1': '1M'
        }
        
        # Get the name of the enum value
        if hasattr(timeframe, 'name'):
            tf_name = timeframe.name
        else:
            tf_name = str(timeframe)
        
        # Return mapped value or default to 1d
        return tf_mapping.get(tf_name, '1d')
    
    def _timeframe_to_timedelta(self, timeframe: str) -> timedelta:
        """Convert CCXT timeframe string to timedelta."""
        # Parse the timeframe string
        if timeframe.endswith('m'):
            minutes = int(timeframe[:-1])
            return timedelta(minutes=minutes)
        elif timeframe.endswith('h'):
            hours = int(timeframe[:-1])
            return timedelta(hours=hours)
        elif timeframe.endswith('d'):
            days = int(timeframe[:-1])
            return timedelta(days=days)
        elif timeframe.endswith('w'):
            weeks = int(timeframe[:-1])
            return timedelta(weeks=weeks)
        elif timeframe.endswith('M'):
            months = int(timeframe[:-1])
            return timedelta(days=months * 30)  # Approximate
        else:
            # Default to 1 day
            return timedelta(days=1)