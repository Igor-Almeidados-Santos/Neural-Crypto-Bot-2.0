"""
Kafka Consumer Module

Provides services for consuming market data events from Kafka and triggering feature generation.
"""
import logging
import asyncio
import json
from typing import Dict, List, Optional, Set, Union, Any, Callable
from datetime import datetime

# For Kafka integration
from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

# For async processing
from concurrent.futures import ThreadPoolExecutor

# For feature generation
from feature_engineering.domain.entities.feature import FeatureTimeframe
from feature_engineering.application.use_cases.generate_features_use_case import GenerateFeaturesUseCase

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Service for consuming market data events from Kafka."""
    
    def __init__(
        self,
        kafka_config: Dict[str, Any],
        generate_features_use_case: GenerateFeaturesUseCase,
        max_workers: int = 4
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            kafka_config: Configuration for Kafka connection
            generate_features_use_case: Use case for generating features
            max_workers: Maximum number of thread workers for parallel processing
        """
        self.kafka_config = kafka_config
        self.generate_features_use_case = generate_features_use_case
        
        # Configure Kafka consumer
        self.consumer_config = {
            'bootstrap.servers': kafka_config.get('bootstrap_servers', 'localhost:9092'),
            'group.id': kafka_config.get('group_id', 'feature_engineering_group'),
            'auto.offset.reset': kafka_config.get('auto_offset_reset', 'earliest'),
            'enable.auto.commit': kafka_config.get('enable_auto_commit', False),
            'max.poll.interval.ms': kafka_config.get('max_poll_interval_ms', 300000),  # 5 minutes
            'session.timeout.ms': kafka_config.get('session_timeout_ms', 30000),  # 30 seconds
            'heartbeat.interval.ms': kafka_config.get('heartbeat_interval_ms', 10000),  # 10 seconds
        }
        
        # Add security configuration if provided
        if 'security_protocol' in kafka_config:
            self.consumer_config['security.protocol'] = kafka_config['security_protocol']
            
            if kafka_config['security_protocol'] == 'SASL_SSL':
                self.consumer_config['sasl.mechanism'] = kafka_config.get('sasl_mechanism', 'PLAIN')
                self.consumer_config['sasl.username'] = kafka_config.get('sasl_username', '')
                self.consumer_config['sasl.password'] = kafka_config.get('sasl_password', '')
        
        # Initialize Kafka consumer
        self.consumer = Consumer(self.consumer_config)
        
        # Topic configurations
        self.market_data_topic = kafka_config.get('market_data_topic', 'market_data')
        self.orderbook_topic = kafka_config.get('orderbook_topic', 'orderbook')
        self.news_topic = kafka_config.get('news_topic', 'news_data')
        self.social_topic = kafka_config.get('social_topic', 'social_data')
        
        # Flag to control consumer loop
        self.running = False
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Event handlers
        self.event_handlers = {
            self.market_data_topic: self._handle_market_data_event,
            self.orderbook_topic: self._handle_orderbook_event,
            self.news_topic: self._handle_news_event,
            self.social_topic: self._handle_social_event
        }
    
    async def initialize(self):
        """Initialize topics and subscribe to them."""
        try:
            # Create admin client to check/create topics
            admin_client = AdminClient({'bootstrap.servers': self.consumer_config['bootstrap.servers']})
            
            # Check if topics exist, create if not
            topics_to_create = []
            for topic in [self.market_data_topic, self.orderbook_topic, self.news_topic, self.social_topic]:
                topics_metadata = admin_client.list_topics(topic=topic)
                if topic not in topics_metadata.topics:
                    logger.info(f"Topic {topic} doesn't exist, creating...")
                    topics_to_create.append(NewTopic(
                        topic,
                        num_partitions=3,
                        replication_factor=1,
                        config={"retention.ms": 604800000}  # 7 days retention
                    ))
            
            if topics_to_create:
                admin_client.create_topics(topics_to_create)
            
            # Subscribe to topics
            self.consumer.subscribe([
                self.market_data_topic,
                self.orderbook_topic,
                self.news_topic,
                self.social_topic
            ])
            
            logger.info(f"Subscribed to topics: {', '.join([self.market_data_topic, self.orderbook_topic, self.news_topic, self.social_topic])}")
        except Exception as e:
            logger.error(f"Error initializing Kafka consumer: {str(e)}")
            raise
    
    async def start(self):
        """Start consuming messages."""
        if self.running:
            logger.warning("Kafka consumer is already running")
            return
        
        self.running = True
        logger.info("Starting Kafka consumer...")
        
        # Start in a separate task to not block
        asyncio.create_task(self._consume_loop())
    
    async def stop(self):
        """Stop consuming messages."""
        logger.info("Stopping Kafka consumer...")
        self.running = False
        
        # Close consumer
        self.consumer.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        logger.info("Kafka consumer stopped")
    
    async def _consume_loop(self):
        """Main loop for consuming messages."""
        try:
            logger.info("Kafka consumer loop started")
            
            while self.running:
                try:
                    # Poll for messages (timeout in seconds)
                    msg = self.consumer.poll(1.0)
                    
                    if msg is None:
                        # No message available within timeout
                        continue
                    
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # End of partition event, not an error
                            logger.debug(f"Reached end of partition {msg.partition()}")
                        else:
                            # Actual error
                            logger.error(f"Kafka error: {msg.error()}")
                    else:
                        # Valid message
                        topic = msg.topic()
                        key = msg.key().decode('utf-8') if msg.key() else None
                        value = msg.value().decode('utf-8')
                        
                        # Process message in the executor to avoid blocking
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            self.executor,
                            self._process_message,
                            topic, key, value
                        )
                    
                    # Manually commit offset after processing
                    self.consumer.commit(asynchronous=False)
                    
                except KafkaException as ke:
                    logger.error(f"Kafka exception: {str(ke)}")
                except Exception as e:
                    logger.error(f"Error in consume loop: {str(e)}")
                
                # Small sleep to avoid CPU spinning
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Fatal error in Kafka consume loop: {str(e)}")
            self.running = False
    
    def _process_message(self, topic: str, key: Optional[str], value: str):
        """Process a Kafka message."""
        try:
            # Parse message as JSON
            data = json.loads(value)
            
            # Get appropriate handler for the topic
            handler = self.event_handlers.get(topic)
            
            if handler:
                # Run the handler
                asyncio.run(handler(key, data))
            else:
                logger.warning(f"No handler defined for topic: {topic}")
        
        except json.JSONDecodeError:
            logger.error(f"Failed to parse message as JSON: {value}")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    async def _handle_market_data_event(self, key: Optional[str], data: Dict[str, Any]):
        """Handle market data events."""
        try:
            # Extract symbol and other data
            symbol = key or data.get('symbol')
            event_type = data.get('event_type')
            
            if not symbol:
                logger.error("Market data event missing symbol")
                return
            
            # Different handling based on event type
            if event_type == 'trade':
                # Trade event, generate features if it's a significant trade
                trade_amount = data.get('amount', 0)
                if trade_amount > data.get('average_trade_size', 0) * 2:
                    logger.info(f"Significant trade detected for {symbol}, generating features")
                    await self._generate_features_for_symbol(symbol)
            
            elif event_type == 'candle':
                # Candle closed event, generate features
                timeframe = data.get('timeframe', '1m')
                
                # Map string timeframe to enum
                tf_enum = self._map_timeframe_string(timeframe)
                
                if tf_enum:
                    logger.info(f"Candle closed for {symbol} at {timeframe}, generating features")
                    await self._generate_features_for_symbol(symbol, tf_enum)
            
            elif event_type == 'ticker':
                # Ticker update, check for significant price change
                price_change_pct = data.get('price_change_pct', 0)
                
                if abs(price_change_pct) > 1.0:  # >1% change
                    logger.info(f"Significant price change for {symbol}: {price_change_pct}%, generating features")
                    await self._generate_features_for_symbol(symbol)
        
        except Exception as e:
            logger.error(f"Error handling market data event: {str(e)}")
    
    async def _handle_orderbook_event(self, key: Optional[str], data: Dict[str, Any]):
        """Handle orderbook events."""
        try:
            # Extract symbol and other data
            symbol = key or data.get('symbol')
            event_type = data.get('event_type')
            
            if not symbol:
                logger.error("Orderbook event missing symbol")
                return
            
            # Different handling based on event type
            if event_type == 'snapshot':
                # Full orderbook snapshot, generate features
                logger.info(f"Orderbook snapshot received for {symbol}, generating features")
                await self._generate_features_for_symbol(symbol, categories={"orderbook"})
            
            elif event_type == 'update':
                # Orderbook update, check for significant changes
                
                # Check bid-ask spread change
                spread_change_pct = data.get('spread_change_pct', 0)
                
                # Check imbalance change
                imbalance_change = data.get('imbalance_change', 0)
                
                if abs(spread_change_pct) > 5.0 or abs(imbalance_change) > 0.5:
                    logger.info(f"Significant orderbook change for {symbol}, generating features")
                    await self._generate_features_for_symbol(symbol, categories={"orderbook"})
        
        except Exception as e:
            logger.error(f"Error handling orderbook event: {str(e)}")
    
    async def _handle_news_event(self, key: Optional[str], data: Dict[str, Any]):
        """Handle news events."""
        try:
            # Extract relevant data
            asset = key or data.get('asset')
            
            if not asset:
                logger.error("News event missing asset")
                return
            
            # Convert asset to symbol format (assuming USD pair)
            symbol = f"{asset}/USD"
            
            # Check if it's a high impact news
            impact = data.get('impact', 'low')
            sentiment_score = data.get('sentiment_score', 0)
            
            if impact in ['high', 'medium'] or abs(sentiment_score) > 0.7:
                logger.info(f"High impact news for {asset}, generating sentiment features")
                await self._generate_features_for_symbol(symbol, categories={"sentiment"})
        
        except Exception as e:
            logger.error(f"Error handling news event: {str(e)}")
    
    async def _handle_social_event(self, key: Optional[str], data: Dict[str, Any]):
        """Handle social media events."""
        try:
            # Extract relevant data
            asset = key or data.get('asset')
            
            if not asset:
                logger.error("Social event missing asset")
                return
            
            # Convert asset to symbol format (assuming USD pair)
            symbol = f"{asset}/USD"
            
            # Check if it's a viral or high-engagement post
            engagement = data.get('engagement', 0)
            sentiment_score = data.get('sentiment_score', 0)
            
            if engagement > 1000 or abs(sentiment_score) > 0.8:
                logger.info(f"Viral social post for {asset}, generating sentiment features")
                await self._generate_features_for_symbol(symbol, categories={"sentiment"})
        
        except Exception as e:
            logger.error(f"Error handling social event: {str(e)}")
    
    async def _generate_features_for_symbol(
        self,
        symbol: str,
        timeframe: Optional[FeatureTimeframe] = None,
        categories: Optional[Set[str]] = None
    ):
        """Generate and store features for a symbol."""
        try:
            # Default to 1 minute timeframe if not specified
            if not timeframe:
                timeframe = FeatureTimeframe.MINUTE_1
            
            # Generate features using the use case
            feature_set = await self.generate_features_use_case.generate_and_store_features(
                symbol=symbol,
                timeframe=timeframe,
                categories=categories
            )
            
            if feature_set:
                logger.info(f"Successfully generated features for {symbol} at {timeframe.name}")
                return True
            else:
                logger.warning(f"No features generated for {symbol} at {timeframe.name}")
                return False
            
        except Exception as e:
            logger.error(f"Error generating features for {symbol}: {str(e)}")
            return False
    
    def _map_timeframe_string(self, timeframe_str: str) -> Optional[FeatureTimeframe]:
        """Map a timeframe string to a FeatureTimeframe enum value."""
        # Common timeframe mappings
        mapping = {
            '1s': FeatureTimeframe.SECOND_1,
            '5s': FeatureTimeframe.SECOND_5,
            '15s': FeatureTimeframe.SECOND_15,
            '30s': FeatureTimeframe.SECOND_30,
            '1m': FeatureTimeframe.MINUTE_1,
            '3m': FeatureTimeframe.MINUTE_3,
            '5m': FeatureTimeframe.MINUTE_5,
            '15m': FeatureTimeframe.MINUTE_15,
            '30m': FeatureTimeframe.MINUTE_30,
            '1h': FeatureTimeframe.HOUR_1,
            '2h': FeatureTimeframe.HOUR_2,
            '4h': FeatureTimeframe.HOUR_4,
            '6h': FeatureTimeframe.HOUR_6,
            '8h': FeatureTimeframe.HOUR_8,
            '12h': FeatureTimeframe.HOUR_12,
            '1d': FeatureTimeframe.DAY_1,
            '3d': FeatureTimeframe.DAY_3,
            '1w': FeatureTimeframe.WEEK_1,
            '1M': FeatureTimeframe.MONTH_1
        }
        
        return mapping.get(timeframe_str)