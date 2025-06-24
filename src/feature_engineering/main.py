"""
Feature Engineering Service Main Module

Entry point for the feature engineering service.
"""
import os
import sys
import asyncio
import logging
import signal
from typing import Dict, Any, List, Optional
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
from structlog import configure, processors, stdlib, threadlocal
import structlog
from python_json_logger import jsonlogger

# Application components
from feature_engineering.infrastructure.data_processor import DataProcessor
from feature_engineering.infrastructure.feature_store import FeatureStore
from feature_engineering.infrastructure.kafka_consumer import KafkaConsumer
from feature_engineering.application.use_cases.generate_features_use_case import GenerateFeaturesUseCase
from feature_engineering.domain.entities.feature import FeatureTimeframe


# Configure structured logging
def setup_logging():
    """Configure structured logging."""
    log_level = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Configure structlog
    configure(
        processors=[
            threadlocal.merge_threadlocal_context,
            processors.add_log_level,
            processors.TimeStamper(fmt="iso"),
            processors.StackInfoRenderer(),
            processors.format_exc_info,
            processors.UnicodeDecoder(),
            stdlib.render_to_log_kwargs,
        ],
        context_class=dict,
        logger_factory=stdlib.LoggerFactory(),
        wrapper_class=stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter('%(timestamp)s %(level)s %(name)s %(message)s')
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(handler)
    
    # Set log level for noisy libraries
    logging.getLogger('kafka').setLevel(logging.WARNING)
    logging.getLogger('confluent_kafka').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


class FeatureEngineeringService:
    """Main service class for the feature engineering service."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the feature engineering service.
        
        Args:
            config: Service configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_processor = None
        self.feature_store = None
        self.generate_features_use_case = None
        self.kafka_consumer = None
        
        # Flag to control service lifecycle
        self.running = False
    
    async def initialize(self):
        """Initialize service components."""
        try:
            self.logger.info("Initializing feature engineering service...")
            
            # Initialize data processor
            self.data_processor = DataProcessor(
                exchange_config=self.config['exchange'],
                redis_config=self.config['redis'],
                arctic_config=self.config['arctic']
            )
            
            # Initialize feature store
            self.feature_store = FeatureStore(db_uri=self.config['database']['uri'])
            await self.feature_store.initialize()
            
            # Initialize use case
            self.generate_features_use_case = GenerateFeaturesUseCase(
                feature_store=self.feature_store,
                data_processor=self.data_processor
            )
            
            # Initialize Kafka consumer if enabled
            if self.config.get('kafka', {}).get('enabled', False):
                self.kafka_consumer = KafkaConsumer(
                    kafka_config=self.config['kafka'],
                    generate_features_use_case=self.generate_features_use_case
                )
                await self.kafka_consumer.initialize()
            
            self.logger.info("Feature engineering service initialized successfully")
        
        except Exception as e:
            self.logger.error(f"Error initializing feature engineering service: {str(e)}")
            raise
    
    async def start(self):
        """Start the service."""
        if self.running:
            self.logger.warning("Service is already running")
            return
        
        try:
            self.logger.info("Starting feature engineering service...")
            self.running = True
            
            # Start Kafka consumer if enabled
            if self.kafka_consumer:
                await self.kafka_consumer.start()
            
            # Start scheduled feature generation if enabled
            if self.config.get('scheduled_generation', {}).get('enabled', False):
                self.logger.info("Starting scheduled feature generation...")
                asyncio.create_task(self._scheduled_feature_generation())
            
            self.logger.info("Feature engineering service started successfully")
        
        except Exception as e:
            self.logger.error(f"Error starting feature engineering service: {str(e)}")
            self.running = False
            raise
    
    async def stop(self):
        """Stop the service."""
        if not self.running:
            self.logger.warning("Service is not running")
            return
        
        try:
            self.logger.info("Stopping feature engineering service...")
            self.running = False
            
            # Stop Kafka consumer if enabled
            if self.kafka_consumer:
                await self.kafka_consumer.stop()
            
            # Close data processor connections
            if self.data_processor:
                await self.data_processor.close()
            
            self.logger.info("Feature engineering service stopped successfully")
        
        except Exception as e:
            self.logger.error(f"Error stopping feature engineering service: {str(e)}")
            raise
    
    async def _scheduled_feature_generation(self):
        """Periodically generate features for configured symbols."""
        try:
            scheduled_config = self.config.get('scheduled_generation', {})
            symbols = scheduled_config.get('symbols', [])
            interval_minutes = scheduled_config.get('interval_minutes', 60)
            timeframes = scheduled_config.get('timeframes', ['MINUTE_1'])
            
            self.logger.info(f"Scheduled feature generation started for {len(symbols)} symbols")
            
            while self.running:
                try:
                    for symbol in symbols:
                        for tf_str in timeframes:
                            # Convert string to enum
                            try:
                                timeframe = getattr(FeatureTimeframe, tf_str)
                                
                                # Generate features
                                self.logger.info(f"Generating scheduled features for {symbol} at {tf_str}")
                                await self.generate_features_use_case.generate_and_store_features(
                                    symbol=symbol,
                                    timeframe=timeframe
                                )
                            except (AttributeError, ValueError):
                                self.logger.error(f"Invalid timeframe: {tf_str}")
                    
                    # Wait for next interval
                    await asyncio.sleep(interval_minutes * 60)
                
                except Exception as e:
                    self.logger.error(f"Error in scheduled feature generation: {str(e)}")
                    await asyncio.sleep(60)  # Wait a bit before retrying
        
        except Exception as e:
            self.logger.error(f"Fatal error in scheduled feature generation: {str(e)}")
    
    async def generate_features_on_demand(
        self,
        symbol: str,
        timeframe_str: str = 'MINUTE_1',
        categories: Optional[List[str]] = None
    ) -> bool:
        """
        Generate features on demand for a specific symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USD")
            timeframe_str: Timeframe string (e.g., "MINUTE_1")
            categories: Optional list of feature categories to generate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert string to enum
            try:
                timeframe = getattr(FeatureTimeframe, timeframe_str)
            except (AttributeError, ValueError):
                self.logger.error(f"Invalid timeframe: {timeframe_str}")
                return False
            
            # Convert categories to set if provided
            categories_set = set(categories) if categories else None
            
            # Generate features
            self.logger.info(f"Generating on-demand features for {symbol} at {timeframe_str}")
            feature_set = await self.generate_features_use_case.generate_and_store_features(
                symbol=symbol,
                timeframe=timeframe,
                categories=categories_set
            )
            
            if feature_set:
                self.logger.info(f"Successfully generated on-demand features for {symbol}")
                return True
            else:
                self.logger.warning(f"No features generated for {symbol}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error generating on-demand features: {str(e)}")
            return False


def load_config() -> Dict[str, Any]:
    """Load service configuration from environment variables."""
    # Database configuration
    db_uri = os.environ.get('DATABASE_URI', 'postgresql+asyncpg://postgres:postgres@localhost:5432/trading')
    
    # Exchange configuration
    exchange_id = os.environ.get('EXCHANGE_ID', 'binance')
    exchange_api_key = os.environ.get('EXCHANGE_API_KEY', '')
    exchange_api_secret = os.environ.get('EXCHANGE_API_SECRET', '')
    
    # Redis configuration
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    redis_port = int(os.environ.get('REDIS_PORT', '6379'))
    redis_password = os.environ.get('REDIS_PASSWORD', '')
    
    # ArcticDB configuration
    arctic_uri = os.environ.get('ARCTIC_URI', 'mongodb://localhost:27017')
    
    # Kafka configuration
    kafka_enabled = os.environ.get('KAFKA_ENABLED', 'false').lower() == 'true'
    kafka_bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    kafka_group_id = os.environ.get('KAFKA_GROUP_ID', 'feature_engineering_group')
    
    # Scheduled generation configuration
    scheduled_enabled = os.environ.get('SCHEDULED_GENERATION_ENABLED', 'false').lower() == 'true'
    scheduled_symbols = os.environ.get('SCHEDULED_SYMBOLS', 'BTC/USD,ETH/USD').split(',')
    scheduled_interval = int(os.environ.get('SCHEDULED_INTERVAL_MINUTES', '60'))
    scheduled_timeframes = os.environ.get('SCHEDULED_TIMEFRAMES', 'MINUTE_1,HOUR_1,DAY_1').split(',')
    
    # Build configuration dictionary
    config = {
        'database': {
            'uri': db_uri
        },
        'exchange': {
            'primary': exchange_id,
            exchange_id: {
                'api_key': exchange_api_key,
                'api_secret': exchange_api_secret,
                'options': {}
            }
        },
        'redis': {
            'host': redis_host,
            'port': redis_port,
            'password': redis_password,
            'db': 0
        },
        'arctic': {
            'uri': arctic_uri
        },
        'kafka': {
            'enabled': kafka_enabled,
            'bootstrap_servers': kafka_bootstrap_servers,
            'group_id': kafka_group_id,
            'market_data_topic': 'market_data',
            'orderbook_topic': 'orderbook',
            'news_topic': 'news_data',
            'social_topic': 'social_data'
        },
        'scheduled_generation': {
            'enabled': scheduled_enabled,
            'symbols': scheduled_symbols,
            'interval_minutes': scheduled_interval,
            'timeframes': scheduled_timeframes
        }
    }
    
    return config


async def main():
    """Main entry point for the feature engineering service."""
    # Set up signal handlers
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config()
    
    # Create service instance
    service = FeatureEngineeringService(config)
    
    # Handle termination signals
    def signal_handler():
        logger.info("Received termination signal")
        stop_event.set()
    
    for signal_name in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signal_name),
            signal_handler
        )
    
    try:
        # Initialize and start the service
        await service.initialize()
        await service.start()
        
        # Wait for termination signal
        await stop_event.wait()
        
        # Stop the service
        await service.stop()
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)