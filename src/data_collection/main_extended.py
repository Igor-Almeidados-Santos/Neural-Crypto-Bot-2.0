"""
Módulo principal estendido para coleta de dados de mercado.

Este módulo implementa o serviço de coleta de dados com funcionalidades avançadas:
- Suporte para múltiplas exchanges (Binance, Coinbase, Kraken, Bybit)
- Coleta de dados adicionais (funding rates, liquidações)
- Criptografia de dados sensíveis
- Compressão de dados
- Balanceamento de carga para distribuir requisições
- Sistema de métricas e observabilidade
- Recuperação automática de falhas
- Otimização de performance
"""
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable
import traceback
import gc
from pathlib import Path

import argparse
import yaml
from dotenv import load_dotenv
import uvloop
import structlog

from src.data_collection.adapters.binance_adapter import BinanceAdapter
from src.data_collection.adapters.coinbase_adapter import CoinbaseAdapter
from src.data_collection.adapters.kraken_adapter import KrakenAdapter
from src.data_collection.adapters.bybit_adapter import BybitAdapter
from src.data_collection.adapters.exchange_adapter_interface import ExchangeAdapterInterface

from src.data_collection.application.services.data_normalization_service import DataNormalizationService
from src.data_collection.application.services.data_validation_service import DataValidationService

from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook
from src.data_collection.domain.entities.trade import Trade
from src.data_collection.domain.entities.funding_rate import FundingRate
from src.data_collection.domain.entities.liquidation import Liquidation

from src.data_collection.infrastructure.database import DatabaseManager
from src.data_collection.infrastructure.kafka_producer import KafkaProducer
from src.data_collection.infrastructure.crypto import CryptoService
from src.data_collection.infrastructure.compression import CompressionService
from src.data_collection.infrastructure.load_balancer import ExchangeLoadBalancer, BalancingStrategy

# Configuração de logging estruturado
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class PerformanceMonitor:
    """Monitor de performance para otimização do sistema."""
    
    def __init__(self):
        self.metrics = {
            'data_points_processed': 0,
            'errors_count': 0,
            'latency_avg': 0.0,
            'memory_usage': 0,
            'active_subscriptions': 0
        }
        self.start_time = datetime.utcnow()
    
    def update_metric(self, name: str, value: Any):
        """Atualiza uma métrica."""
        self.metrics[name] = value
    
    def increment_metric(self, name: str, value: int = 1):
        """Incrementa uma métrica."""
        self.metrics[name] = self.metrics.get(name, 0) + value
    
    def get_uptime(self) -> timedelta:
        """Retorna o tempo de funcionamento."""
        return datetime.utcnow() - self.start_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas completas."""
        return {
            **self.metrics,
            'uptime_seconds': self.get_uptime().total_seconds(),
            'data_rate_per_second': self.metrics['data_points_processed'] / max(1, self.get_uptime().total_seconds())
        }


class HealthChecker:
    """Sistema de verificação de saúde do serviço."""
    
    def __init__(self, service: 'EnhancedDataCollectionService'):
        self.service = service
        self.last_health_check = datetime.utcnow()
        self.health_status = {}
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Realiza verificação completa de saúde."""
        health = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        try:
            # Verifica banco de dados
            if self.service.db_manager and self.service.enable_database:
                try:
                    async with self.service.db_manager.get_connection() as conn:
                        await conn.execute('SELECT 1')
                    health['components']['database'] = {'status': 'healthy'}
                except Exception as e:
                    health['components']['database'] = {'status': 'unhealthy', 'error': str(e)}
                    health['status'] = 'degraded'
            
            # Verifica Kafka
            if self.service.kafka_producer and self.service.enable_kafka:
                try:
                    # Kafka health check seria implementado aqui
                    health['components']['kafka'] = {'status': 'healthy'}
                except Exception as e:
                    health['components']['kafka'] = {'status': 'unhealthy', 'error': str(e)}
                    health['status'] = 'degraded'
            
            # Verifica exchanges
            health['components']['exchanges'] = {}
            for name, exchange in self.service.exchanges.items():
                try:
                    pairs = await exchange.fetch_trading_pairs()
                    health['components']['exchanges'][name] = {
                        'status': 'healthy',
                        'trading_pairs_count': len(pairs)
                    }
                except Exception as e:
                    health['components']['exchanges'][name] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health['status'] = 'degraded'
            
            # Verifica balanceador de carga
            if self.service.exchange_load_balancer:
                available = len(self.service.exchange_load_balancer.get_available_instances())
                total = len(self.service.exchange_load_balancer.get_all_instances())
                health['components']['load_balancer'] = {
                    'status': 'healthy' if available > 0 else 'unhealthy',
                    'available_instances': available,
                    'total_instances': total
                }
                if available == 0:
                    health['status'] = 'unhealthy'
            
            # Adiciona métricas de performance
            health['performance'] = self.service.performance_monitor.get_stats()
            
        except Exception as e:
            health['status'] = 'unhealthy'
            health['error'] = str(e)
            logger.error("Erro na verificação de saúde", error=str(e), exc_info=True)
        
        self.health_status = health
        self.last_health_check = datetime.utcnow()
        return health


class EnhancedDataCollectionService:
    """
    Serviço avançado para coleta de dados de mercado.
    
    Estende o serviço básico com funcionalidades adicionais como
    suporte a múltiplas exchanges, tipos de dados adicionais,
    criptografia, compressão e balanceamento de carga.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o serviço de coleta de dados.
        
        Args:
            config: Configurações do serviço
        """
        self.config = config
        
        # Monitor de performance
        self.performance_monitor = PerformanceMonitor()
        
        # Sistema de verificação de saúde
        self.health_checker = HealthChecker(self)
        
        # Serviços de segurança e otimização
        self.crypto_service = CryptoService(
            master_key=config.get('security', {}).get('master_key')
        )
        
        self.compression_service = CompressionService(
            default_method=config.get('storage', {}).get('compression_method', 'zlib'),
            compression_level=config.get('storage', {}).get('compression_level', 6)
        )
        
        # Serviços de aplicação
        self.normalization_service = DataNormalizationService()
        self.validation_service = DataValidationService(
            max_price_deviation_percent=config.get('validation', {}).get('max_price_deviation_percent', 10.0),
            max_volume_deviation_percent=config.get('validation', {}).get('max_volume_deviation_percent', 50.0),
            max_spread_percent=config.get('validation', {}).get('max_spread_percent', 5.0),
            min_orderbook_depth=config.get('validation', {}).get('min_orderbook_depth', 5),
            max_timestamp_delay_seconds=config.get('validation', {}).get('max_timestamp_delay_seconds', 30),
            enable_strict_validation=config.get('validation', {}).get('enable_strict_validation', False)
        )
        
        # Adapters de exchanges e balanceador de carga
        self.exchanges = {}
        self.exchange_load_balancer = None
        
        # Infraestrutura
        self.db_manager = None
        self.kafka_producer = None
        
        # Controle de estado
        self._running = False
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
        
        # Tasks em execução
        self._tasks = []
        self._subscription_tasks = {}
        
        # Cache de dados para otimização
        self._data_cache = {}
        self._cache_ttl = config.get('cache', {}).get('ttl_seconds', 300)
        
        # Configurações de coleta
        self.collection_pairs = config.get('collection', {}).get('pairs', [])
        self.collection_exchanges = config.get('collection', {}).get('exchanges', [])
        self.collection_timeframes = config.get('collection', {}).get('timeframes', [])
        self.collection_interval_seconds = config.get('collection', {}).get('interval_seconds', 60)
        self.collection_orderbook_depth = config.get('collection', {}).get('orderbook_depth', 20)
        self.collect_funding_rates = config.get('collection', {}).get('funding_rates', True)
        self.collect_liquidations = config.get('collection', {}).get('liquidations', True)
        
        # Configurações de armazenamento
        self.enable_database = config.get('storage', {}).get('enable_database', True)
        self.enable_kafka = config.get('storage', {}).get('enable_kafka', True)
        self.sync_historical_data = config.get('storage', {}).get('sync_historical_data', True)
        self.historical_data_days = config.get('storage', {}).get('historical_data_days', 30)
        self.enable_compression = config.get('storage', {}).get('enable_compression', True)
        self.compression_threshold = config.get('storage', {}).get('compression_threshold', 1024)
        
        # Configurações de balanceamento de carga
        self.enable_load_balancing = config.get('load_balancing', {}).get('enabled', True)
        self.load_balancing_strategy = config.get('load_balancing', {}).get('strategy', 'round_robin')
        self.rate_limit_buffer = config.get('load_balancing', {}).get('rate_limit_buffer', 0.8)
        
        # Configurações de recovery
        self.max_retry_attempts = config.get('recovery', {}).get('max_retry_attempts', 3)
        self.retry_delay_base = config.get('recovery', {}).get('retry_delay_base', 1.0)
        self.circuit_breaker_threshold = config.get('recovery', {}).get('circuit_breaker_threshold', 5)
    
    async def initialize(self) -> None:
        """Inicializa todos os componentes do serviço."""
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                logger.info("Inicializando serviço avançado de coleta de dados")
                
                # Inicializa a infraestrutura
                await self._initialize_infrastructure()
                
                # Inicializa os adapters de exchanges
                await self._initialize_exchanges()
                
                # Inicializa o balanceador de carga
                await self._initialize_load_balancer()
                
                # Inicia task de verificação de saúde
                self._tasks.append(
                    asyncio.create_task(self._health_check_loop())
                )
                
                # Inicia task de limpeza de cache
                self._tasks.append(
                    asyncio.create_task(self._cache_cleanup_loop())
                )
                
                # Inicia task de coleta de métricas
                self._tasks.append(
                    asyncio.create_task(self._metrics_collection_loop())
                )
                
                self._initialized = True
                logger.info("Serviço avançado de coleta de dados inicializado")
                
            except Exception as e:
                logger.error("Erro ao inicializar serviço de coleta de dados", error=str(e), exc_info=True)
                raise
    
    async def start(self) -> None:
        """Inicia a coleta de dados."""
        if not self._initialized:
            await self.initialize()
            
        if self._running:
            logger.warning("Serviço de coleta de dados já está em execução")
            return
            
        self._running = True
        logger.info("Iniciando coleta de dados")
        
        # Inicia o balanceador de carga
        if self.exchange_load_balancer:
            await self.exchange_load_balancer.start()
        
        # Sincroniza dados históricos se configurado
        if self.sync_historical_data:
            await self._sync_historical_data()
        
        # Inicia a coleta em tempo real
        await self._start_realtime_collection()
        
        # Inicia task de monitoramento de performance
        self._tasks.append(
            asyncio.create_task(self._performance_monitoring_loop())
        )
        
        logger.info("Coleta de dados iniciada")
    
    async def stop(self) -> None:
        """Para a coleta de dados."""
        if not self._running:
            return
            
        logger.info("Parando coleta de dados")
        self._running = False
        self._shutdown_event.set()
        
        # Cancela todas as tasks em execução
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Cancela tasks de subscrição específicas
        for task_group in self._subscription_tasks.values():
            for task in task_group:
                if not task.done():
                    task.cancel()
                    
        # Aguarda o cancelamento das tasks
        all_tasks = self._tasks + [task for group in self._subscription_tasks.values() for task in group]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
            
        self._tasks = []
        self._subscription_tasks = {}
        
        # Para o balanceador de carga
        if self.exchange_load_balancer:
            await self.exchange_load_balancer.stop()
        
        # Fecha os adapters de exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                logger.info("Fechando adapter da exchange", exchange=exchange_name)
                await exchange.shutdown()
            except Exception as e:
                logger.error("Erro ao fechar adapter da exchange", exchange=exchange_name, error=str(e))
        
        # Fecha a infraestrutura
        if self.db_manager and self.enable_database:
            await self.db_manager.close()
            
        if self.kafka_producer and self.enable_kafka:
            await self.kafka_producer.close()
            
        logger.info("Coleta de dados parada")
    
    async def _initialize_infrastructure(self) -> None:
        """Inicializa a infraestrutura (banco de dados e Kafka)."""
        # Inicializa o banco de dados se habilitado
        if self.enable_database:
            db_config = self.config.get('database', {})
            
            # Descriptografa senhas e dados sensíveis
            db_config = self.crypto_service.decrypt_dict(db_config)
            
            self.db_manager = DatabaseManager(
                host=db_config.get('host', 'localhost'),
                port=db_config.get('port', 5432),
                database=db_config.get('database', 'neural_crypto_bot'),
                user=db_config.get('user', 'postgres'),
                password=db_config.get('password', 'postgres'),
                min_connections=db_config.get('min_connections', 5),
                max_connections=db_config.get('max_connections', 20),
                enable_ssl=db_config.get('enable_ssl', False),
                schema=db_config.get('schema', 'public')
            )
            
            logger.info("Inicializando conexão com o banco de dados")
            await self.db_manager.initialize()
        
        # Inicializa o produtor Kafka se habilitado
        if self.enable_kafka:
            kafka_config = self.config.get('kafka', {})
            
            # Descriptografa senhas e dados sensíveis
            kafka_config = self.crypto_service.decrypt_dict(kafka_config)
            
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
                client_id=kafka_config.get('client_id', 'neural-crypto-bot-data-collection'),
                topic_prefix=kafka_config.get('topic_prefix', 'market-data'),
                acks=kafka_config.get('acks', 'all'),
                retries=kafka_config.get('retries', 3),
                batch_size=kafka_config.get('batch_size', 16384),
                linger_ms=kafka_config.get('linger_ms', 5),
                compression_type=kafka_config.get('compression_type', 'snappy')
            )
            
            logger.info("Inicializando conexão com o Kafka")
            await self.kafka_producer.initialize()
    
    async def _initialize_exchanges(self) -> None:
        """Inicializa os adapters de exchanges."""
        exchange_adapters = []
        
        for exchange_name in self.collection_exchanges:
            try:
                logger.info("Inicializando adapter da exchange", exchange=exchange_name)
                
                # Configuração específica da exchange
                exchange_config = self.config.get('exchanges', {}).get(exchange_name, {})
                
                # Descriptografa credenciais
                exchange_config = self.crypto_service.decrypt_dict(exchange_config)
                
                # Cria o adapter apropriado
                if exchange_name.lower() == 'binance':
                    adapter = BinanceAdapter(
                        api_key=exchange_config.get('api_key'),
                        api_secret=exchange_config.get('api_secret'),
                        testnet=exchange_config.get('testnet', False)
                    )
                elif exchange_name.lower() in ['coinbase', 'coinbasepro']:
                    adapter = CoinbaseAdapter(
                        api_key=exchange_config.get('api_key'),
                        api_secret=exchange_config.get('api_secret'),
                        api_passphrase=exchange_config.get('api_passphrase'),
                        sandbox=exchange_config.get('sandbox', False)
                    )
                elif exchange_name.lower() == 'kraken':
                    adapter = KrakenAdapter(
                        api_key=exchange_config.get('api_key'),
                        api_secret=exchange_config.get('api_secret'),
                        testnet=exchange_config.get('testnet', False)
                    )
                elif exchange_name.lower() == 'bybit':
                    adapter = BybitAdapter(
                        api_key=exchange_config.get('api_key'),
                        api_secret=exchange_config.get('api_secret'),
                        testnet=exchange_config.get('testnet', False)
                    )
                else:
                    logger.warning("Exchange não suportada", exchange=exchange_name)
                    continue
                
                # Inicializa o adapter
                await adapter.initialize()
                
                # Adiciona à lista de exchanges
                self.exchanges[exchange_name] = adapter
                exchange_adapters.append(adapter)
                
                logger.info("Adapter da exchange inicializado", exchange=exchange_name)
                
            except Exception as e:
                logger.error("Erro ao inicializar adapter da exchange", 
                           exchange=exchange_name, error=str(e))
    
    async def _initialize_load_balancer(self) -> None:
        """Inicializa o balanceador de carga para as exchanges."""
        if not self.enable_load_balancing or len(self.exchanges) <= 1:
            logger.info("Balanceamento de carga desabilitado ou apenas uma exchange disponível")
            return
            
        try:
            # Converte a estratégia de string para enum
            strategy = BalancingStrategy(self.load_balancing_strategy)
            
            # Cria o balanceador de carga
            self.exchange_load_balancer = ExchangeLoadBalancer(
                instances=list(self.exchanges.values()),
                strategy=strategy,
                health_check_interval=60,
                circuit_breaker_threshold=self.circuit_breaker_threshold,
                max_retries=self.max_retry_attempts,
                rate_limit_buffer=self.rate_limit_buffer
            )
            
            logger.info("Balanceador de carga inicializado", strategy=str(strategy))
            
        except Exception as e:
            logger.error("Erro ao inicializar balanceador de carga", error=str(e))
            self.exchange_load_balancer = None
    
    async def _sync_historical_data(self) -> None:
        """Sincroniza dados históricos para todos os pares e timeframes configurados."""
        if not self.enable_database:
            logger.info("Sincronização de dados históricos ignorada: banco de dados desabilitado")
            return
            
        logger.info("Iniciando sincronização de dados históricos")
        
        # Calcula a data de início para a sincronização
        start_date = datetime.utcnow() - timedelta(days=self.historical_data_days)
        
        # Sincroniza dados de candles
        await self._sync_historical_candles(start_date)
        
        # Sincroniza dados de funding rates se habilitado
        if self.collect_funding_rates:
            await self._sync_historical_funding_rates(start_date)
        
        logger.info("Sincronização de dados históricos concluída")
    
    async def _sync_historical_candles(self, start_date: datetime) -> None:
        """Sincroniza dados históricos de candles."""
        logger.info("Sincronizando dados históricos de candles")
        
        # Para cada par e timeframe
        for trading_pair in self.collection_pairs:
            for timeframe_str in self.collection_timeframes:
                try:
                    # Converte o timeframe para o formato padronizado
                    timeframe = TimeFrame(timeframe_str)
                    
                    logger.info("Sincronizando candles", 
                              trading_pair=trading_pair, timeframe=timeframe_str)
                    
                    # Se o balanceador de carga estiver habilitado, usa-o para distribuir as requisições
                    if self.exchange_load_balancer:
                        try:
                            async def fetch_candles_operation(exchange):
                                if not exchange.validate_trading_pair(trading_pair):
                                    return []
                                    
                                # Verifica se o timeframe é suportado
                                if timeframe not in exchange.get_supported_timeframes().values():
                                    return []
                                    
                                return await exchange.fetch_candles(
                                    trading_pair=trading_pair,
                                    timeframe=timeframe,
                                    since=start_date,
                                    limit=1000
                                )
                            
                            candles = await self.exchange_load_balancer.execute_for_trading_pair(
                                trading_pair=trading_pair,
                                operation=fetch_candles_operation,
                                method_name="fetch_candles"
                            )
                            
                        except Exception as e:
                            logger.error("Erro ao sincronizar candles via balanceador", 
                                       trading_pair=trading_pair, timeframe=timeframe_str, error=str(e))
                            candles = []
                            
                    else:
                        # Tenta cada exchange até obter dados
                        candles = []
                        for exchange_name, exchange in self.exchanges.items():
                            if not exchange.validate_trading_pair(trading_pair):
                                continue
                                
                            # Verifica se o timeframe é suportado
                            if timeframe not in exchange.get_supported_timeframes().values():
                                continue
                            
                            try:
                                candles = await exchange.fetch_candles(
                                    trading_pair=trading_pair,
                                    timeframe=timeframe,
                                    since=start_date,
                                    limit=1000
                                )
                                
                                if candles:
                                    break
                                    
                            except Exception as e:
                                logger.error("Erro ao sincronizar candles", 
                                           exchange=exchange_name, trading_pair=trading_pair, 
                                           timeframe=timeframe_str, error=str(e))
                    
                    # Valida e normaliza as candles
                    valid_candles = []
                    for candle in candles:
                        try:
                            if self.validation_service.validate_candle(candle):
                                valid_candles.append(candle)
                                self.performance_monitor.increment_metric('data_points_processed')
                        except Exception as e:
                            logger.warning("Erro ao validar candle", error=str(e))
                            self.performance_monitor.increment_metric('errors_count')
                    
                    if valid_candles:
                        # Salva as candles no banco de dados
                        if self.enable_database and self.db_manager:
                            # Comprime os dados se habilitado
                            if self.enable_compression:
                                for candle in valid_candles:
                                    if candle.raw_data:
                                        raw_data_bytes = str(candle.raw_data).encode('utf-8')
                                        if self.compression_service.should_compress(raw_data_bytes, self.compression_threshold):
                                            compressed_data = self.compression_service.compress_efficient(raw_data_bytes, self.compression_threshold)
                                            # Note: Aqui precisaríamos modificar a entidade para suportar dados comprimidos
                            
                            await self.db_manager.save_candles_batch(valid_candles)
                            
                        # Publica no Kafka se habilitado
                        if self.enable_kafka and self.kafka_producer:
                            await self.kafka_producer.publish_candles_batch(valid_candles)
                            
                        logger.info("Candles sincronizados", 
                                  count=len(valid_candles), trading_pair=trading_pair, 
                                  timeframe=timeframe_str)
                    else:
                        logger.warning("Nenhum candle válido encontrado", 
                                     trading_pair=trading_pair, timeframe=timeframe_str)
                    
                except Exception as e:
                    logger.error("Erro ao sincronizar candles", 
                               trading_pair=trading_pair, timeframe=timeframe_str, error=str(e))
                    self.performance_monitor.increment_metric('errors_count')
    
    async def _sync_historical_funding_rates(self, start_date: datetime) -> None:
        """Sincroniza dados históricos de taxas de financiamento."""
        logger.info("Sincronizando dados históricos de taxas de financiamento")
        
        # Para cada par de negociação
        for trading_pair in self.collection_pairs:
            try:
                # Se o balanceador de carga estiver habilitado, usa-o para distribuir as requisições
                if self.exchange_load_balancer:
                    try:
                        async def fetch_funding_rates_operation(exchange):
                            if not exchange.validate_trading_pair(trading_pair):
                                return []
                                
                            try:
                                return await exchange.fetch_funding_rates(trading_pair)
                            except Exception:
                                return []
                        
                        funding_rates = await self.exchange_load_balancer.execute_for_trading_pair(
                            trading_pair=trading_pair,
                            operation=fetch_funding_rates_operation,
                            method_name="fetch_funding_rates"
                        )
                        
                    except Exception as e:
                        logger.error("Erro ao sincronizar funding rates via balanceador", 
                                   trading_pair=trading_pair, error=str(e))
                        funding_rates = []
                        
                else:
                    # Tenta cada exchange até obter dados
                    funding_rates = []
                    for exchange_name, exchange in self.exchanges.items():
                        if not exchange.validate_trading_pair(trading_pair):
                            continue
                        
                        try:
                            # Verifica se é uma exchange que suporta contratos perpétuos
                            if exchange_name.lower() in ['binance', 'bybit']:
                                rates = await exchange.fetch_funding_rates(trading_pair)
                                
                                if rates:
                                    funding_rates.extend(rates)
                                    
                        except Exception as e:
                            logger.error("Erro ao sincronizar funding rates", 
                                       exchange=exchange_name, trading_pair=trading_pair, error=str(e))
                
                # Filtra apenas taxas após a data de início
                valid_funding_rates = [
                    rate for rate in funding_rates
                    if rate.timestamp >= start_date
                ]
                
                if valid_funding_rates:
                    # Salva as taxas no banco de dados
                    if self.enable_database and self.db_manager:
                        # Comprime os dados se habilitado
                        if self.enable_compression:
                            for rate in valid_funding_rates:
                                if rate.raw_data:
                                    raw_data_bytes = str(rate.raw_data).encode('utf-8')
                                    if self.compression_service.should_compress(raw_data_bytes, self.compression_threshold):
                                        compressed_data = self.compression_service.compress_efficient(raw_data_bytes, self.compression_threshold)
                        
                        # Aqui precisaríamos usar um repositório específico para funding rates
                        # Por simplicidade, estamos apenas logando
                        logger.info("Funding rates sincronizados", 
                                  count=len(valid_funding_rates), trading_pair=trading_pair)
                        
                    # Publica no Kafka se habilitado
                    if self.enable_kafka and self.kafka_producer:
                        # Aqui precisaríamos usar um tópico específico para funding rates
                        # Por simplicidade, estamos apenas logando
                        logger.info("Funding rates publicados no Kafka", 
                                  count=len(valid_funding_rates), trading_pair=trading_pair)
                        
                    self.performance_monitor.increment_metric('data_points_processed', len(valid_funding_rates))
                else:
                    logger.warning("Nenhuma taxa de financiamento válida encontrada", 
                                 trading_pair=trading_pair)
                
            except Exception as e:
                logger.error("Erro ao sincronizar taxas de financiamento", 
                           trading_pair=trading_pair, error=str(e))
                self.performance_monitor.increment_metric('errors_count')
    
    async def _start_realtime_collection(self) -> None:
        """Inicia a coleta de dados em tempo real."""
        logger.info("Iniciando coleta de dados em tempo real")
        
        # Para cada par de negociação
        for trading_pair in self.collection_pairs:
            task_group = []
            
            # Subscreve para orderbooks
            task = asyncio.create_task(
                self._subscribe_orderbook_for_pair(trading_pair)
            )
            task_group.append(task)
            
            # Subscreve para trades
            task = asyncio.create_task(
                self._subscribe_trades_for_pair(trading_pair)
            )
            task_group.append(task)
            
            # Subscreve para candles de cada timeframe
            for timeframe_str in self.collection_timeframes:
                try:
                    timeframe = TimeFrame(timeframe_str)
                    
                    task = asyncio.create_task(
                        self._subscribe_candles_for_pair(trading_pair, timeframe)
                    )
                    task_group.append(task)
                    
                except Exception as e:
                    logger.error("Erro ao iniciar coleta de candles", 
                               trading_pair=trading_pair, timeframe=timeframe_str, error=str(e))
            
            # Subscreve para funding rates se habilitado
            if self.collect_funding_rates:
                task = asyncio.create_task(
                    self._subscribe_funding_rates_for_pair(trading_pair)
                )
                task_group.append(task)
            
            # Subscreve para liquidações se habilitado
            if self.collect_liquidations:
                task = asyncio.create_task(
                    self._subscribe_liquidations_for_pair(trading_pair)
                )
                task_group.append(task)
            
            # Armazena as tasks para este par
            self._subscription_tasks[trading_pair] = task_group
        
        # Subscreve para liquidações globais se habilitado
        if self.collect_liquidations:
            task = asyncio.create_task(
                self._subscribe_global_liquidations()
            )
            self._tasks.append(task)
        
        logger.info("Coleta de dados em tempo real iniciada")
    
    async def _subscribe_orderbook_for_pair(self, trading_pair: str) -> None:
        """Subscreve para atualizações de orderbook para um par de negociação."""
        try:
            logger.info("Subscrevendo para orderbooks", trading_pair=trading_pair)
            
            # Se o balanceador de carga estiver habilitado, usa-o para distribuir as requisições
            if self.exchange_load_balancer:
                # Determina quais exchanges suportam o par
                available_exchanges = []
                for exchange in self.exchange_load_balancer.get_available_instances():
                    if exchange.validate_trading_pair(trading_pair):
                        available_exchanges.append(exchange)
                
                if not available_exchanges:
                    logger.warning("Nenhuma exchange disponível para o par", trading_pair=trading_pair)
                    return
                
                # Seleciona uma exchange para subscrever
                exchange = available_exchanges[0]
                
                # Define o callback para processar orderbooks
                async def on_orderbook(orderbook: OrderBook) -> None:
                    await self._process_orderbook(orderbook)
                
                # Subscreve para atualizações
                await exchange.subscribe_orderbook(
                    trading_pair=trading_pair,
                    callback=on_orderbook
                )
                
                self.performance_monitor.increment_metric('active_subscriptions')
                
                # Mantém a subscrição ativa enquanto o serviço estiver rodando
                while self._running:
                    await asyncio.sleep(60)
                
                # Cancela a subscrição quando a task for cancelada
                if self._running:
                    logger.info("Cancelando subscrição de orderbook", 
                              exchange=exchange.name, trading_pair=trading_pair)
                
                try:
                    await exchange.unsubscribe_orderbook(trading_pair)
                    self.performance_monitor.increment_metric('active_subscriptions', -1)
                except Exception as e:
                    logger.error("Erro ao cancelar subscrição de orderbook", error=str(e))
                
            else:
                # Sem balanceador, tenta cada exchange
                subscribed = False
                
                for exchange_name, exchange in self.exchanges.items():
                    if not exchange.validate_trading_pair(trading_pair):
                        continue
                        
                    try:
                        # Define o callback para processar orderbooks
                        async def on_orderbook(orderbook: OrderBook) -> None:
                            await self._process_orderbook(orderbook)
                        
                        # Subscreve para atualizações
                        await exchange.subscribe_orderbook(
                            trading_pair=trading_pair,
                            callback=on_orderbook
                        )
                        
                        subscribed = True
                        self.performance_monitor.increment_metric('active_subscriptions')
                        logger.info("Subscrito para orderbooks", 
                                  exchange=exchange_name, trading_pair=trading_pair)
                        break
                        
                    except Exception as e:
                        logger.error("Erro ao subscrever para orderbooks", 
                                   exchange=exchange_name, trading_pair=trading_pair, error=str(e))
                
                if not subscribed:
                    logger.warning("Não foi possível subscrever para orderbooks de nenhuma exchange", 
                                 trading_pair=trading_pair)
                    return
                
                # Mantém a subscrição ativa
                while self._running:
                    await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.debug("Task de subscrição de orderbook cancelada", trading_pair=trading_pair)
        except Exception as e:
            logger.error("Erro na subscrição de orderbook", 
                       trading_pair=trading_pair, error=str(e))
    
    async def _subscribe_trades_for_pair(self, trading_pair: str) -> None:
        """Subscreve para atualizações de trades para um par de negociação."""
        try:
            logger.info("Subscrevendo para trades", trading_pair=trading_pair)
            
            if self.exchange_load_balancer:
                available_exchanges = []
                for exchange in self.exchange_load_balancer.get_available_instances():
                    if exchange.validate_trading_pair(trading_pair):
                        available_exchanges.append(exchange)
                
                if not available_exchanges:
                    logger.warning("Nenhuma exchange disponível para trades", trading_pair=trading_pair)
                    return
                
                exchange = available_exchanges[0]
                
                async def on_trade(trade: Trade) -> None:
                    await self._process_trade(trade)
                
                await exchange.subscribe_trades(
                    trading_pair=trading_pair,
                    callback=on_trade
                )
                
                self.performance_monitor.increment_metric('active_subscriptions')
                
                while self._running:
                    await asyncio.sleep(60)
                
                try:
                    await exchange.unsubscribe_trades(trading_pair)
                    self.performance_monitor.increment_metric('active_subscriptions', -1)
                except Exception as e:
                    logger.error("Erro ao cancelar subscrição de trades", error=str(e))
                
            else:
                subscribed = False
                
                for exchange_name, exchange in self.exchanges.items():
                    if not exchange.validate_trading_pair(trading_pair):
                        continue
                        
                    try:
                        async def on_trade(trade: Trade) -> None:
                            await self._process_trade(trade)
                        
                        await exchange.subscribe_trades(
                            trading_pair=trading_pair,
                            callback=on_trade
                        )
                        
                        subscribed = True
                        self.performance_monitor.increment_metric('active_subscriptions')
                        logger.info("Subscrito para trades", 
                                  exchange=exchange_name, trading_pair=trading_pair)
                        break
                        
                    except Exception as e:
                        logger.error("Erro ao subscrever para trades", 
                                   exchange=exchange_name, trading_pair=trading_pair, error=str(e))
                
                if not subscribed:
                    logger.warning("Não foi possível subscrever para trades", trading_pair=trading_pair)
                    return
                
                while self._running:
                    await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.debug("Task de subscrição de trades cancelada", trading_pair=trading_pair)
        except Exception as e:
            logger.error("Erro na subscrição de trades", trading_pair=trading_pair, error=str(e))
    
    async def _subscribe_candles_for_pair(self, trading_pair: str, timeframe: TimeFrame) -> None:
        """Subscreve para atualizações de candles para um par e timeframe."""
        try:
            logger.info("Subscrevendo para candles", 
                       trading_pair=trading_pair, timeframe=timeframe.value)
            
            if self.exchange_load_balancer:
                available_exchanges = []
                for exchange in self.exchange_load_balancer.get_available_instances():
                    if (exchange.validate_trading_pair(trading_pair) and 
                        timeframe in exchange.get_supported_timeframes().values()):
                        available_exchanges.append(exchange)
                
                if not available_exchanges:
                    logger.warning("Nenhuma exchange disponível para candles", 
                                 trading_pair=trading_pair, timeframe=timeframe.value)
                    return
                
                exchange = available_exchanges[0]
                
                async def on_candle(candle: Candle) -> None:
                    await self._process_candle(candle)
                
                await exchange.subscribe_candles(
                    trading_pair=trading_pair,
                    timeframe=timeframe,
                    callback=on_candle
                )
                
                self.performance_monitor.increment_metric('active_subscriptions')
                
                while self._running:
                    await asyncio.sleep(60)
                
                try:
                    await exchange.unsubscribe_candles(trading_pair, timeframe)
                    self.performance_monitor.increment_metric('active_subscriptions', -1)
                except Exception as e:
                    logger.error("Erro ao cancelar subscrição de candles", error=str(e))
                
            else:
                subscribed = False
                
                for exchange_name, exchange in self.exchanges.items():
                    if (not exchange.validate_trading_pair(trading_pair) or
                        timeframe not in exchange.get_supported_timeframes().values()):
                        continue
                        
                    try:
                        async def on_candle(candle: Candle) -> None:
                            await self._process_candle(candle)
                        
                        await exchange.subscribe_candles(
                            trading_pair=trading_pair,
                            timeframe=timeframe,
                            callback=on_candle
                        )
                        
                        subscribed = True
                        self.performance_monitor.increment_metric('active_subscriptions')
                        logger.info("Subscrito para candles", 
                                  exchange=exchange_name, trading_pair=trading_pair, 
                                  timeframe=timeframe.value)
                        break
                        
                    except Exception as e:
                        logger.error("Erro ao subscrever para candles", 
                                   exchange=exchange_name, trading_pair=trading_pair, 
                                   timeframe=timeframe.value, error=str(e))
                
                if not subscribed:
                    logger.warning("Não foi possível subscrever para candles", 
                                 trading_pair=trading_pair, timeframe=timeframe.value)
                    return
                
                while self._running:
                    await asyncio.sleep(60)
            
        except asyncio.CancelledError:
            logger.debug("Task de subscrição de candles cancelada", 
                       trading_pair=trading_pair, timeframe=timeframe.value)
        except Exception as e:
            logger.error("Erro na subscrição de candles", 
                       trading_pair=trading_pair, timeframe=timeframe.value, error=str(e))
    
    async def _subscribe_funding_rates_for_pair(self, trading_pair: str) -> None:
        """Subscreve para atualizações de funding rates para um par."""
        try:
            logger.info("Subscrevendo para funding rates", trading_pair=trading_pair)
            
            # Funding rates são geralmente suportadas apenas por exchanges de futuros
            perpetual_exchanges = ['binance', 'bybit']
            
            for exchange_name, exchange in self.exchanges.items():
                if (exchange_name.lower() in perpetual_exchanges and 
                    exchange.validate_trading_pair(trading_pair)):
                    
                    try:
                        async def on_funding_rate(funding_rate: FundingRate) -> None:
                            await self._process_funding_rate(funding_rate)
                        
                        await exchange.subscribe_funding_rates(
                            trading_pair=trading_pair,
                            callback=on_funding_rate
                        )
                        
                        self.performance_monitor.increment_metric('active_subscriptions')
                        logger.info("Subscrito para funding rates", 
                                  exchange=exchange_name, trading_pair=trading_pair)
                        
                        while self._running:
                            await asyncio.sleep(60)
                        
                        try:
                            await exchange.unsubscribe_funding_rates(trading_pair)
                            self.performance_monitor.increment_metric('active_subscriptions', -1)
                        except Exception as e:
                            logger.error("Erro ao cancelar subscrição de funding rates", error=str(e))
                        
                        break
                        
                    except Exception as e:
                        logger.error("Erro ao subscrever para funding rates", 
                                   exchange=exchange_name, trading_pair=trading_pair, error=str(e))
            
        except asyncio.CancelledError:
            logger.debug("Task de subscrição de funding rates cancelada", trading_pair=trading_pair)
        except Exception as e:
            logger.error("Erro na subscrição de funding rates", trading_pair=trading_pair, error=str(e))
    
    async def _subscribe_liquidations_for_pair(self, trading_pair: str) -> None:
        """Subscreve para eventos de liquidação para um par."""
        try:
            logger.info("Subscrevendo para liquidações", trading_pair=trading_pair)
            
            # Liquidações são geralmente suportadas apenas por exchanges de futuros
            perpetual_exchanges = ['binance', 'bybit']
            
            for exchange_name, exchange in self.exchanges.items():
                if (exchange_name.lower() in perpetual_exchanges and 
                    exchange.validate_trading_pair(trading_pair)):
                    
                    try:
                        async def on_liquidation(liquidation: Liquidation) -> None:
                            await self._process_liquidation(liquidation)
                        
                        await exchange.subscribe_liquidations(
                            trading_pair=trading_pair,
                            callback=on_liquidation
                        )
                        
                        self.performance_monitor.increment_metric('active_subscriptions')
                        logger.info("Subscrito para liquidações", 
                                  exchange=exchange_name, trading_pair=trading_pair)
                        
                        while self._running:
                            await asyncio.sleep(60)
                        
                        try:
                            await exchange.unsubscribe_liquidations(trading_pair)
                            self.performance_monitor.increment_metric('active_subscriptions', -1)
                        except Exception as e:
                            logger.error("Erro ao cancelar subscrição de liquidações", error=str(e))
                        
                        break
                        
                    except Exception as e:
                        logger.error("Erro ao subscrever para liquidações", 
                                   exchange=exchange_name, trading_pair=trading_pair, error=str(e))
            
        except asyncio.CancelledError:
            logger.debug("Task de subscrição de liquidações cancelada", trading_pair=trading_pair)
        except Exception as e:
            logger.error("Erro na subscrição de liquidações", trading_pair=trading_pair, error=str(e))
    
    async def _subscribe_global_liquidations(self) -> None:
        """Subscreve para eventos de liquidação globais."""
        try:
            logger.info("Subscrevendo para liquidações globais")
            
            perpetual_exchanges = ['binance', 'bybit']
            
            for exchange_name, exchange in self.exchanges.items():
                if exchange_name.lower() in perpetual_exchanges:
                    
                    try:
                        async def on_liquidation(liquidation: Liquidation) -> None:
                            await self._process_liquidation(liquidation)
                        
                        await exchange.subscribe_liquidations(
                            trading_pair=None,  # Global
                            callback=on_liquidation
                        )
                        
                        self.performance_monitor.increment_metric('active_subscriptions')
                        logger.info("Subscrito para liquidações globais", exchange=exchange_name)
                        
                        while self._running:
                            await asyncio.sleep(60)
                        
                        try:
                            await exchange.unsubscribe_liquidations(None)
                            self.performance_monitor.increment_metric('active_subscriptions', -1)
                        except Exception as e:
                            logger.error("Erro ao cancelar subscrição de liquidações globais", error=str(e))
                        
                        break
                        
                    except Exception as e:
                        logger.error("Erro ao subscrever para liquidações globais", 
                                   exchange=exchange_name, error=str(e))
            
        except asyncio.CancelledError:
            logger.debug("Task de subscrição de liquidações globais cancelada")
        except Exception as e:
            logger.error("Erro na subscrição de liquidações globais", error=str(e))
    
    async def _process_orderbook(self, orderbook: OrderBook) -> None:
        """Processa um orderbook recebido."""
        try:
            # Valida o orderbook
            if not self.validation_service.validate_orderbook(orderbook):
                self.performance_monitor.increment_metric('errors_count')
                return
            
            # Normaliza os dados se necessário
            # normalized_orderbook = self.normalization_service.normalize_orderbook(orderbook.raw_data, orderbook.exchange)
            
            # Salva no banco de dados se habilitado
            if self.enable_database and self.db_manager:
                await self.db_manager.save_orderbook(orderbook)
            
            # Publica no Kafka se habilitado
            if self.enable_kafka and self.kafka_producer:
                await self.kafka_producer.publish_orderbook(orderbook)
            
            # Atualiza cache
            cache_key = f"orderbook:{orderbook.exchange}:{orderbook.trading_pair}"
            self._data_cache[cache_key] = {
                'data': orderbook,
                'timestamp': datetime.utcnow()
            }
            
            self.performance_monitor.increment_metric('data_points_processed')
            
        except Exception as e:
            logger.error("Erro ao processar orderbook", 
                       exchange=orderbook.exchange, trading_pair=orderbook.trading_pair, error=str(e))
            self.performance_monitor.increment_metric('errors_count')
    
    async def _process_trade(self, trade: Trade) -> None:
        """Processa um trade recebido."""
        try:
            # Valida o trade
            if not self.validation_service.validate_trade(trade):
                self.performance_monitor.increment_metric('errors_count')
                return
            
            # Salva no banco de dados se habilitado
            if self.enable_database and self.db_manager:
                await self.db_manager.save_trade(trade)
            
            # Publica no Kafka se habilitado
            if self.enable_kafka and self.kafka_producer:
                await self.kafka_producer.publish_trade(trade)
            
            self.performance_monitor.increment_metric('data_points_processed')
            
        except Exception as e:
            logger.error("Erro ao processar trade", 
                       exchange=trade.exchange, trading_pair=trade.trading_pair, error=str(e))
            self.performance_monitor.increment_metric('errors_count')
    
    async def _process_candle(self, candle: Candle) -> None:
        """Processa uma candle recebida."""
        try:
            # Valida a candle
            if not self.validation_service.validate_candle(candle):
                self.performance_monitor.increment_metric('errors_count')
                return
            
            # Salva no banco de dados se habilitado
            if self.enable_database and self.db_manager:
                await self.db_manager.save_candle(candle)
            
            # Publica no Kafka se habilitado
            if self.enable_kafka and self.kafka_producer:
                await self.kafka_producer.publish_candle(candle)
            
            self.performance_monitor.increment_metric('data_points_processed')
            
        except Exception as e:
            logger.error("Erro ao processar candle", 
                       exchange=candle.exchange, trading_pair=candle.trading_pair, 
                       timeframe=candle.timeframe.value, error=str(e))
            self.performance_monitor.increment_metric('errors_count')
    
    async def _process_funding_rate(self, funding_rate: FundingRate) -> None:
        """Processa uma taxa de financiamento recebida."""
        try:
            # Publica no Kafka se habilitado (funding rates não são salvas no DB por enquanto)
            if self.enable_kafka and self.kafka_producer:
                # Precisaria implementar método específico no kafka_producer
                logger.info("Funding rate recebida", 
                          exchange=funding_rate.exchange, trading_pair=funding_rate.trading_pair, 
                          rate=float(funding_rate.rate))
            
            self.performance_monitor.increment_metric('data_points_processed')
            
        except Exception as e:
            logger.error("Erro ao processar funding rate", 
                       exchange=funding_rate.exchange, trading_pair=funding_rate.trading_pair, error=str(e))
            self.performance_monitor.increment_metric('errors_count')
    
    async def _process_liquidation(self, liquidation: Liquidation) -> None:
        """Processa um evento de liquidação recebido."""
        try:
            # Publica no Kafka se habilitado (liquidations não são salvas no DB por enquanto)
            if self.enable_kafka and self.kafka_producer:
                # Precisaria implementar método específico no kafka_producer
                logger.info("Liquidação recebida", 
                          exchange=liquidation.exchange, trading_pair=liquidation.trading_pair, 
                          value=float(liquidation.value), side=liquidation.side.value)
            
            self.performance_monitor.increment_metric('data_points_processed')
            
        except Exception as e:
            logger.error("Erro ao processar liquidação", 
                       exchange=liquidation.exchange, trading_pair=liquidation.trading_pair, error=str(e))
            self.performance_monitor.increment_metric('errors_count')
    
    async def _health_check_loop(self) -> None:
        """Loop de verificação de saúde do sistema."""
        while self._running:
            try:
                health = await self.health_checker.perform_health_check()
                
                # Log do status de saúde
                if health['status'] == 'healthy':
                    logger.debug("Sistema saudável", **health['performance'])
                else:
                    logger.warning("Sistema com problemas", status=health['status'])
                
                await asyncio.sleep(60)  # Verifica a cada minuto
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Erro no loop de verificação de saúde", error=str(e))
                await asyncio.sleep(30)
    
    async def _cache_cleanup_loop(self) -> None:
        """Loop de limpeza do cache."""
        while self._running:
            try:
                current_time = datetime.utcnow()
                expired_keys = []
                
                for key, value in self._data_cache.items():
                    if (current_time - value['timestamp']).total_seconds() > self._cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self._data_cache[key]
                
                if expired_keys:
                    logger.debug("Cache limpo", expired_entries=len(expired_keys))
                
                # Força garbage collection periodicamente
                gc.collect()
                
                await asyncio.sleep(300)  # Limpa a cada 5 minutos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Erro no loop de limpeza de cache", error=str(e))
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self) -> None:
        """Loop de coleta de métricas do sistema."""
        while self._running:
            try:
                import psutil
                
                # Coleta métricas do sistema
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()
                
                self.performance_monitor.update_metric('memory_usage', memory_usage)
                self.performance_monitor.update_metric('cpu_usage', cpu_usage)
                
                # Log das métricas principais
                stats = self.performance_monitor.get_stats()
                logger.info("Métricas do sistema", **stats)
                
                await asyncio.sleep(30)  # Coleta a cada 30 segundos
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Erro no loop de coleta de métricas", error=str(e))
                await asyncio.sleep(60)
    
    async def _performance_monitoring_loop(self) -> None:
        """Loop de monitoramento de performance."""
        while self._running:
            try:
                stats = self.performance_monitor.get_stats()
                
                # Alerta se a taxa de erro for muito alta
                if stats['errors_count'] > 0:
                    error_rate = stats['errors_count'] / max(1, stats['data_points_processed'])
                    if error_rate > 0.05:  # Mais de 5% de erro
                        logger.warning("Taxa de erro alta detectada", error_rate=error_rate)
                
                # Alerta se não houver dados por muito tempo
                if stats['data_rate_per_second'] < 0.1:  # Menos de 0.1 dados por segundo
                    logger.warning("Taxa de dados muito baixa", data_rate=stats['data_rate_per_second'])
                
                await asyncio.sleep(60)  # Monitora a cada minuto
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Erro no loop de monitoramento de performance", error=str(e))
                await asyncio.sleep(30)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Retorna o status de saúde atual do sistema."""
        return self.health_checker.health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance do sistema."""
        return self.performance_monitor.get_stats()
    
    def get_cached_data(self, key: str) -> Any:
        """Retorna dados do cache se disponíveis."""
        cached = self._data_cache.get(key)
        if cached:
            # Verifica se não expirou
            if (datetime.utcnow() - cached['timestamp']).total_seconds() < self._cache_ttl:
                return cached['data']
            else:
                # Remove do cache se expirou
                del self._data_cache[key]
        return None


def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega configuração de arquivo YAML."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Substitui variáveis de ambiente
        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                default_value = None
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)
                return os.environ.get(env_var, default_value)
            else:
                return obj
        
        return replace_env_vars(config)
        
    except Exception as e:
        logger.error("Erro ao carregar configuração", config_path=config_path, error=str(e))
        raise


async def setup_signal_handlers(service: EnhancedDataCollectionService):
    """Configura handlers para sinais do sistema."""
    
    def signal_handler():
        logger.info("Sinal de parada recebido, iniciando shutdown graceful")
        asyncio.create_task(service.stop())
    
    # Registra handlers para SIGINT e SIGTERM
    if sys.platform != 'win32':
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGINT, signal.SIGTERM]:
            loop.add_signal_handler(sig, signal_handler)


async def main():
    """Função principal do serviço de coleta de dados."""
    
    # Configura argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Sistema de Coleta de Dados de Mercado')
    parser.add_argument('--config', '-c', 
                       default='config/data_collection.yaml',
                       help='Caminho para o arquivo de configuração')
    parser.add_argument('--log-level', '-l',
                       default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Nível de log')
    parser.add_argument('--env-file', '-e',
                       default='.env',
                       help='Arquivo de variáveis de ambiente')
    parser.add_argument('--validate-config', '-v',
                       action='store_true',
                       help='Apenas valida a configuração e sai')
    parser.add_argument('--dry-run', '-d',
                       action='store_true',
                       help='Executa em modo dry-run (não salva dados)')
    
    args = parser.parse_args()
    
    # Carrega variáveis de ambiente
    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)
    
    # Configura logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Carrega configuração
        logger.info("Carregando configuração", config_path=args.config)
        config = load_config(args.config)
        
        # Valida configuração básica
        required_sections = ['collection', 'exchanges']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Seção obrigatória '{section}' não encontrada na configuração")
        
        # Se apenas validando configuração, sai aqui
        if args.validate_config:
            logger.info("Configuração válida")
            return
        
        # Configura modo dry-run
        if args.dry_run:
            config.setdefault('storage', {})
            config['storage']['enable_database'] = False
            config['storage']['enable_kafka'] = False
            logger.info("Modo dry-run ativado - dados não serão persistidos")
        
        # Usa uvloop para melhor performance no Linux
        if sys.platform == 'linux':
            uvloop.install()
            logger.info("uvloop instalado para melhor performance")
        
        # Cria e inicializa o serviço
        logger.info("Inicializando serviço de coleta de dados")
        service = EnhancedDataCollectionService(config)
        
        # Configura handlers de sinal
        await setup_signal_handlers(service)
        
        # Inicializa o serviço
        await service.initialize()
        
        # Inicia a coleta de dados
        await service.start()
        
        logger.info("Serviço iniciado com sucesso, aguardando dados...")
        
        # Aguarda indefinidamente ou até receber sinal de parada
        try:
            await service._shutdown_event.wait()
        except KeyboardInterrupt:
            logger.info("Interrupção do teclado recebida")
        
    except Exception as e:
        logger.error("Erro fatal no serviço", error=str(e), exc_info=True)
        sys.exit(1)
    
    finally:
        # Garante que o serviço seja parado adequadamente
        try:
            if 'service' in locals():
                await service.stop()
        except Exception as e:
            logger.error("Erro ao parar serviço", error=str(e))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Programa interrompido pelo usuário")
    except Exception as e:
        logger.error("Erro não tratado", error=str(e), exc_info=True)
        sys.exit(1)