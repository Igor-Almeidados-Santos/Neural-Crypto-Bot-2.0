"""
Módulo principal estendido para coleta de dados de mercado.

Este módulo implementa o serviço de coleta de dados com funcionalidades avançadas:
- Suporte para múltiplas exchanges (Binance, Coinbase, Kraken, Bybit)
- Coleta de dados adicionais (funding rates, liquidações)
- Criptografia de dados sensíveis
- Compressão de dados
- Balanceamento de carga para distribuir requisições
"""
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable

import argparse
from dotenv import load_dotenv

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

from src.data_collection.domain.repositories.candle_repository import CandleRepository
from src.data_collection.domain.repositories.orderbook_repository import OrderBookRepository
from src.data_collection.domain.repositories.funding_rate_repository import FundingRateRepository
from src.data_collection.domain.repositories.liquidation_repository import LiquidationRepository

from src.data_collection.infrastructure.database import DatabaseManager
from src.data_collection.infrastructure.kafka_producer import KafkaProducer
from src.data_collection.infrastructure.crypto import CryptoService
from src.data_collection.infrastructure.compression import CompressionService
from src.data_collection.infrastructure.load_balancer import ExchangeLoadBalancer, BalancingStrategy


# Configuração do logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_collection.log')
    ]
)
logger = logging.getLogger(__name__)


class EnhancedDataCollectionService:
    """
    Serviço avançado para coleta de dados de mercado.
    
    Estende o serviço básico com funcionalidades adicionais como
    suporte a múltiplas exchanges, tipos de dados adicionais,
    criptografia, compressão e balanceamento de carga.
    """
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        Inicializa o serviço de coleta de dados.
        
        Args:
            config: Configurações do serviço
        """
        self.config = config
        
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
        
        # Tasks em execução
        self._tasks = []
        
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
    
    async def initialize(self) -> None:
        """
        Inicializa todos os componentes do serviço.
        """
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
                
                self._initialized = True
                logger.info("Serviço avançado de coleta de dados inicializado")
                
            except Exception as e:
                logger.error(f"Erro ao inicializar serviço de coleta de dados: {str(e)}", exc_info=True)
                raise
    
    async def start(self) -> None:
        """
        Inicia a coleta de dados.
        """
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
    
    async def stop(self) -> None:
        """
        Para a coleta de dados.
        """
        if not self._running:
            return
            
        logger.info("Parando coleta de dados")
        self._running = False
        
        # Cancela todas as tasks em execução
        for task in self._tasks:
            if not task.done():
                task.cancel()
                
        # Aguarda o cancelamento das tasks
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
        self._tasks = []
        
        # Para o balanceador de carga
        if self.exchange_load_balancer:
            await self.exchange_load_balancer.stop()
        
        # Fecha os adapters de exchanges
        for exchange_name, exchange in self.exchanges.items():
            try:
                logger.info(f"Fechando adapter da exchange {exchange_name}")
                await exchange.shutdown()
            except Exception as e:
                logger.error(f"Erro ao fechar adapter da exchange {exchange_name}: {str(e)}", exc_info=True)
        
        # Fecha a infraestrutura
        if self.db_manager and self.enable_database:
            await self.db_manager.close()
            
        if self.kafka_producer and self.enable_kafka:
            await self.kafka_producer.close()
            
        logger.info("Coleta de dados parada")
    
    async def _initialize_infrastructure(self) -> None:
        """
        Inicializa a infraestrutura (banco de dados e Kafka).
        """
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
        """
        Inicializa os adapters de exchanges.
        """
        exchange_adapters = []
        
        for exchange_name in self.collection_exchanges:
            try:
                logger.info(f"Inicializando adapter da exchange {exchange_name}")
                
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
                    logger.warning(f"Exchange não suportada: {exchange_name}")
                    continue
                
                # Inicializa o adapter
                await adapter.initialize()
                
                # Adiciona à lista de exchanges
                self.exchanges[exchange_name] = adapter
                exchange_adapters.append(adapter)
                
                logger.info(f"Adapter da exchange {exchange_name} inicializado")
                
            except Exception as e:
                logger.error(f"Erro ao inicializar adapter da exchange {exchange_name}: {str(e)}", exc_info=True)
    
    async def _initialize_load_balancer(self) -> None:
        """
        Inicializa o balanceador de carga para as exchanges.
        """
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
                circuit_breaker_threshold=5,
                max_retries=3,
                rate_limit_buffer=self.rate_limit_buffer
            )
            
            logger.info(f"Balanceador de carga inicializado com estratégia {strategy}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar balanceador de carga: {str(e)}", exc_info=True)
            self.exchange_load_balancer = None
    
    async def _sync_historical_data(self) -> None:
        """
        Sincroniza dados históricos para todos os pares e timeframes configurados.
        """
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
        """
        Sincroniza dados históricos de candles.
        
        Args:
            start_date: Data de início para sincronização
        """
        logger.info("Sincronizando dados históricos de candles")
        
        # Para cada par e timeframe
        for trading_pair in self.collection_pairs:
            for timeframe_str in self.collection_timeframes:
                try:
                    # Converte o timeframe para o formato padronizado
                    timeframe = TimeFrame(timeframe_str)
                    
                    logger.info(f"Sincronizando candles para {trading_pair}:{timeframe}")
                    
                    # Se o balanceador de carga estiver habilitado, usa-o para distribuir as requisições
                    if self.exchange_load_balancer:
                        try:
                            async def fetch_candles_operation(exchange):
                                if not exchange.validate_trading_pair(trading_pair):
                                    return []
                                    
                                # Verifica se o timeframe é suportado
                                if timeframe.value not in exchange.get_supported_timeframes().values():
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
                            logger.error(f"Erro ao sincronizar candles via balanceador para {trading_pair}:{timeframe}: {str(e)}")
                            candles = []
                            
                    else:
                        # Tenta cada exchange até obter dados
                        candles = []
                        for exchange_name, exchange in self.exchanges.items():
                            if not exchange.validate_trading_pair(trading_pair):
                                continue
                                
                            # Verifica se o timeframe é suportado
                            if timeframe.value not in exchange.get_supported_timeframes().values():
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
                                logger.error(f"Erro ao sincronizar candles para {exchange_name}:{trading_pair}:{timeframe}: {str(e)}")
                    
                    # Valida e normaliza as candles
                    valid_candles = []
                    for candle in candles:
                        try:
                            if self.validation_service.validate_candle(candle):
                                valid_candles.append(candle)
                        except Exception as e:
                            logger.warning(f"Erro ao validar candle: {str(e)}")
                    
                    if valid_candles:
                        # Salva as candles no banco de dados
                        if self.enable_database and self.db_manager:
                            # Comprime os dados se habilitado
                            if self.enable_compression:
                                # Comprime os dados antes de salvar
                                for candle in valid_candles:
                                    if candle.raw_data:
                                        raw_data_bytes = str(candle.raw_data).encode('utf-8')
                                        if self.compression_service.should_compress(raw_data_bytes, self.compression_threshold):
                                            compressed_data = self.compression_service.compress_efficient(raw_data_bytes, self.compression_threshold)
                                            candle.raw_data = compressed_data
                            
                            await self.db_manager.save_candles_batch(valid_candles)
                            
                        # Publica no Kafka se habilitado
                        if self.enable_kafka and self.kafka_producer:
                            await self.kafka_producer.publish_candles_batch(valid_candles)
                            
                        logger.info(f"Sincronizados {len(valid_candles)} candles para {trading_pair}:{timeframe}")
                    else:
                        logger.warning(f"Nenhum candle válido encontrado para {trading_pair}:{timeframe}")
                    
                except Exception as e:
                    logger.error(f"Erro ao sincronizar candles para {trading_pair}:{timeframe}: {str(e)}", exc_info=True)
    
    async def _sync_historical_funding_rates(self, start_date: datetime) -> None:
        """
        Sincroniza dados históricos de taxas de financiamento.
        
        Args:
            start_date: Data de início para sincronização
        """
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
                        logger.error(f"Erro ao sincronizar funding rates via balanceador para {trading_pair}: {str(e)}")
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
                            logger.error(f"Erro ao sincronizar funding rates para {exchange_name}:{trading_pair}: {str(e)}")
                
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
                                        rate.raw_data = compressed_data
                        
                        # Aqui precisaríamos usar um repositório específico para funding rates
                        # Por simplicidade, estamos apenas logando
                        logger.info(f"Salvariam {len(valid_funding_rates)} taxas de financiamento para {trading_pair}")
                        
                    # Publica no Kafka se habilitado
                    if self.enable_kafka and self.kafka_producer:
                        # Aqui precisaríamos usar um tópico específico para funding rates
                        # Por simplicidade, estamos apenas logando
                        logger.info(f"Publicariam {len(valid_funding_rates)} taxas de financiamento para {trading_pair}")
                        
                    logger.info(f"Sincronizados {len(valid_funding_rates)} taxas de financiamento para {trading_pair}")
                else:
                    logger.warning(f"Nenhuma taxa de financiamento válida encontrada para {trading_pair}")
                
            except Exception as e:
                logger.error(f"Erro ao sincronizar taxas de financiamento para {trading_pair}: {str(e)}", exc_info=True)
    
    async def _start_realtime_collection(self) -> None:
        """
        Inicia a coleta de dados em tempo real.
        """
        logger.info("Iniciando coleta de dados em tempo real")
        
        # Para cada par de negociação
        for trading_pair in self.collection_pairs:
            # Subscrita para orderbooks
            task = asyncio.create_task(
                self._subscribe_orderbook_for_pair(trading_pair)
            )
            self._tasks.append(task)
            
            # Subscrita para trades
            task = asyncio.create_task(
                self._subscribe_trades_for_pair(trading_pair)
            )
            self._tasks.append(task)
            
            # Subscrita para candles de cada timeframe
            for timeframe_str in self.collection_timeframes:
                try:
                    timeframe = TimeFrame(timeframe_str)
                    
                    task = asyncio.create_task(
                        self._subscribe_candles_for_pair(trading_pair, timeframe)
                    )
                    self._tasks.append(task)
                    
                except Exception as e:
                    logger.error(f"Erro ao iniciar coleta de candles para {trading_pair}:{timeframe_str}: {str(e)}")
            
            # Subscrita para funding rates se habilitado
            if self.collect_funding_rates:
                task = asyncio.create_task(
                    self._subscribe_funding_rates_for_pair(trading_pair)
                )
                self._tasks.append(task)
            
            # Subscrita para liquidações se habilitado
            if self.collect_liquidations:
                task = asyncio.create_task(
                    self._subscribe_liquidations_for_pair(trading_pair)
                )
                self._tasks.append(task)
        
        # Subscrita para liquidações globais se habilitado
        if self.collect_liquidations:
            task = asyncio.create_task(
                self._subscribe_global_liquidations()
            )
            self._tasks.append(task)
        
        logger.info("Coleta de dados em tempo real iniciada")
    
    async def _subscribe_orderbook_for_pair(self, trading_pair: str) -> None:
        """
        Subscreve para atualizações de orderbook para um par de negociação.
        
        Args:
            trading_pair: Par de negociação
        """
        try:
            logger.info(f"Subscrevendo para orderbooks de {trading_pair}")
            
            # Se o balanceador de carga estiver habilitado, usa-o para distribuir as requisições
            if self.exchange_load_balancer:
                # Determina quais exchanges suportam o par
                available_exchanges = []
                for exchange in self.exchange_load_balancer.get_available_instances():
                    if exchange.validate_trading_pair(trading_pair):
                        available_exchanges.append(exchange)
                
                if not available_exchanges:
                    logger.warning(f"Nenhuma exchange disponível para o par {trading_pair}")
                    return
                
                # Seleciona uma exchange para subscrever
                exchange = available_exchanges[0]  # Simplificado; o balanceador poderia fazer uma seleção melhor
                
                # Define o callback para processar orderbooks
                async def on_orderbook(orderbook: OrderBook) -> None:
                    await self._process_orderbook(orderbook)
                
                # Subscreve para atualizações
                await exchange.subscribe_orderbook(
                    trading_pair=trading_pair,
                    callback=on_orderbook
                )
                
                # Mantém a subscrição ativa enquanto o serviço estiver rodando
                while self._running:
                    await asyncio.sleep(60)  # Verifica a cada minuto
                
                # Cancela a subscrição quando a task for cancelada
                if self._running:  # Só loga se não for um desligamento normal
                    logger.info(f"Cancelando subscrição de orderbook para {exchange.name}:{trading_pair}")
                
                try:
                    await exchange.unsubscribe_orderbook(trading_pair)
                except Exception as e:
                    logger.error(f"Erro ao cancelar subscrição de orderbook: {str(e)}")
                
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
                        logger.info(f"Subscrito para orderbooks de {exchange_name}:{trading_pair}")
                        break
                        
                    except Exception as e:
                        logger.error(f"Erro ao subscrever para orderbooks de {exchange_name}:{trading_pair}: {str(e)}")
                
                if not subscribed:
                    logger.warning(f"Não foi possível subscrever para orderbooks de nenhuma exchange para {trading_pair}")
                    return"""
Módulo principal estendido para coleta de dados de mercado.

Este módulo implementa o serviço de coleta de dados com funcionalidades avançadas:
- Suporte para múltiplas exchanges (Binance, Coinbase, Kraken, Bybit)
- Coleta de dados adicionais (funding rates, liquidações)
- Criptografia de dados sensíveis
- Compressão de dados
- Balanceamento de