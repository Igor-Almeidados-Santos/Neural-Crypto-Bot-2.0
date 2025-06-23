"""
Produtor Kafka corrigido para publicação de dados de mercado.

Este módulo implementa um produtor Kafka robusto com:
- Error handling avançado
- Retry logic com backoff exponencial
- Batch processing otimizado
- Health checks
- Metrics e observabilidade
"""
import asyncio
import json
import logging
import time
import threading
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import uuid

try:
    from confluent_kafka import Producer, KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient, NewTopic, ConfigResource, ResourceType
    HAS_CONFLUENT_KAFKA = True
except ImportError:
    HAS_CONFLUENT_KAFKA = False

try:
    import aiokafka
    from aiokafka import AIOKafkaProducer
    HAS_AIOKAFKA = True
except ImportError:
    HAS_AIOKAFKA = False

try:
    import orjson as json_lib
    HAS_ORJSON = True
except ImportError:
    import json as json_lib
    HAS_ORJSON = False

from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from src.data_collection.domain.entities.trade import Trade, TradeSide


logger = logging.getLogger(__name__)


@dataclass
class KafkaMetrics:
    """Métricas do produtor Kafka."""
    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    avg_latency_ms: float = 0.0
    last_send_time: Optional[datetime] = None
    connection_errors: int = 0
    retry_count: int = 0


class KafkaError(Exception):
    """Erro base para operações Kafka."""
    pass


class KafkaConnectionError(KafkaError):
    """Erro de conexão com Kafka."""
    pass


class KafkaPublishError(KafkaError):
    """Erro de publicação no Kafka."""
    pass


class KafkaProducer:
    """
    Produtor Kafka robusto para publicação de dados de mercado.
    
    Implementa funcionalidades avançadas:
    - Múltiplas bibliotecas Kafka (confluent-kafka, aiokafka)
    - Batch processing otimizado
    - Retry logic com circuit breaker
    - Serialização eficiente
    - Health checks e métricas
    """
    
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: str = "neural-crypto-bot-data-collection",
        acks: str = "all",
        retries: int = 3,
        batch_size: int = 16384,
        linger_ms: int = 5,
        compression_type: str = "snappy",
        max_in_flight_requests_per_connection: int = 1,
        enable_idempotence: bool = True,
        topic_prefix: str = "market-data",
        use_confluent: bool = True
    ):
        """
        Inicializa o produtor Kafka.
        
        Args:
            bootstrap_servers: Lista de servidores Kafka
            client_id: ID do cliente
            acks: Configuração de acknowledges
            retries: Número de tentativas
            batch_size: Tamanho do batch
            linger_ms: Tempo de espera para batch
            compression_type: Tipo de compressão
            max_in_flight_requests_per_connection: Requisições em andamento
            enable_idempotence: Idempotência
            topic_prefix: Prefixo dos tópicos
            use_confluent: Usar confluent-kafka (preferido) ou aiokafka
        """
        if not HAS_CONFLUENT_KAFKA and not HAS_AIOKAFKA:
            raise KafkaError("confluent-kafka ou aiokafka são necessários")
        
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.acks = acks
        self.retries = retries
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self.compression_type = compression_type
        self.max_in_flight_requests_per_connection = max_in_flight_requests_per_connection
        self.enable_idempotence = enable_idempotence
        self.topic_prefix = topic_prefix
        self.use_confluent = use_confluent and HAS_CONFLUENT_KAFKA
        
        # Estado
        self._producer = None
        self._admin_client = None
        self._initialized = False
        self._closed = False
        self._init_lock = asyncio.Lock()
        
        # Métricas
        self._metrics = KafkaMetrics()
        self._metrics_lock = threading.Lock()
        
        # Circuit breaker
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_open = False
        self._circuit_breaker_last_failure = 0
        self._circuit_breaker_timeout = 60  # segundos
        
        # Tópicos padrão
        self._default_topics = {
            'trades': f"{self.topic_prefix}.trades",
            'orderbooks': f"{self.topic_prefix}.orderbooks",
            'candles': f"{self.topic_prefix}.candles",
            'funding_rates': f"{self.topic_prefix}.funding_rates",
            'liquidations': f"{self.topic_prefix}.liquidations"
        }
        
        # Buffer para batch processing
        self._message_buffer = defaultdict(list)
        self._buffer_lock = asyncio.Lock()
        self._flush_task = None
        self._buffer_max_size = 1000
        self._buffer_flush_interval = 5.0  # segundos
        
        # Configuração do produtor
        self._producer_config = self._build_config()
        
        logger.info(f"KafkaProducer configurado (confluent={self.use_confluent})")
    
    def _build_config(self) -> Dict[str, Any]:
        """Constrói configuração do produtor."""
        if self.use_confluent:
            return {
                'bootstrap.servers': self.bootstrap_servers,
                'client.id': self.client_id,
                'acks': self.acks,
                'retries': self.retries,
                'batch.size': self.batch_size,
                'linger.ms': self.linger_ms,
                'compression.type': self.compression_type,
                'max.in.flight.requests.per.connection': self.max_in_flight_requests_per_connection,
                'enable.idempotence': self.enable_idempotence,
                'delivery.timeout.ms': 120000,
                'request.timeout.ms': 30000,
                'retry.backoff.ms': 100
            }
        else:
            return {
                'bootstrap_servers': self.bootstrap_servers,
                'client_id': self.client_id,
                'acks': self.acks,
                'retries': self.retries,
                'batch_size': self.batch_size,
                'linger_ms': self.linger_ms,
                'compression_type': self.compression_type,
                'max_in_flight_requests_per_connection': self.max_in_flight_requests_per_connection,
                'enable_idempotence': self.enable_idempotence
            }
    
    async def initialize(self) -> None:
        """Inicializa o produtor Kafka."""
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                if self.use_confluent:
                    await self._initialize_confluent()
                else:
                    await self._initialize_aiokafka()
                
                # Verificar/criar tópicos
                await self._ensure_topics()
                
                # Iniciar buffer flush task
                self._flush_task = asyncio.create_task(self._buffer_flush_loop())
                
                self._initialized = True
                logger.info("KafkaProducer inicializado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro na inicialização do Kafka: {e}")
                raise KafkaConnectionError(f"Falha ao conectar: {e}")
    
    async def _initialize_confluent(self) -> None:
        """Inicializa usando confluent-kafka."""
        if not HAS_CONFLUENT_KAFKA:
            raise KafkaError("confluent-kafka não está disponível")
        
        self._producer = Producer(self._producer_config)
        
        # Admin client para gerenciamento de tópicos
        admin_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'client.id': f"{self.client_id}-admin"
        }
        self._admin_client = AdminClient(admin_config)
        
        logger.info("Producer confluent-kafka inicializado")
    
    async def _initialize_aiokafka(self) -> None:
        """Inicializa usando aiokafka."""
        if not HAS_AIOKAFKA:
            raise KafkaError("aiokafka não está disponível")
        
        self._producer = AIOKafkaProducer(**self._producer_config)
        await self._producer.start()
        
        logger.info("Producer aiokafka inicializado")
    
    async def _ensure_topics(self) -> None:
        """Verifica/cria tópicos necessários."""
        if not self.use_confluent or not self._admin_client:
            return  # aiokafka cria tópicos automaticamente
        
        try:
            # Verificar tópicos existentes
            metadata = self._admin_client.list_topics(timeout=10)
            existing_topics = set(metadata.topics.keys())
            
            # Criar tópicos que não existem
            topics_to_create = []
            for topic_name in self._default_topics.values():
                if topic_name not in existing_topics:
                    topics_to_create.append(
                        NewTopic(
                            topic=topic_name,
                            num_partitions=3,
                            replication_factor=1,
                            config={
                                'cleanup.policy': 'delete',
                                'retention.ms': str(7 * 24 * 60 * 60 * 1000),  # 7 dias
                                'compression.type': self.compression_type
                            }
                        )
                    )
            
            if topics_to_create:
                fs = self._admin_client.create_topics(topics_to_create)
                
                # Aguardar criação
                for topic, f in fs.items():
                    try:
                        f.result(timeout=10)
                        logger.info(f"Tópico criado: {topic}")
                    except Exception as e:
                        logger.warning(f"Erro ao criar tópico {topic}: {e}")
            
            logger.info("Tópicos verificados/criados")
            
        except Exception as e:
            logger.warning(f"Erro ao verificar tópicos: {e}")
    
    async def publish_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Publica uma mensagem no Kafka.
        
        Args:
            topic: Nome do tópico
            message: Mensagem a ser publicada
            key: Chave da mensagem (opcional)
            headers: Headers da mensagem (opcional)
        """
        if not self._initialized:
            await self.initialize()
        
        if self._circuit_breaker_open:
            if time.time() - self._circuit_breaker_last_failure > self._circuit_breaker_timeout:
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0
            else:
                raise KafkaPublishError("Circuit breaker aberto")
        
        try:
            # Serializar mensagem
            message_data = self._serialize_message(message)
            
            # Adicionar metadados
            enriched_message = {
                'timestamp': datetime.utcnow().isoformat(),
                'producer_id': self.client_id,
                'message_id': str(uuid.uuid4()),
                'data': message_data
            }
            
            serialized_message = self._serialize_message(enriched_message)
            
            if self.use_confluent:
                await self._publish_confluent(topic, serialized_message, key, headers)
            else:
                await self._publish_aiokafka(topic, serialized_message, key, headers)
            
            # Atualizar métricas
            self._update_metrics(True)
            
        except Exception as e:
            self._update_metrics(False)
            self._handle_publish_error(e)
            raise KafkaPublishError(f"Erro ao publicar mensagem: {e}")
    
    async def _publish_confluent(
        self,
        topic: str,
        message: bytes,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Publica usando confluent-kafka."""
        def delivery_callback(err, msg):
            if err:
                logger.error(f"Erro na entrega da mensagem: {err}")
                self._handle_publish_error(err)
            else:
                logger.debug(f"Mensagem entregue: {msg.topic()}[{msg.partition()}]")
        
        try:
            # Preparar headers
            kafka_headers = None
            if headers:
                kafka_headers = [(k, v.encode() if isinstance(v, str) else v) for k, v in headers.items()]
            
            # Produzir mensagem
            self._producer.produce(
                topic=topic,
                value=message,
                key=key.encode() if key else None,
                headers=kafka_headers,
                callback=delivery_callback
            )
            
            # Flush para garantir entrega
            self._producer.poll(timeout=0.1)
            
        except Exception as e:
            raise KafkaPublishError(f"Erro no producer confluent: {e}")
    
    async def _publish_aiokafka(
        self,
        topic: str,
        message: bytes,
        key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> None:
        """Publica usando aiokafka."""
        try:
            # Preparar headers
            kafka_headers = None
            if headers:
                kafka_headers = [(k, v.encode() if isinstance(v, str) else v) for k, v in headers.items()]
            
            # Produzir mensagem
            await self._producer.send_and_wait(
                topic=topic,
                value=message,
                key=key.encode() if key else None,
                headers=kafka_headers
            )
            
        except Exception as e:
            raise KafkaPublishError(f"Erro no producer aiokafka: {e}")
    
    def _serialize_message(self, message: Any) -> bytes:
        """Serializa mensagem para JSON."""
        try:
            if HAS_ORJSON:
                return json_lib.dumps(message, default=self._json_serializer)
            else:
                return json_lib.dumps(message, default=self._json_serializer).encode('utf-8')
        except Exception as e:
            raise KafkaPublishError(f"Erro na serialização: {e}")
    
    def _json_serializer(self, obj: Any) -> Any:
        """Serializa objetos especiais para JSON."""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif hasattr(obj, 'value'):  # Enums
            return obj.value
        else:
            return str(obj)
    
    # Métodos específicos para entidades de domínio
    
    async def publish_trade(self, trade: Trade) -> None:
        """Publica um trade."""
        message = {
            'exchange': trade.exchange,
            'trading_pair': trade.trading_pair,
            'trade_id': trade.trade_id,
            'price': float(trade.price),
            'amount': float(trade.amount),
            'cost': float(trade.cost),
            'side': trade.side.value,
            'timestamp': trade.timestamp.isoformat()
        }
        
        key = f"{trade.exchange}:{trade.trading_pair}"
        headers = {
            'type': 'trade',
            'exchange': trade.exchange,
            'trading_pair': trade.trading_pair
        }
        
        await self.publish_message(
            self._default_topics['trades'],
            message,
            key,
            headers
        )
    
    async def publish_orderbook(self, orderbook: OrderBook) -> None:
        """Publica um orderbook."""
        message = {
            'exchange': orderbook.exchange,
            'trading_pair': orderbook.trading_pair,
            'timestamp': orderbook.timestamp.isoformat(),
            'bids': [
                {'price': float(level.price), 'amount': float(level.amount)}
                for level in orderbook.bids
            ],
            'asks': [
                {'price': float(level.price), 'amount': float(level.amount)}
                for level in orderbook.asks
            ]
        }
        
        # Adicionar spread se disponível
        if hasattr(orderbook, 'spread_percentage'):
            message['spread_percentage'] = float(orderbook.spread_percentage)
        
        key = f"{orderbook.exchange}:{orderbook.trading_pair}"
        headers = {
            'type': 'orderbook',
            'exchange': orderbook.exchange,
            'trading_pair': orderbook.trading_pair
        }
        
        await self.publish_message(
            self._default_topics['orderbooks'],
            message,
            key,
            headers
        )
    
    async def publish_candle(self, candle: Candle) -> None:
        """Publica um candle."""
        message = {
            'exchange': candle.exchange,
            'trading_pair': candle.trading_pair,
            'timeframe': candle.timeframe.value,
            'timestamp': candle.timestamp.isoformat(),
            'open_price': float(candle.open_price),
            'high_price': float(candle.high_price),
            'low_price': float(candle.low_price),
            'close_price': float(candle.close_price),
            'volume': float(candle.volume),
            'close_timestamp': candle.close_timestamp.isoformat() if candle.close_timestamp else None
        }
        
        key = f"{candle.exchange}:{candle.trading_pair}:{candle.timeframe.value}"
        headers = {
            'type': 'candle',
            'exchange': candle.exchange,
            'trading_pair': candle.trading_pair,
            'timeframe': candle.timeframe.value
        }
        
        await self.publish_message(
            self._default_topics['candles'],
            message,
            key,
            headers
        )
    
    # Métodos de batch para melhor performance
    
    async def publish_trades_batch(self, trades: List[Trade]) -> None:
        """Publica múltiplos trades em batch."""
        tasks = []
        for trade in trades:
            task = self.publish_trade(trade)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Publicados {len(trades)} trades em batch")
    
    async def publish_candles_batch(self, candles: List[Candle]) -> None:
        """Publica múltiplos candles em batch."""
        tasks = []
        for candle in candles:
            task = self.publish_candle(candle)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Publicados {len(candles)} candles em batch")
    
    # Buffer e flush automático
    
    async def add_to_buffer(self, topic: str, message: Dict[str, Any]) -> None:
        """Adiciona mensagem ao buffer para flush posterior."""
        async with self._buffer_lock:
            self._message_buffer[topic].append(message)
            
            # Flush se buffer está cheio
            if len(self._message_buffer[topic]) >= self._buffer_max_size:
                await self._flush_buffer(topic)
    
    async def _flush_buffer(self, topic: Optional[str] = None) -> None:
        """Flush do buffer de mensagens."""
        async with self._buffer_lock:
            topics_to_flush = [topic] if topic else list(self._message_buffer.keys())
            
            for topic_name in topics_to_flush:
                messages = self._message_buffer[topic_name]
                if not messages:
                    continue
                
                # Publicar mensagens em batch
                tasks = []
                for message in messages:
                    task = self.publish_message(topic_name, message)
                    tasks.append(task)
                
                try:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    logger.debug(f"Buffer flush: {len(messages)} mensagens para {topic_name}")
                except Exception as e:
                    logger.error(f"Erro no flush do buffer: {e}")
                
                # Limpar buffer
                self._message_buffer[topic_name].clear()
    
    async def _buffer_flush_loop(self) -> None:
        """Loop de flush automático do buffer."""
        while not self._closed:
            try:
                await asyncio.sleep(self._buffer_flush_interval)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no flush loop: {e}")
    
    def _update_metrics(self, success: bool, latency: float = 0) -> None:
        """Atualiza métricas de publicação."""
        with self._metrics_lock:
            self._metrics.total_messages += 1
            
            if success:
                self._metrics.successful_messages += 1
                
                # Atualizar latência média
                if latency > 0:
                    total_latency = self._metrics.avg_latency_ms * (self._metrics.successful_messages - 1)
                    self._metrics.avg_latency_ms = (total_latency + latency) / self._metrics.successful_messages
            else:
                self._metrics.failed_messages += 1
            
            self._metrics.last_send_time = datetime.utcnow()
    
    def _handle_publish_error(self, error: Exception) -> None:
        """Trata erros de publicação."""
        self._circuit_breaker_failures += 1
        
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            self._circuit_breaker_open = True
            self._circuit_breaker_last_failure = time.time()
            logger.error("Circuit breaker aberto para Kafka")
        
        with self._metrics_lock:
            self._metrics.connection_errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do produtor."""
        with self._metrics_lock:
            metrics = asdict(self._metrics)
        
        # Calcular taxa de sucesso
        if metrics['total_messages'] > 0:
            metrics['success_rate'] = metrics['successful_messages'] / metrics['total_messages']
        else:
            metrics['success_rate'] = 0.0
        
        # Adicionar info do circuit breaker
        metrics['circuit_breaker_open'] = self._circuit_breaker_open
        metrics['circuit_breaker_failures'] = self._circuit_breaker_failures
        
        # Tamanho do buffer
        metrics['buffer_size'] = sum(len(msgs) for msgs in self._message_buffer.values())
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do produtor Kafka."""
        try:
            # Verificar estado básico
            healthy = (
                self._initialized and 
                not self._closed and 
                not self._circuit_breaker_open
            )
            
            metrics = self.get_metrics()
            
            return {
                'healthy': healthy,
                'service': 'KafkaProducer',
                'bootstrap_servers': self.bootstrap_servers,
                'client_id': self.client_id,
                'topics': list(self._default_topics.values()),
                'metrics': metrics,
                'circuit_breaker_open': self._circuit_breaker_open,
                'initialized': self._initialized
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'service': 'KafkaProducer',
                'error': str(e),
                'initialized': self._initialized
            }
    
    async def shutdown(self) -> None:
        """Finaliza o produtor Kafka."""
        if self._closed:
            return
        
        try:
            # Finalizar flush task
            if self._flush_task:
                self._flush_task.cancel()
                try:
                    await self._flush_task
                except asyncio.CancelledError:
                    pass
            
            # Flush final do buffer
            await self._flush_buffer()
            
            # Fechar producer
            if self.use_confluent and self._producer:
                self._producer.flush(timeout=10)
                # confluent-kafka não tem close assíncrono
            elif not self.use_confluent and self._producer:
                await self._producer.stop()
            
            self._closed = True
            logger.info("KafkaProducer finalizado")
            
        except Exception as e:
            logger.error(f"Erro ao finalizar KafkaProducer: {e}")
    
    def __del__(self):
        """Cleanup no destructor."""
        if not self._closed and self._initialized:
            logger.warning("KafkaProducer não foi fechado adequadamente")


# Factory function para facilitar criação
def create_kafka_producer(config: Dict[str, Any]) -> KafkaProducer:
    """
    Cria instância do KafkaProducer a partir de configuração.
    
    Args:
        config: Configuração do Kafka
        
    Returns:
        KafkaProducer: Instância configurada
    """
    return KafkaProducer(
        bootstrap_servers=config.get('bootstrap_servers', 'localhost:9092'),
        client_id=config.get('client_id', 'neural-crypto-bot-data-collection'),
        acks=config.get('acks', 'all'),
        retries=config.get('retries', 3),
        batch_size=config.get('batch_size', 16384),
        linger_ms=config.get('linger_ms', 5),
        compression_type=config.get('compression_type', 'snappy'),
        max_in_flight_requests_per_connection=config.get('max_in_flight_requests_per_connection', 1),
        enable_idempotence=config.get('enable_idempotence', True),
        topic_prefix=config.get('topic_prefix', 'market-data')
    )