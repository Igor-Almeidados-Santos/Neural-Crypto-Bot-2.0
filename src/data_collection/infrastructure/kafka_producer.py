"""
Produtor Kafka para publicação de dados de mercado.

Este módulo implementa um produtor Kafka para publicar dados
de mercado coletados, permitindo que outros serviços consumam
esses dados em tempo real.
"""
import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Callable

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook
from src.data_collection.domain.entities.trade import Trade


logger = logging.getLogger(__name__)


class KafkaProducer:
    """
    Produtor Kafka para publicação de dados de mercado.
    
    Implementa funcionalidades para publicar dados de mercado
    como trades, orderbooks e candles em tópicos Kafka.
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
        topic_prefix: str = "market-data"
    ):
        """
        Inicializa o produtor Kafka.
        
        Args:
            bootstrap_servers: Lista de servidores Kafka no formato "host1:port1,host2:port2"
            client_id: ID do cliente Kafka
            acks: Configuração de acknowledges ("0", "1", "all")
            retries: Número de tentativas de reenvio em caso de falha
            batch_size: Tamanho máximo do batch em bytes
            linger_ms: Tempo de espera para acumular mensagens em milissegundos
            compression_type: Tipo de compressão ("none", "gzip", "snappy", "lz4", "zstd")
            max_in_flight_requests_per_connection: Número máximo de requisições em andamento
            enable_idempotence: Se True, habilita idempotência para garantir entrega exatamente uma vez
            topic_prefix: Prefixo para os nomes dos tópicos
        """
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
        
        # Configura o produtor Kafka
        self._producer_config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': client_id,
            'acks': acks,
            'retries': retries,
            'batch.size': batch_size,
            'linger.ms': linger_ms,
            'compression.type': compression_type,
            'max.in.flight.requests.per.connection': max_in_flight_requests_per_connection,
            'enable.idempotence': enable_idempotence
        }
        
        # Inicializa o produtor
        self._producer = None
        
        # Flag para controle de estado
        self._initialized = False
        
        # Locks para inicialização e publicação
        self._init_lock = asyncio.Lock()
        self._publish_lock = asyncio.Lock()
        
        # Tópicos padrão
        self._default_topics = {
            'trades': f"{self.topic_prefix}.trades",
            'orderbooks': f"{self.topic_prefix}.orderbooks",
            'candles': f"{self.topic_prefix}.candles"
        }
        
        # Callbacks de entrega
        self._delivery_callbacks = {}
    
    async def initialize(self) -> None:
        """
        Inicializa o produtor Kafka.
        
        Esta função deve ser chamada antes de publicar mensagens.
        """
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                logger.info(f"Inicializando produtor Kafka: {self.bootstrap_servers}")
                
                # Cria o produtor
                self._producer = Producer(self._producer_config)
                
                # Verifica/cria os tópicos
                await self._ensure_topics()
                
                self._initialized = True
                logger.info("Produtor Kafka inicializado")
                
            except Exception as e:
                logger.error(f"Erro ao inicializar produtor Kafka: {str(e)}", exc_info=True)
                raise
    
    async def close(self) -> None:
        """
        Fecha o produtor Kafka.
        """
        if self._producer:
            logger.info("Fechando produtor Kafka")
            
            # Aguarda a entrega de todas as mensagens pendentes
            self._producer.flush()
            
            # Libera o produtor
            self._producer = None
            self._initialized = False
            
            logger.info("Produtor Kafka fechado")
    
    async def publish_trade(self, trade: Trade, topic: Optional[str] = None) -> None:
        """
        Publica uma transação em um tópico Kafka.
        
        Args:
            trade: Entidade Trade a ser publicada
            topic: Nome do tópico (opcional, usa o tópico padrão se não fornecido)
            
        Raises:
            Exception: Se ocorrer um erro ao publicar
        """
        if not self._initialized:
            await self.initialize()
            
        # Usa o tópico padrão se não fornecido
        if topic is None:
            topic = self._default_topics['trades']
            
        # Converte a entidade para JSON
        message = self._trade_to_json(trade)
        
        # Publica a mensagem
        await self._publish_message(topic, message, key=f"{trade.exchange}:{trade.trading_pair}")
    
    async def publish_trades_batch(self, trades: List[Trade], topic: Optional[str] = None) -> None:
        """
        Publica um lote de transações em um tópico Kafka.
        
        Args:
            trades: Lista de entidades Trade a serem publicadas
            topic: Nome do tópico (opcional, usa o tópico padrão se não fornecido)
            
        Raises:
            Exception: Se ocorrer um erro ao publicar
        """
        if not trades:
            return
            
        if not self._initialized:
            await self.initialize()
            
        # Usa o tópico padrão se não fornecido
        if topic is None:
            topic = self._default_topics['trades']
            
        # Publica cada trade
        for trade in trades:
            # Converte a entidade para JSON
            message = self._trade_to_json(trade)
            
            # Publica a mensagem sem aguardar confirmação
            self._producer.produce(
                topic=topic,
                key=f"{trade.exchange}:{trade.trading_pair}",
                value=message,
                callback=self._delivery_report
            )
            
        # Faz o flush para garantir que as mensagens sejam enviadas
        self._producer.poll(0)
    
    async def publish_orderbook(self, orderbook: OrderBook, topic: Optional[str] = None) -> None:
        """
        Publica um orderbook em um tópico Kafka.
        
        Args:
            orderbook: Entidade OrderBook a ser publicada
            topic: Nome do tópico (opcional, usa o tópico padrão se não fornecido)
            
        Raises:
            Exception: Se ocorrer um erro ao publicar
        """
        if not self._initialized:
            await self.initialize()
            
        # Usa o tópico padrão se não fornecido
        if topic is None:
            topic = self._default_topics['orderbooks']
            
        # Converte a entidade para JSON
        message = self._orderbook_to_json(orderbook)
        
        # Publica a mensagem
        await self._publish_message(topic, message, key=f"{orderbook.exchange}:{orderbook.trading_pair}")
    
    async def publish_candle(self, candle: Candle, topic: Optional[str] = None) -> None:
        """
        Publica uma vela em um tópico Kafka.
        
        Args:
            candle: Entidade Candle a ser publicada
            topic: Nome do tópico (opcional, usa o tópico padrão se não fornecido)
            
        Raises:
            Exception: Se ocorrer um erro ao publicar
        """
        if not self._initialized:
            await self.initialize()
            
        # Usa o tópico padrão se não fornecido
        if topic is None:
            topic = self._default_topics['candles']
            
        # Converte a entidade para JSON
        message = self._candle_to_json(candle)
        
        # Publica a mensagem
        await self._publish_message(topic, message, key=f"{candle.exchange}:{candle.trading_pair}:{candle.timeframe.value}")
    
    async def publish_candles_batch(self, candles: List[Candle], topic: Optional[str] = None) -> None:
        """
        Publica um lote de velas em um tópico Kafka.
        
        Args:
            candles: Lista de entidades Candle a serem publicadas
            topic: Nome do tópico (opcional, usa o tópico padrão se não fornecido)
            
        Raises:
            Exception: Se ocorrer um erro ao publicar
        """
        if not candles:
            return
            
        if not self._initialized:
            await self.initialize()
            
        # Usa o tópico padrão se não fornecido
        if topic is None:
            topic = self._default_topics['candles']
            
        # Publica cada candle
        for candle in candles:
            # Converte a entidade para JSON
            message = self._candle_to_json(candle)
            
            # Publica a mensagem sem aguardar confirmação
            self._producer.produce(
                topic=topic,
                key=f"{candle.exchange}:{candle.trading_pair}:{candle.timeframe.value}",
                value=message,
                callback=self._delivery_report
            )
            
        # Faz o flush para garantir que as mensagens sejam enviadas
        self._producer.poll(0)
    
    async def _publish_message(self, topic: str, message: str, key: Optional[str] = None) -> None:
        """
        Publica uma mensagem em um tópico Kafka.
        
        Args:
            topic: Nome do tópico
            message: Mensagem a ser publicada (JSON)
            key: Chave da mensagem (opcional)
            
        Raises:
            Exception: Se ocorrer um erro ao publicar
        """
        async with self._publish_lock:
            try:
                # Gera um ID único para esta mensagem
                message_id = f"{topic}-{datetime.utcnow().timestamp()}"
                
                # Cria um futuro para aguardar a confirmação
                delivery_future = asyncio.Future()
                self._delivery_callbacks[message_id] = delivery_future
                
                # Publica a mensagem
                self._producer.produce(
                    topic=topic,
                    key=key,
                    value=message,
                    callback=lambda err, msg: self._delivery_callback(err, msg, message_id)
                )
                
                # Faz o poll para iniciar o envio
                self._producer.poll(0)
                
                # Aguarda a confirmação de entrega (com timeout)
                try:
                    await asyncio.wait_for(delivery_future, timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout ao aguardar confirmação de entrega para {topic}")
                finally:
                    # Remove o callback mesmo em caso de timeout
                    self._delivery_callbacks.pop(message_id, None)
                
            except Exception as e:
                logger.error(f"Erro ao publicar mensagem no tópico {topic}: {str(e)}", exc_info=True)
                raise
    
    def _delivery_callback(self, err, msg, message_id: str) -> None:
        """
        Callback chamado quando uma mensagem é entregue ou falha.
        
        Args:
            err: Erro (None se entregue com sucesso)
            msg: Mensagem entregue
            message_id: ID único da mensagem
        """
        future = self._delivery_callbacks.get(message_id)
        if future is None:
            return
            
        if err is not None:
            future.set_exception(Exception(f"Erro ao entregar mensagem: {err}"))
        else:
            future.set_result(True)
    
    def _delivery_report(self, err, msg) -> None:
        """
        Relatório de entrega para mensagens em batch.
        
        Args:
            err: Erro (None se entregue com sucesso)
            msg: Mensagem entregue
        """
        if err is not None:
            logger.error(f"Erro ao entregar mensagem: {err}")
        else:
            logger.debug(f"Mensagem entregue: {msg.topic()} [{msg.partition()}]")
    
    async def _ensure_topics(self) -> None:
        """
        Verifica se os tópicos necessários existem e cria-os se necessário.
        """
        try:
            # Cria um cliente de administração
            admin_client = AdminClient({'bootstrap.servers': self.bootstrap_servers})
            
            # Lista os tópicos existentes
            metadata = admin_client.list_topics(timeout=10)
            existing_topics = metadata.topics
            
            # Tópicos a serem criados
            topics_to_create = []
            
            for topic_name in self._default_topics.values():
                if topic_name not in existing_topics:
                    logger.info(f"Tópico não encontrado, será criado: {topic_name}")
                    
                    # Configuração do tópico
                    topic_config = {
                        'cleanup.policy': 'delete',       # Política de limpeza
                        'retention.ms': '86400000',       # Retenção de 24 horas
                        'segment.ms': '3600000',          # Segmentos de 1 hora
                        'segment.bytes': '1073741824',    # Segmentos de 1 GB
                        'min.insync.replicas': '1'        # Replicação mínima
                    }
                    
                    # Cria o tópico
                    topics_to_create.append(NewTopic(
                        topic_name,
                        num_partitions=12,              # Número de partições
                        replication_factor=1,           # Fator de replicação
                        config=topic_config
                    ))
            
            # Cria os tópicos em paralelo
            if topics_to_create:
                logger.info(f"Criando {len(topics_to_create)} tópicos")
                admin_client.create_topics(topics_to_create)
            
        except Exception as e:
            logger.warning(f"Erro ao verificar/criar tópicos: {str(e)}")
            # Continua mesmo em caso de erro, pois os tópicos podem ser criados automaticamente
    
    def _trade_to_json(self, trade: Trade) -> str:
        """
        Converte uma entidade Trade para JSON.
        
        Args:
            trade: Entidade Trade
            
        Returns:
            str: Representação JSON da entidade
        """
        # Cria um dicionário com os dados
        data = {
            "id": trade.id,
            "exchange": trade.exchange,
            "trading_pair": trade.trading_pair,
            "price": float(trade.price),
            "amount": float(trade.amount),
            "cost": float(trade.cost),
            "timestamp": trade.timestamp.isoformat(),
            "side": trade.side.value,
            "taker": trade.taker
        }
        
        # Converte para JSON
        return json.dumps(data)
    
    def _orderbook_to_json(self, orderbook: OrderBook) -> str:
        """
        Converte uma entidade OrderBook para JSON.
        
        Args:
            orderbook: Entidade OrderBook
            
        Returns:
            str: Representação JSON da entidade
        """
        # Cria um dicionário com os dados
        data = {
            "exchange": orderbook.exchange,
            "trading_pair": orderbook.trading_pair,
            "timestamp": orderbook.timestamp.isoformat(),
            "bids": [
                {"price": float(bid.price), "amount": float(bid.amount), "count": bid.count}
                for bid in orderbook.bids
            ],
            "asks": [
                {"price": float(ask.price), "amount": float(ask.amount), "count": ask.count}
                for ask in orderbook.asks
            ]
        }
        
        # Adiciona algumas métricas calculadas
        try:
            data["mid_price"] = float(orderbook.mid_price)
            data["spread"] = float(orderbook.spread)
            data["spread_percentage"] = float(orderbook.spread_percentage)
        except Exception:
            pass
        
        # Converte para JSON
        return json.dumps(data)
    
    def _candle_to_json(self, candle: Candle) -> str:
        """
        Converte uma entidade Candle para JSON.
        
        Args:
            candle: Entidade Candle
            
        Returns:
            str: Representação JSON da entidade
        """
        # Cria um dicionário com os dados
        data = {
            "exchange": candle.exchange,
            "trading_pair": candle.trading_pair,
            "timestamp": candle.timestamp.isoformat(),
            "timeframe": candle.timeframe.value,
            "open": float(candle.open),
            "high": float(candle.high),
            "low": float(candle.low),
            "close": float(candle.close),
            "volume": float(candle.volume)
        }
        
        # Adiciona o número de trades se disponível
        if candle.trades is not None:
            data["trades"] = candle.trades
        
        # Adiciona algumas métricas calculadas
        try:
            data["is_bullish"] = candle.is_bullish
            data["is_bearish"] = candle.is_bearish
            data["body_size"] = float(candle.body_size)
            data["range"] = float(candle.range)
        except Exception:
            pass
        
        # Converte para JSON
        return json.dumps(data)