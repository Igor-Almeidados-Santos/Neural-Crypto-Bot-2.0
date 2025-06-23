"""
Cliente WebSocket genérico e reutilizável para conexões com exchanges.

Este módulo implementa um cliente WebSocket robusto com funcionalidades
avançadas como reconexão automática, heartbeat, rate limiting e
gerenciamento de subscriptions.
"""
import asyncio
import json
import logging
import ssl
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from dataclasses import dataclass, field
from enum import Enum
import traceback

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, InvalidStatusCode, InvalidHandshake
import aiohttp

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Estados possíveis da conexão WebSocket."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class WebSocketConfig:
    """Configuração para conexão WebSocket."""
    url: str
    ping_interval: int = 20
    ping_timeout: int = 10
    close_timeout: int = 10
    max_size: int = 2**20  # 1MB
    max_queue: int = 32
    compression: Optional[str] = None
    extra_headers: Dict[str, str] = field(default_factory=dict)
    subprotocols: List[str] = field(default_factory=list)
    ssl_context: Optional[ssl.SSLContext] = None
    
    # Configurações de reconexão
    enable_auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    initial_reconnect_delay: float = 1.0
    max_reconnect_delay: float = 60.0
    reconnect_backoff_factor: float = 2.0
    
    # Configurações de rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    
    # Configurações de heartbeat
    enable_heartbeat: bool = True
    heartbeat_interval: int = 30
    heartbeat_timeout: int = 10


@dataclass
class SubscriptionInfo:
    """Informações sobre uma subscription ativa."""
    id: str
    channel: str
    symbol: Optional[str]
    callback: Callable[[Dict[str, Any]], Awaitable[None]]
    created_at: datetime
    last_message_at: Optional[datetime] = None
    message_count: int = 0
    error_count: int = 0


class WebSocketClient:
    """
    Cliente WebSocket genérico e robusto para exchanges de criptomoedas.
    
    Características:
    - Reconexão automática com backoff exponencial
    - Rate limiting inteligente
    - Heartbeat/ping-pong automático
    - Gerenciamento de subscriptions
    - Métricas e observabilidade
    - Error handling robusto
    """
    
    def __init__(self, config: WebSocketConfig):
        """
        Inicializa o cliente WebSocket.
        
        Args:
            config: Configuração da conexão
        """
        self.config = config
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._last_reconnect_time = 0.0
        
        # Subscriptions ativas
        self._subscriptions: Dict[str, SubscriptionInfo] = {}
        self._subscription_lock = asyncio.Lock()
        
        # Rate limiting
        self._request_timestamps: List[float] = []
        self._rate_limit_lock = asyncio.Lock()
        
        # Tasks de controle
        self._receive_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Event handlers
        self._on_connect_handlers: List[Callable[[], Awaitable[None]]] = []
        self._on_disconnect_handlers: List[Callable[[], Awaitable[None]]] = []
        self._on_error_handlers: List[Callable[[Exception], Awaitable[None]]] = []
        
        # Métricas
        self._metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'connection_count': 0,
            'reconnection_count': 0,
            'error_count': 0,
            'bytes_sent': 0,
            'bytes_received': 0
        }
        
        # Controle de shutdown
        self._shutdown_event = asyncio.Event()
        self._is_closing = False
    
    @property
    def state(self) -> ConnectionState:
        """Estado atual da conexão."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Retorna True se a conexão está ativa."""
        return self._state == ConnectionState.CONNECTED and self._websocket and not self._websocket.closed
    
    @property
    def subscriptions(self) -> Dict[str, SubscriptionInfo]:
        """Subscriptions ativas."""
        return self._subscriptions.copy()
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Métricas da conexão."""
        return {
            **self._metrics,
            'uptime_seconds': time.time() - self._last_reconnect_time if self._last_reconnect_time else 0,
            'subscription_count': len(self._subscriptions),
            'state': self._state.value,
            'reconnect_attempts': self._reconnect_attempts
        }
    
    def add_connect_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Adiciona handler para evento de conexão."""
        self._on_connect_handlers.append(handler)
    
    def add_disconnect_handler(self, handler: Callable[[], Awaitable[None]]) -> None:
        """Adiciona handler para evento de desconexão."""
        self._on_disconnect_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable[[Exception], Awaitable[None]]) -> None:
        """Adiciona handler para eventos de erro."""
        self._on_error_handlers.append(handler)
    
    async def connect(self) -> None:
        """
        Estabelece conexão WebSocket.
        
        Raises:
            ConnectionError: Se não conseguir conectar após várias tentativas
        """
        if self._state in [ConnectionState.CONNECTING, ConnectionState.CONNECTED]:
            return
        
        self._state = ConnectionState.CONNECTING
        logger.info(f"Conectando ao WebSocket: {self.config.url}")
        
        try:
            # Configura SSL se necessário
            ssl_context = self.config.ssl_context
            if ssl_context is None and self.config.url.startswith('wss://'):
                ssl_context = ssl.create_default_context()
            
            # Estabelece conexão
            self._websocket = await websockets.connect(
                self.config.url,
                ping_interval=self.config.ping_interval,
                ping_timeout=self.config.ping_timeout,
                close_timeout=self.config.close_timeout,
                max_size=self.config.max_size,
                max_queue=self.config.max_queue,
                compression=self.config.compression,
                extra_headers=self.config.extra_headers,
                subprotocols=self.config.subprotocols,
                ssl=ssl_context
            )
            
            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            self._last_reconnect_time = time.time()
            self._metrics['connection_count'] += 1
            
            logger.info("Conexão WebSocket estabelecida com sucesso")
            
            # Inicia tasks de controle
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            if self.config.enable_heartbeat:
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Executa handlers de conexão
            await self._execute_handlers(self._on_connect_handlers)
            
            # Reativa subscriptions se existirem
            await self._reactivate_subscriptions()
            
        except Exception as e:
            self._state = ConnectionState.ERROR
            self._metrics['error_count'] += 1
            logger.error(f"Erro ao conectar WebSocket: {str(e)}")
            
            if self.config.enable_auto_reconnect:
                await self._schedule_reconnect()
            else:
                await self._execute_error_handlers(e)
                raise ConnectionError(f"Falha ao conectar: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Desconecta do WebSocket graciosamente."""
        if self._state in [ConnectionState.DISCONNECTED, ConnectionState.CLOSING]:
            return
        
        self._state = ConnectionState.CLOSING
        self._is_closing = True
        logger.info("Iniciando desconexão do WebSocket")
        
        # Cancela tasks
        await self._cancel_tasks()
        
        # Fecha conexão
        if self._websocket and not self._websocket.closed:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Erro ao fechar WebSocket: {str(e)}")
        
        self._state = ConnectionState.DISCONNECTED
        self._websocket = None
        
        # Executa handlers de desconexão
        await self._execute_handlers(self._on_disconnect_handlers)
        
        logger.info("WebSocket desconectado")
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """
        Envia mensagem via WebSocket.
        
        Args:
            message: Mensagem a ser enviada
            
        Raises:
            ConnectionError: Se não estiver conectado
            Exception: Se ocorrer erro no envio
        """
        if not self.is_connected:
            raise ConnectionError("WebSocket não está conectado")
        
        # Aplica rate limiting
        await self._apply_rate_limit()
        
        try:
            message_str = json.dumps(message)
            await self._websocket.send(message_str)
            
            self._metrics['messages_sent'] += 1
            self._metrics['bytes_sent'] += len(message_str.encode('utf-8'))
            
            logger.debug(f"Mensagem enviada: {message_str}")
            
        except Exception as e:
            self._metrics['error_count'] += 1
            logger.error(f"Erro ao enviar mensagem: {str(e)}")
            raise
    
    async def subscribe(
        self,
        subscription_id: str,
        channel: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
        symbol: Optional[str] = None,
        subscribe_message: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Cria uma subscription para um canal.
        
        Args:
            subscription_id: ID único da subscription
            channel: Canal a ser subscrito
            callback: Função callback para processar mensagens
            symbol: Símbolo opcional (para subscriptions específicas)
            subscribe_message: Mensagem de subscription personalizada
        """
        async with self._subscription_lock:
            if subscription_id in self._subscriptions:
                logger.warning(f"Subscription {subscription_id} já existe")
                return
            
            subscription = SubscriptionInfo(
                id=subscription_id,
                channel=channel,
                symbol=symbol,
                callback=callback,
                created_at=datetime.utcnow()
            )
            
            self._subscriptions[subscription_id] = subscription
            
            # Envia mensagem de subscription se conectado
            if self.is_connected and subscribe_message:
                await self.send_message(subscribe_message)
            
            logger.info(f"Subscription criada: {subscription_id} para canal {channel}")
    
    async def unsubscribe(
        self,
        subscription_id: str,
        unsubscribe_message: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Remove uma subscription.
        
        Args:
            subscription_id: ID da subscription a ser removida
            unsubscribe_message: Mensagem de unsubscribe personalizada
        """
        async with self._subscription_lock:
            if subscription_id not in self._subscriptions:
                logger.warning(f"Subscription {subscription_id} não encontrada")
                return
            
            # Envia mensagem de unsubscribe se conectado
            if self.is_connected and unsubscribe_message:
                await self.send_message(unsubscribe_message)
            
            # Remove subscription
            del self._subscriptions[subscription_id]
            
            logger.info(f"Subscription removida: {subscription_id}")
    
    async def _receive_loop(self) -> None:
        """Loop principal de recebimento de mensagens."""
        try:
            async for raw_message in self._websocket:
                if self._is_closing:
                    break
                
                try:
                    # Parse da mensagem
                    if isinstance(raw_message, bytes):
                        message = json.loads(raw_message.decode('utf-8'))
                        self._metrics['bytes_received'] += len(raw_message)
                    else:
                        message = json.loads(raw_message)
                        self._metrics['bytes_received'] += len(raw_message.encode('utf-8'))
                    
                    self._metrics['messages_received'] += 1
                    
                    # Processa a mensagem
                    await self._process_message(message)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Erro ao decodificar mensagem JSON: {str(e)}")
                    self._metrics['error_count'] += 1
                except Exception as e:
                    logger.error(f"Erro ao processar mensagem: {str(e)}")
                    self._metrics['error_count'] += 1
                    
        except ConnectionClosed:
            logger.warning("Conexão WebSocket fechada pelo servidor")
        except Exception as e:
            logger.error(f"Erro no loop de recepção: {str(e)}")
            await self._execute_error_handlers(e)
        finally:
            if not self._is_closing and self.config.enable_auto_reconnect:
                await self._schedule_reconnect()
    
    async def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Processa uma mensagem recebida.
        
        Args:
            message: Mensagem recebida
        """
        # Identifica subscription baseada na mensagem
        # Este método deve ser sobrescrito por implementações específicas
        subscription_id = await self._identify_subscription(message)
        
        if subscription_id and subscription_id in self._subscriptions:
            subscription = self._subscriptions[subscription_id]
            subscription.last_message_at = datetime.utcnow()
            subscription.message_count += 1
            
            try:
                await subscription.callback(message)
            except Exception as e:
                subscription.error_count += 1
                logger.error(
                    f"Erro no callback da subscription {subscription_id}: {str(e)}"
                )
        else:
            logger.debug(f"Mensagem não associada a subscription: {message}")
    
    async def _identify_subscription(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Identifica qual subscription deve processar a mensagem.
        
        Este método deve ser sobrescrito por implementações específicas.
        
        Args:
            message: Mensagem recebida
            
        Returns:
            ID da subscription ou None se não identificada
        """
        # Implementação padrão básica
        return None
    
    async def _heartbeat_loop(self) -> None:
        """Loop de heartbeat para manter conexão ativa."""
        while not self._is_closing and self.is_connected:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if self.is_connected:
                    # Envia ping personalizado se necessário
                    await self._send_heartbeat()
                    
            except Exception as e:
                logger.error(f"Erro no heartbeat: {str(e)}")
                break
    
    async def _send_heartbeat(self) -> None:
        """
        Envia heartbeat customizado.
        
        Pode ser sobrescrito por implementações específicas.
        """
        # Implementação padrão não faz nada
        # O websockets já lida com ping/pong automaticamente
        pass
    
    async def _apply_rate_limit(self) -> None:
        """Aplica rate limiting nas requisições."""
        async with self._rate_limit_lock:
            now = time.time()
            
            # Remove timestamps antigos
            self._request_timestamps = [
                ts for ts in self._request_timestamps
                if now - ts < self.config.rate_limit_window
            ]
            
            # Verifica se excedeu o limite
            if len(self._request_timestamps) >= self.config.rate_limit_requests:
                sleep_time = self.config.rate_limit_window - (now - self._request_timestamps[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit atingido, aguardando {sleep_time:.2f}s")
                    await asyncio.sleep(sleep_time)
            
            self._request_timestamps.append(now)
    
    async def _schedule_reconnect(self) -> None:
        """Agenda uma tentativa de reconexão."""
        if self._is_closing or self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error("Máximo de tentativas de reconexão atingido")
            self._state = ConnectionState.ERROR
            return
        
        self._reconnect_attempts += 1
        delay = min(
            self.config.initial_reconnect_delay * (self.config.reconnect_backoff_factor ** (self._reconnect_attempts - 1)),
            self.config.max_reconnect_delay
        )
        
        logger.info(f"Reagendando reconexão em {delay:.2f}s (tentativa {self._reconnect_attempts})")
        
        self._state = ConnectionState.RECONNECTING
        self._reconnect_task = asyncio.create_task(self._reconnect_after_delay(delay))
    
    async def _reconnect_after_delay(self, delay: float) -> None:
        """Reconecta após um delay."""
        try:
            await asyncio.sleep(delay)
            
            if not self._is_closing:
                self._metrics['reconnection_count'] += 1
                await self.connect()
                
        except Exception as e:
            logger.error(f"Erro na reconexão: {str(e)}")
            await self._execute_error_handlers(e)
    
    async def _reactivate_subscriptions(self) -> None:
        """Reativa todas as subscriptions após reconexão."""
        if not self._subscriptions:
            return
        
        logger.info(f"Reativando {len(self._subscriptions)} subscriptions")
        
        for subscription_id, subscription in self._subscriptions.items():
            try:
                # Este método deve ser implementado por classes específicas
                await self._reactivate_subscription(subscription)
            except Exception as e:
                logger.error(f"Erro ao reativar subscription {subscription_id}: {str(e)}")
    
    async def _reactivate_subscription(self, subscription: SubscriptionInfo) -> None:
        """
        Reativa uma subscription específica.
        
        Deve ser implementado por classes específicas.
        """
        pass
    
    async def _cancel_tasks(self) -> None:
        """Cancela todas as tasks ativas."""
        tasks = [
            self._receive_task,
            self._heartbeat_task,
            self._reconnect_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _execute_handlers(self, handlers: List[Callable[[], Awaitable[None]]]) -> None:
        """Executa uma lista de handlers."""
        for handler in handlers:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Erro ao executar handler: {str(e)}")
    
    async def _execute_error_handlers(self, error: Exception) -> None:
        """Executa handlers de erro."""
        for handler in self._on_error_handlers:
            try:
                await handler(error)
            except Exception as e:
                logger.error(f"Erro ao executar error handler: {str(e)}")
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.disconnect()


class ExchangeWebSocketClient(WebSocketClient):
    """
    Cliente WebSocket especializado para exchanges.
    
    Extends WebSocketClient com funcionalidades específicas para exchanges
    de criptomoedas, incluindo identificação automática de subscriptions
    e tratamento de mensagens específicas.
    """
    
    def __init__(self, config: WebSocketConfig, exchange_name: str):
        """
        Inicializa o cliente específico para exchange.
        
        Args:
            config: Configuração da conexão
            exchange_name: Nome da exchange (binance, coinbase, etc.)
        """
        super().__init__(config)
        self.exchange_name = exchange_name.lower()
        
        # Mapeamento de canais para identificação de mensagens
        self._channel_mapping: Dict[str, str] = {}
    
    def register_channel_mapping(self, message_key: str, subscription_id: str) -> None:
        """
        Registra mapeamento entre chave de mensagem e subscription ID.
        
        Args:
            message_key: Chave na mensagem que identifica o canal
            subscription_id: ID da subscription correspondente
        """
        self._channel_mapping[message_key] = subscription_id
    
    async def _identify_subscription(self, message: Dict[str, Any]) -> Optional[str]:
        """
        Identifica subscription baseada na mensagem da exchange.
        
        Args:
            message: Mensagem recebida
            
        Returns:
            ID da subscription ou None
        """
        # Implementação específica por exchange
        if self.exchange_name == "binance":
            return self._identify_binance_subscription(message)
        elif self.exchange_name == "coinbase":
            return self._identify_coinbase_subscription(message)
        elif self.exchange_name == "kraken":
            return self._identify_kraken_subscription(message)
        elif self.exchange_name == "bybit":
            return self._identify_bybit_subscription(message)
        
        return None
    
    def _identify_binance_subscription(self, message: Dict[str, Any]) -> Optional[str]:
        """Identifica subscription para mensagens da Binance."""
        if 'stream' in message and 'data' in message:
            stream_name = message['stream']
            return self._channel_mapping.get(stream_name)
        return None
    
    def _identify_coinbase_subscription(self, message: Dict[str, Any]) -> Optional[str]:
        """Identifica subscription para mensagens da Coinbase."""
        if 'type' in message and 'product_id' in message:
            channel = f"{message['type']}:{message['product_id']}"
            return self._channel_mapping.get(channel)
        return None
    
    def _identify_kraken_subscription(self, message: Dict[str, Any]) -> Optional[str]:
        """Identifica subscription para mensagens da Kraken."""
        if isinstance(message, list) and len(message) >= 3:
            channel_info = message[-1]  # Último elemento contém info do canal
            if isinstance(channel_info, dict) and 'pair' in channel_info:
                channel = f"{channel_info.get('name', '')}:{channel_info['pair']}"
                return self._channel_mapping.get(channel)
        return None
    
    def _identify_bybit_subscription(self, message: Dict[str, Any]) -> Optional[str]:
        """Identifica subscription para mensagens da Bybit."""
        if 'topic' in message:
            topic = message['topic']
            return self._channel_mapping.get(topic)
        return None