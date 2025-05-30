"""
Cliente WebSocket para comunicação com exchanges.

Este módulo implementa um cliente WebSocket genérico para comunicação
com exchanges de criptomoedas, utilizando a biblioteca websockets.
"""
import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
import time

import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)


class WebSocketClient:
    """
    Cliente WebSocket para comunicação com exchanges.
    
    Implementa funcionalidades para estabelecer conexões WebSocket,
    enviar e receber mensagens, e gerenciar reconexões automáticas.
    """
    
    def __init__(
        self, 
        url: str, 
        ping_interval: Optional[int] = 30,
        ping_timeout: Optional[int] = 10,
        close_timeout: Optional[int] = 10,
        max_message_size: Optional[int] = 10 * 1024 * 1024,  # 10 MB
        max_queue_size: Optional[int] = 1000
    ):
        """
        Inicializa o cliente WebSocket.
        
        Args:
            url: URL do endpoint WebSocket
            ping_interval: Intervalo entre pings em segundos (None para desabilitar)
            ping_timeout: Timeout para resposta de ping em segundos
            close_timeout: Timeout para fechamento da conexão em segundos
            max_message_size: Tamanho máximo de mensagem em bytes
            max_queue_size: Tamanho máximo da fila de mensagens
        """
        self.url = url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        self.max_message_size = max_message_size
        self.max_queue_size = max_queue_size
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connect_lock = asyncio.Lock()
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._last_recv_time = 0
        
        # Callbacks
        self._on_message_callbacks: List[Callable[[str], Coroutine[Any, Any, None]]] = []
        self._on_connect_callbacks: List[Callable[[], Coroutine[Any, Any, None]]] = []
        self._on_close_callbacks: List[Callable[[int, str], Coroutine[Any, Any, None]]] = []
        self._on_error_callbacks: List[Callable[[Exception], Coroutine[Any, Any, None]]] = []
        
    async def connect(self) -> None:
        """
        Estabelece conexão com o servidor WebSocket.
        
        Raises:
            ConnectionError: Se não for possível estabelecer a conexão
        """
        # Impede conexões simultâneas
        async with self._connect_lock:
            if self.is_connected():
                return
                
            try:
                logger.debug(f"Conectando ao WebSocket: {self.url}")
                
                # Conecta ao WebSocket
                self._ws = await websockets.connect(
                    self.url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    close_timeout=self.close_timeout,
                    max_size=self.max_message_size
                )
                
                # Inicia as tasks de processamento
                self._running = True
                self._processing_task = asyncio.create_task(self._process_messages())
                
                if self.ping_interval:
                    self._ping_task = asyncio.create_task(self._ping_loop())
                
                self._last_recv_time = time.time()
                
                logger.debug(f"Conectado ao WebSocket: {self.url}")
                
                # Chama os callbacks de conexão
                for callback in self._on_connect_callbacks:
                    try:
                        await callback()
                    except Exception as e:
                        logger.error(f"Erro no callback de conexão: {str(e)}", exc_info=True)
                
            except Exception as e:
                logger.error(f"Erro ao conectar ao WebSocket {self.url}: {str(e)}", exc_info=True)
                raise ConnectionError(f"Falha ao conectar ao WebSocket: {str(e)}")
    
    async def close(self) -> None:
        """
        Fecha a conexão com o servidor WebSocket.
        """
        self._running = False
        
        # Cancela as tasks
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        
        # Fecha a conexão
        if self._ws:
            try:
                await self._ws.close()
                logger.debug(f"Conexão WebSocket fechada: {self.url}")
            except Exception as e:
                logger.error(f"Erro ao fechar conexão WebSocket {self.url}: {str(e)}", exc_info=True)
            
            self._ws = None
    
    def is_connected(self) -> bool:
        """
        Verifica se o cliente está conectado ao servidor.
        
        Returns:
            bool: True se estiver conectado, False caso contrário
        """
        return self._ws is not None and not self._ws.closed
    
    async def send(self, message: Union[str, bytes, Dict]) -> None:
        """
        Envia uma mensagem para o servidor.
        
        Args:
            message: Mensagem a ser enviada (string, bytes ou dict que será convertido para JSON)
            
        Raises:
            ConnectionError: Se não estiver conectado ao servidor
            ValueError: Se a mensagem for inválida
        """
        if not self.is_connected():
            await self.connect()
        
        if not self._ws:
            raise ConnectionError("Não conectado ao servidor WebSocket")
        
        try:
            # Converte dict para JSON
            if isinstance(message, dict):
                message = json.dumps(message)
            
            await self._ws.send(message)
            logger.debug(f"Mensagem enviada: {message[:200]}...")
            
        except Exception as e:
            logger.error(f"Erro ao enviar mensagem: {str(e)}", exc_info=True)
            await self._handle_error(e)
    
    def on_message(self, callback: Callable[[str], Coroutine[Any, Any, None]]) -> None:
        """
        Registra um callback para receber mensagens.
        
        Args:
            callback: Função assíncrona a ser chamada quando uma mensagem for recebida
        """
        self._on_message_callbacks.append(callback)
    
    def on_connect(self, callback: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """
        Registra um callback para o evento de conexão.
        
        Args:
            callback: Função assíncrona a ser chamada quando a conexão for estabelecida
        """
        self._on_connect_callbacks.append(callback)
    
    def on_close(self, callback: Callable[[int, str], Coroutine[Any, Any, None]]) -> None:
        """
        Registra um callback para o evento de fechamento da conexão.
        
        Args:
            callback: Função assíncrona a ser chamada quando a conexão for fechada
        """
        self._on_close_callbacks.append(callback)
    
    def on_error(self, callback: Callable[[Exception], Coroutine[Any, Any, None]]) -> None:
        """
        Registra um callback para o evento de erro.
        
        Args:
            callback: Função assíncrona a ser chamada quando ocorrer um erro
        """
        self._on_error_callbacks.append(callback)
    
    async def _process_messages(self) -> None:
        """
        Processa mensagens recebidas do WebSocket.
        
        Esta função é executada como uma task em segundo plano.
        """
        while self._running and self._ws:
            try:
                # Recebe a próxima mensagem
                message = await self._ws.recv()
                self._last_recv_time = time.time()
                
                # Coloca na fila para processamento
                if self._message_queue.qsize() >= self.max_queue_size:
                    # Se a fila estiver cheia, remove a mensagem mais antiga
                    _ = await self._message_queue.get()
                    self._message_queue.task_done()
                    logger.warning("Fila de mensagens cheia, descartando mensagem mais antiga")
                
                await self._message_queue.put(message)
                
                # Processa as mensagens na fila
                while not self._message_queue.empty():
                    msg = await self._message_queue.get()
                    
                    # Chama os callbacks
                    for callback in self._on_message_callbacks:
                        try:
                            await callback(msg)
                        except Exception as e:
                            logger.error(f"Erro no callback de mensagem: {str(e)}", exc_info=True)
                    
                    self._message_queue.task_done()
                
            except (ConnectionClosedError, ConnectionClosedOK, ConnectionClosed) as e:
                code = e.code if hasattr(e, 'code') else 1000
                reason = e.reason if hasattr(e, 'reason') else "Connection closed"
                
                logger.info(f"Conexão WebSocket fechada: {code}, {reason}")
                
                # Chama os callbacks de fechamento
                for callback in self._on_close_callbacks:
                    try:
                        await callback(code, reason)
                    except Exception as e:
                        logger.error(f"Erro no callback de fechamento: {str(e)}", exc_info=True)
                
                # Sai do loop se a conexão foi fechada normalmente
                if isinstance(e, ConnectionClosedOK) or not self._running:
                    break
                
                # Tenta reconectar
                await self._reconnect()
                
            except Exception as e:
                logger.error(f"Erro ao processar mensagens: {str(e)}", exc_info=True)
                await self._handle_error(e)
                
                # Pequena pausa para evitar loop infinito
                await asyncio.sleep(1)
    
    async def _ping_loop(self) -> None:
        """
        Envia pings periódicos para manter a conexão ativa.
        
        Esta função é executada como uma task em segundo plano.
        """
        while self._running and self._ws:
            try:
                await asyncio.sleep(self.ping_interval)
                
                # Verifica se a conexão está ativa
                if self.is_connected():
                    # Envia um ping manual se necessário
                    elapsed = time.time() - self._last_recv_time
                    if elapsed > self.ping_interval:
                        logger.debug(f"Enviando ping manual (último recebimento: {elapsed:.1f}s atrás)")
                        pong_waiter = await self._ws.ping()
                        await asyncio.wait_for(pong_waiter, timeout=self.ping_timeout)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Erro no loop de ping: {str(e)}", exc_info=True)
                await self._handle_error(e)
                
                # Reconecta se houver problemas
                if self._running:
                    await self._reconnect()
    
    @retry(
        retry=retry_if_exception_type((ConnectionError, ConnectionRefusedError, ConnectionResetError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30)
    )
    async def _reconnect(self) -> None:
        """
        Tenta reconectar ao servidor WebSocket.
        
        Esta função utiliza uma estratégia de retry com backoff exponencial.
        """
        if not self._running:
            return
            
        try:
            # Fecha a conexão atual se existir
            if self._ws:
                await self._ws.close()
                self._ws = None
            
            # Tenta conectar novamente
            logger.info(f"Tentando reconectar ao WebSocket: {self.url}")
            await self.connect()
            
        except Exception as e:
            logger.error(f"Falha ao reconectar: {str(e)}", exc_info=True)
            raise
    
    async def _handle_error(self, error: Exception) -> None:
        """
        Processa erros ocorridos durante a comunicação.
        
        Args:
            error: Exceção ocorrida
        """
        # Chama os callbacks de erro
        for callback in self._on_error_callbacks:
            try:
                await callback(error)
            except Exception as e:
                logger.error(f"Erro no callback de erro: {str(e)}", exc_info=True)
        
        # Reconecta se for um erro de conexão
        if isinstance(error, (ConnectionError, ConnectionRefusedError, ConnectionResetError)):
            if self._running:
                await self._reconnect()