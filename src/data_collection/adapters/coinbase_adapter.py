"""
Adaptador para a exchange Coinbase Pro.

Este módulo implementa o adaptador para a exchange Coinbase Pro, seguindo
a interface comum definida para todos os adaptadores de exchanges.
"""
import asyncio
import base64
import hmac
import hashlib
import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable

import ccxt.async_support as ccxt
from websockets.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed
import orjson

from data_collection.adapters.exchange_adapter_interface import ExchangeAdapterInterface
from data_collection.domain.entities.candle import Candle, TimeFrame
from data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from data_collection.domain.entities.trade import Trade, TradeSide
from data_collection.infrastructure.websocket_client import WebSocketClient


logger = logging.getLogger(__name__)


class CoinbaseAdapter(ExchangeAdapterInterface):
    """
    Adaptador para a exchange Coinbase Pro.
    
    Implementa a interface ExchangeAdapterInterface para a exchange Coinbase Pro,
    utilizando a biblioteca ccxt para requisições REST e WebSockets nativos
    da Coinbase para streaming de dados.
    """
    
    # Mapeamento de timeframes da Coinbase para o formato padronizado
    TIMEFRAME_MAP = {
        "1m": TimeFrame.MINUTE_1,
        "5m": TimeFrame.MINUTE_5,
        "15m": TimeFrame.MINUTE_15,
        "1h": TimeFrame.HOUR_1,
        "6h": TimeFrame.HOUR_6,
        "1d": TimeFrame.DAY_1
    }
    
    # Mapeamento reverso, do formato padronizado para o formato da Coinbase
    REVERSE_TIMEFRAME_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        sandbox: bool = False
    ):
        """
        Inicializa o adaptador para a Coinbase Pro.
        
        Args:
            api_key: Chave de API da Coinbase Pro (opcional)
            api_secret: Chave secreta da API da Coinbase Pro (opcional)
            api_passphrase: Passphrase da API da Coinbase Pro (opcional)
            sandbox: Se True, utiliza o ambiente de sandbox da Coinbase Pro
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._api_passphrase = api_passphrase
        self._sandbox = sandbox
        
        # Define o endpoint base com base no ambiente (sandbox ou produção)
        base_url = "https://api-public.sandbox.exchange.coinbase.com" if sandbox else "https://api.exchange.coinbase.com"
        
        # Inicializa o cliente ccxt para a Coinbase Pro
        self._exchange = ccxt.coinbasepro({
            'apiKey': api_key,
            'secret': api_secret,
            'password': api_passphrase,
            'enableRateLimit': True,
            'urls': {
                'api': base_url
            }
        })
        
        # Configurações do WebSocket
        self._ws_base_url = "wss://ws-feed-public.sandbox.exchange.coinbase.com" if sandbox else "wss://ws-feed.exchange.coinbase.com"
        self._ws_clients: Dict[str, WebSocketClient] = {}
        
        # Callbacks para eventos WebSocket
        self._orderbook_callbacks: Dict[str, Callable[[OrderBook], Awaitable[None]]] = {}
        self._trade_callbacks: Dict[str, Callable[[Trade], Awaitable[None]]] = {}
        self._candle_callbacks: Dict[str, Dict[TimeFrame, Callable[[Candle], Awaitable[None]]]] = {}
        
        # Cache de dados de mercado
        self._markets: Dict[str, Dict] = {}
        self._trading_pairs: Set[str] = set()
        self._local_orderbooks: Dict[str, Dict] = {}
        
        # Task para manter o orderbook atualizado
        self._orderbook_maintenance_tasks: Dict[str, asyncio.Task] = {}
        
        # WebSocket principal para subscriptions
        self._main_ws_client: Optional[WebSocketClient] = None
        self._subscriptions: Dict[str, List[Dict[str, Any]]] = {
            "level2": [],
            "ticker": [],
            "matches": [],
            "heartbeat": []
        }
        
        # Flags para controle de estado
        self._initialized = False
        self._ws_connected = False
    
    @property
    def name(self) -> str:
        """
        Retorna o nome da exchange.
        
        Returns:
            str: Nome da exchange
        """
        return "coinbase"
    
    @property
    def supported_trading_pairs(self) -> Set[str]:
        """
        Retorna o conjunto de pares de negociação suportados pela exchange.
        
        Returns:
            Set[str]: Conjunto de pares de negociação suportados
        """
        return self._trading_pairs
    
    async def initialize(self) -> None:
        """
        Inicializa o adaptador de exchange.
        
        Carrega dados de mercado e inicializa recursos necessários.
        """
        if self._initialized:
            return
            
        try:
            logger.info("Inicializando adaptador Coinbase Pro")
            
            # Carrega os mercados via ccxt
            self._markets = await self._exchange.load_markets(reload=True)
            
            # Extrai os pares de negociação padronizados
            self._trading_pairs = {symbol for symbol in self._markets.keys() if '/' in symbol}
            
            # Inicializa o WebSocket principal se necessário
            if self._api_key and self._api_secret and self._api_passphrase:
                await self._initialize_main_websocket()
            
            logger.info(f"Adaptador Coinbase Pro inicializado com {len(self._trading_pairs)} pares de negociação")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar adaptador Coinbase Pro: {str(e)}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """
        Finaliza o adaptador de exchange.
        
        Libera recursos e fecha conexões.
        """
        if not self._initialized:
            return
            
        try:
            logger.info("Finalizando adaptador Coinbase Pro")
            
            # Cancela todas as tasks de manutenção de orderbook
            for task in self._orderbook_maintenance_tasks.values():
                if not task.done():
                    task.cancel()
            
            # Fecha o WebSocket principal
            if self._main_ws_client:
                await self._main_ws_client.close()
                self._main_ws_client = None
            
            # Fecha todos os clientes WebSocket
            for ws_client in self._ws_clients.values():
                await ws_client.close()
            
            # Fecha o cliente ccxt
            await self._exchange.close()
            
            # Limpa estruturas de dados
            self._ws_clients.clear()
            self._orderbook_callbacks.clear()
            self._trade_callbacks.clear()
            self._candle_callbacks.clear()
            self._local_orderbooks.clear()
            self._orderbook_maintenance_tasks.clear()
            self._subscriptions = {
                "level2": [],
                "ticker": [],
                "matches": [],
                "heartbeat": []
            }
            
            self._initialized = False
            self._ws_connected = False
            
            logger.info("Adaptador Coinbase Pro finalizado")
            
        except Exception as e:
            logger.error(f"Erro ao finalizar adaptador Coinbase Pro: {str(e)}", exc_info=True)
            raise
    
    async def fetch_trading_pairs(self) -> List[str]:
        """
        Obtém a lista de pares de negociação disponíveis na exchange.
        
        Returns:
            List[str]: Lista de pares de negociação no formato padronizado (ex: BTC/USDT)
        """
        if not self._initialized:
            await self.initialize()
        
        # Recarrega os mercados para garantir dados atualizados
        self._markets = await self._exchange.load_markets(reload=True)
        
        # Extrai os pares de negociação padronizados
        self._trading_pairs = {symbol for symbol in self._markets.keys() if '/' in symbol}
        
        return sorted(list(self._trading_pairs))
    
    async def fetch_ticker(self, trading_pair: str) -> Dict[str, Any]:
        """
        Obtém informações de ticker para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            
        Returns:
            Dict[str, Any]: Informações de ticker no formato padronizado
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Obtém o ticker via ccxt
        ticker = await self._exchange.fetch_ticker(trading_pair)
        
        # Formata a resposta
        return {
            "exchange": self.name,
            "trading_pair": trading_pair,
            "timestamp": datetime.fromtimestamp(ticker['timestamp'] / 1000),
            "last": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "high": ticker['high'],
            "low": ticker['low'],
            "volume": ticker['volume'],
            "change": ticker['change'],
            "percentage": ticker['percentage'],
            "average": ticker['average'],
            "vwap": ticker.get('vwap'),
            "open": ticker.get('open'),
            "close": ticker.get('close'),
            "last_trade_id": ticker.get('info', {}).get('trade_id'),
            "trades": None  # Coinbase não fornece o número de trades
        }
    
    async def fetch_orderbook(self, trading_pair: str, depth: int = 20) -> OrderBook:
        """
        Obtém o livro de ofertas para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            depth: Profundidade do orderbook a ser obtido
            
        Returns:
            OrderBook: Entidade OrderBook com os dados do livro de ofertas
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Limita a profundidade a valores permitidos pela Coinbase Pro
        adjusted_depth = min(depth, 50)  # Coinbase Pro tem limite de 50 níveis
        
        # Obtém o orderbook via ccxt
        orderbook_data = await self._exchange.fetch_order_book(trading_pair, limit=adjusted_depth)
        
        # Converte para o formato da entidade OrderBook
        bids = [
            OrderBookLevel(
                price=Decimal(str(price)),
                amount=Decimal(str(amount))
            ) for price, amount in orderbook_data['bids']
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(price)),
                amount=Decimal(str(amount))
            ) for price, amount in orderbook_data['asks']
        ]
        
        return OrderBook(
            exchange=self.name,
            trading_pair=trading_pair,
            timestamp=datetime.fromtimestamp(orderbook_data['timestamp'] / 1000) if orderbook_data['timestamp'] else datetime.utcnow(),
            bids=bids,
            asks=asks,
            raw_data=orderbook_data
        )
    
    async def fetch_trades(
        self, 
        trading_pair: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """
        Obtém transações recentes para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            since: Timestamp a partir do qual obter transações (opcional)
            limit: Número máximo de transações a retornar (opcional)
            
        Returns:
            List[Trade]: Lista de entidades Trade com os dados das transações
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Converte datetime para timestamp em milissegundos
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Obtém as transações via ccxt
        trades_data = await self._exchange.fetch_trades(
            trading_pair, 
            since=since_ts, 
            limit=limit
        )
        
        # Converte para o formato da entidade Trade
        trades = []
        for trade_data in trades_data:
            side = TradeSide.BUY if trade_data['side'] == 'buy' else TradeSide.SELL
            
            trade = Trade(
                id=str(trade_data['id']),
                exchange=self.name,
                trading_pair=trading_pair,
                price=Decimal(str(trade_data['price'])),
                amount=Decimal(str(trade_data['amount'])),
                cost=Decimal(str(trade_data['cost'])),
                timestamp=datetime.fromtimestamp(trade_data['timestamp'] / 1000),
                side=side,
                taker=True,  # Coinbase Pro não distingue entre taker e maker nas transações públicas
                raw_data=trade_data
            )
            
            trades.append(trade)
        
        return trades
    
    async def fetch_candles(
        self, 
        trading_pair: str, 
        timeframe: TimeFrame,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Obtém velas históricas para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
            since: Timestamp a partir do qual obter velas (opcional)
            limit: Número máximo de velas a retornar (opcional)
            
        Returns:
            List[Candle]: Lista de entidades Candle com os dados das velas
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Converte o timeframe para o formato da Coinbase Pro
        coinbase_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not coinbase_timeframe:
            raise ValueError(f"Timeframe não suportado pela Coinbase Pro: {timeframe}")
        
        # Converte datetime para timestamp em milissegundos
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Obtém as velas via ccxt
        candles_data = await self._exchange.fetch_ohlcv(
            trading_pair, 
            timeframe=coinbase_timeframe, 
            since=since_ts, 
            limit=limit
        )
        
        # Converte para o formato da entidade Candle
        candles = []
        for candle_data in candles_data:
            timestamp = datetime.fromtimestamp(candle_data[0] / 1000)
            open_price = Decimal(str(candle_data[1]))
            high_price = Decimal(str(candle_data[2]))
            low_price = Decimal(str(candle_data[3]))
            close_price = Decimal(str(candle_data[4]))
            volume = Decimal(str(candle_data[5]))
            
            candle = Candle(
                exchange=self.name,
                trading_pair=trading_pair,
                timestamp=timestamp,
                timeframe=timeframe,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                raw_data=candle_data
            )
            
            candles.append(candle)
        
        return candles
    
    async def subscribe_orderbook(
        self, 
        trading_pair: str, 
        callback: Callable[[OrderBook], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de orderbook em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            callback: Função assíncrona a ser chamada com cada atualização de orderbook
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._orderbook_callbacks[trading_pair] = callback
        
        # Inicializa o orderbook local com uma snapshot
        await self._initialize_orderbook(trading_pair)
        
        # Verifica se o WebSocket principal está inicializado
        if not self._main_ws_client:
            await self._initialize_main_websocket()
        
        # Prepara a subscrição para o orderbook
        product_id = trading_pair.replace('/', '-')
        subscription = {
            "product_id": product_id
        }
        
        # Adiciona à lista de subscrições
        if subscription not in self._subscriptions["level2"]:
            self._subscriptions["level2"].append(subscription)
            
            # Atualiza as subscrições no WebSocket
            await self._update_subscriptions()
        
        # Inicia a task de manutenção do orderbook
        if trading_pair not in self._orderbook_maintenance_tasks:
            task = asyncio.create_task(self._maintain_orderbook(trading_pair))
            self._orderbook_maintenance_tasks[trading_pair] = task
    
    async def subscribe_trades(
        self, 
        trading_pair: str, 
        callback: Callable[[Trade], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de trades em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            callback: Função assíncrona a ser chamada com cada nova transação
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._trade_callbacks[trading_pair] = callback
        
        # Verifica se o WebSocket principal está inicializado
        if not self._main_ws_client:
            await self._initialize_main_websocket()