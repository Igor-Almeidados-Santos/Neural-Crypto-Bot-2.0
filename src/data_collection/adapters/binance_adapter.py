"""
Adaptador para a exchange Binance.

Este módulo implementa o adaptador para a exchange Binance, seguindo
a interface comum definida para todos os adaptadores de exchanges.
"""
import asyncio
import hmac
import hashlib
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable
from urllib.parse import urlencode

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


class BinanceAdapter(ExchangeAdapterInterface):
    """
    Adaptador para a exchange Binance.
    
    Implementa a interface ExchangeAdapterInterface para a exchange Binance,
    utilizando a biblioteca ccxt para requisições REST e WebSockets nativos
    da Binance para streaming de dados.
    """
    
    # Mapeamento de timeframes da Binance para o formato padronizado
    TIMEFRAME_MAP = {
        "1m": TimeFrame.MINUTE_1,
        "3m": TimeFrame.MINUTE_3,
        "5m": TimeFrame.MINUTE_5,
        "15m": TimeFrame.MINUTE_15,
        "30m": TimeFrame.MINUTE_30,
        "1h": TimeFrame.HOUR_1,
        "2h": TimeFrame.HOUR_2,
        "4h": TimeFrame.HOUR_4,
        "6h": TimeFrame.HOUR_6,
        "8h": TimeFrame.HOUR_8,
        "12h": TimeFrame.HOUR_12,
        "1d": TimeFrame.DAY_1,
        "3d": TimeFrame.DAY_3,
        "1w": TimeFrame.WEEK_1,
        "1M": TimeFrame.MONTH_1
    }
    
    # Mapeamento reverso, do formato padronizado para o formato da Binance
    REVERSE_TIMEFRAME_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}
    
    def validate_trading_pair(self, trading_pair: str) -> bool:
        """
        Valida se um par de negociação é suportado pela exchange.
        
        Args:
            trading_pair: Par de negociação a ser validado
            
        Returns:
            bool: True se o par é suportado, False caso contrário
        """
        return trading_pair in self._trading_pairs
    
    def get_supported_timeframes(self) -> Dict[str, TimeFrame]:
        """
        Obtém os timeframes suportados pela exchange.
        
        Returns:
            Dict[str, TimeFrame]: Dicionário de timeframes suportados,
            mapeando o código da exchange para o TimeFrame padronizado
        """
        return self.TIMEFRAME_MAP
    
    async def _initialize_orderbook(self, trading_pair: str) -> None:
        """
        Inicializa o orderbook local com uma snapshot da Binance.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        try:
            # Obtém uma snapshot do orderbook via REST API
            orderbook_data = await self._exchange.fetch_order_book(trading_pair, limit=1000)
            
            # Inicializa o orderbook local
            self._local_orderbooks[trading_pair] = {
                "lastUpdateId": orderbook_data.get('nonce', 0),
                "bids": {Decimal(str(price)): Decimal(str(amount)) for price, amount in orderbook_data['bids']},
                "asks": {Decimal(str(price)): Decimal(str(amount)) for price, amount in orderbook_data['asks']},
                "timestamp": datetime.fromtimestamp(orderbook_data['timestamp'] / 1000) if orderbook_data['timestamp'] else datetime.utcnow()
            }
            
            logger.debug(f"Orderbook inicializado para {trading_pair} com {len(orderbook_data['bids'])} bids e {len(orderbook_data['asks'])} asks")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar orderbook para {trading_pair}: {str(e)}", exc_info=True)
            raise
    
    async def _maintain_orderbook(self, trading_pair: str) -> None:
        """
        Mantém o orderbook local atualizado com snapshots periódicas.
        
        Esta função é executada como uma task para garantir que o orderbook
        local não fique dessincronizado por muito tempo.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        while True:
            try:
                # Atualiza o orderbook a cada 30 minutos para evitar dessincronização
                await asyncio.sleep(1800)  # 30 minutos
                
                logger.debug(f"Atualizando orderbook para {trading_pair} com nova snapshot")
                await self._initialize_orderbook(trading_pair)
                
            except asyncio.CancelledError:
                logger.debug(f"Tarefa de manutenção de orderbook para {trading_pair} cancelada")
                break
                
            except Exception as e:
                logger.error(f"Erro na manutenção do orderbook para {trading_pair}: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Espera 1 minuto antes de tentar novamente
    
    async def _handle_orderbook_message(self, message: str) -> None:
        """
        Processa mensagens de atualização de orderbook recebidas via WebSocket.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = orjson.loads(message)
            
            # Verifica se é uma mensagem de orderbook
            if 'e' in data and data['e'] == 'depthUpdate':
                # Extrai o símbolo e converte para o formato padronizado
                symbol = data['s']
                trading_pair = self._exchange.markets_by_id[symbol]['symbol']
                
                # Verifica se temos um orderbook local para este par
                if trading_pair not in self._local_orderbooks:
                    logger.warning(f"Recebida atualização de orderbook para {trading_pair}, mas não há orderbook local inicializado")
                    return
                
                # Atualiza o orderbook local
                local_book = self._local_orderbooks[trading_pair]
                
                # Verifica se a atualização é mais recente que o orderbook local
                if data['u'] <= local_book['lastUpdateId']:
                    return
                
                # Atualiza o ID da última atualização
                local_book['lastUpdateId'] = data['u']
                local_book['timestamp'] = datetime.fromtimestamp(data['E'] / 1000)
                
                # Atualiza as ofertas de compra (bids)
                for bid in data['b']:
                    price = Decimal(str(bid[0]))
                    amount = Decimal(str(bid[1]))
                    
                    if amount == 0:
                        local_book['bids'].pop(price, None)
                    else:
                        local_book['bids'][price] = amount
                
                # Atualiza as ofertas de venda (asks)
                for ask in data['a']:
                    price = Decimal(str(ask[0]))
                    amount = Decimal(str(ask[1]))
                    
                    if amount == 0:
                        local_book['asks'].pop(price, None)
                    else:
                        local_book['asks'][price] = amount
                
                # Converte o orderbook local para a entidade OrderBook
                bids = [
                    OrderBookLevel(price=price, amount=amount)
                    for price, amount in sorted(local_book['bids'].items(), reverse=True)
                ]
                
                asks = [
                    OrderBookLevel(price=price, amount=amount)
                    for price, amount in sorted(local_book['asks'].items())
                ]
                
                orderbook = OrderBook(
                    exchange=self.name,
                    trading_pair=trading_pair,
                    timestamp=local_book['timestamp'],
                    bids=bids,
                    asks=asks,
                    raw_data=data
                )
                
                # Chama o callback registrado para este par
                if trading_pair in self._orderbook_callbacks:
                    callback = self._orderbook_callbacks[trading_pair]
                    asyncio.create_task(callback(orderbook))
        
        except Exception as e:
            logger.error(f"Erro ao processar mensagem de orderbook: {str(e)}", exc_info=True)
    
    async def _handle_trade_message(self, message: str) -> None:
        """
        Processa mensagens de trades recebidas via WebSocket.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = orjson.loads(message)
            
            # Verifica se é uma mensagem de trade
            if 'e' in data and data['e'] == 'trade':
                # Extrai o símbolo e converte para o formato padronizado
                symbol = data['s']
                trading_pair = self._exchange.markets_by_id[symbol]['symbol']
                
                # Cria a entidade Trade
                price = Decimal(str(data['p']))
                amount = Decimal(str(data['q']))
                cost = price * amount
                
                side = TradeSide.BUY if data['m'] is False else TradeSide.SELL
                
                trade = Trade(
                    id=str(data['t']),
                    exchange=self.name,
                    trading_pair=trading_pair,
                    price=price,
                    amount=amount,
                    cost=cost,
                    timestamp=datetime.fromtimestamp(data['T'] / 1000),
                    side=side,
                    taker=True,  # A Binance só envia trades de takers via WebSocket
                    raw_data=data
                )
                
                # Chama o callback registrado para este par
                if trading_pair in self._trade_callbacks:
                    callback = self._trade_callbacks[trading_pair]
                    asyncio.create_task(callback(trade))
        
        except Exception as e:
            logger.error(f"Erro ao processar mensagem de trade: {str(e)}", exc_info=True)
    
    async def _handle_candle_message(self, message: str) -> None:
        """
        Processa mensagens de velas recebidas via WebSocket.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = orjson.loads(message)
            
            # Verifica se é uma mensagem de vela
            if 'e' in data and data['e'] == 'kline':
                # Extrai o símbolo e converte para o formato padronizado
                symbol = data['s']
                trading_pair = self._exchange.markets_by_id[symbol]['symbol']
                
                # Extrai os dados da vela
                kline = data['k']
                
                # Converte o timeframe da Binance para o formato padronizado
                binance_timeframe = kline['i']
                timeframe = self.TIMEFRAME_MAP.get(binance_timeframe)
                
                if not timeframe:
                    logger.warning(f"Timeframe não suportado: {binance_timeframe}")
                    return
                
                # Cria a entidade Candle
                open_price = Decimal(str(kline['o']))
                high_price = Decimal(str(kline['h']))
                low_price = Decimal(str(kline['l']))
                close_price = Decimal(str(kline['c']))
                volume = Decimal(str(kline['v']))
                
                candle = Candle(
                    exchange=self.name,
                    trading_pair=trading_pair,
                    timestamp=datetime.fromtimestamp(kline['t'] / 1000),
                    timeframe=timeframe,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    trades=kline.get('n'),
                    raw_data=kline
                )
                
                # Chama o callback registrado para este par e timeframe
                if (trading_pair in self._candle_callbacks and 
                    timeframe in self._candle_callbacks[trading_pair]):
                    callback = self._candle_callbacks[trading_pair][timeframe]
                    asyncio.create_task(callback(candle))
        
        except Exception as e:
            logger.error(f"Erro ao processar mensagem de vela: {str(e)}", exc_info=True) __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Inicializa o adaptador para a Binance.
        
        Args:
            api_key: Chave de API da Binance (opcional)
            api_secret: Chave secreta da API da Binance (opcional)
            testnet: Se True, utiliza o ambiente de testes da Binance
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        
        # Define o endpoint base com base no ambiente (testnet ou produção)
        base_url = "https://testnet.binance.vision/api" if testnet else "https://api.binance.com"
        
        # Inicializa o cliente ccxt para a Binance
        self._exchange = ccxt.binance({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'recvWindow': 10000,
                'warnOnFetchOpenOrdersWithoutSymbol': False,
            }
        })
        
        # Configurações do WebSocket
        self._ws_base_url = "wss://testnet.binance.vision/ws" if testnet else "wss://stream.binance.com:9443/ws"
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
        return "binance"
    
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
            logger.info("Inicializando adaptador Binance")
            
            # Carrega os mercados via ccxt
            self._markets = await self._exchange.load_markets(reload=True)
            
            # Extrai os pares de negociação padronizados
            self._trading_pairs = {symbol for symbol in self._markets.keys() if '/' in symbol}
            
            logger.info(f"Adaptador Binance inicializado com {len(self._trading_pairs)} pares de negociação")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar adaptador Binance: {str(e)}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """
        Finaliza o adaptador de exchange.
        
        Libera recursos e fecha conexões.
        """
        if not self._initialized:
            return
            
        try:
            logger.info("Finalizando adaptador Binance")
            
            # Cancela todas as tasks de manutenção de orderbook
            for task in self._orderbook_maintenance_tasks.values():
                if not task.done():
                    task.cancel()
            
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
            
            self._initialized = False
            self._ws_connected = False
            
            logger.info("Adaptador Binance finalizado")
            
        except Exception as e:
            logger.error(f"Erro ao finalizar adaptador Binance: {str(e)}", exc_info=True)
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
            "last_trade_id": ticker.get('info', {}).get('lastId'),
            "trades": ticker.get('info', {}).get('count')
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
        
        # Limita a profundidade a valores permitidos pela Binance
        allowed_depths = [5, 10, 20, 50, 100, 500, 1000, 5000]
        adjusted_depth = min([d for d in allowed_depths if d >= depth], default=100)
        
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
                taker=trade_data['takerOrMaker'] == 'taker',
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
        
        # Converte o timeframe para o formato da Binance
        binance_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not binance_timeframe:
            raise ValueError(f"Timeframe não suportado pela Binance: {timeframe}")
        
        # Converte datetime para timestamp em milissegundos
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Obtém as velas via ccxt
        candles_data = await self._exchange.fetch_ohlcv(
            trading_pair, 
            timeframe=binance_timeframe, 
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
        
        # Converte o par de negociação para o formato da Binance
        symbol = trading_pair.replace('/', '').lower()
        stream_name = f"{symbol}@depth"
        
        # Armazena o callback
        self._orderbook_callbacks[trading_pair] = callback
        
        # Inicializa o orderbook local com uma snapshot
        await self._initialize_orderbook(trading_pair)
        
        # Cria um WebSocketClient se ainda não existir
        if stream_name not in self._ws_clients:
            url = f"{self._ws_base_url}/{stream_name}"
            
            ws_client = WebSocketClient(url)
            ws_client.on_message(self._handle_orderbook_message)
            
            await ws_client.connect()
            self._ws_clients[stream_name] = ws_client
            
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
        
        # Converte o par de negociação para o formato da Binance
        symbol = trading_pair.replace('/', '').lower()
        stream_name = f"{symbol}@trade"
        
        # Armazena o callback
        self._trade_callbacks[trading_pair] = callback
        
        # Cria um WebSocketClient se ainda não existir
        if stream_name not in self._ws_clients:
            url = f"{self._ws_base_url}/{stream_name}"
            
            ws_client = WebSocketClient(url)
            ws_client.on_message(self._handle_trade_message)
            
            await ws_client.connect()
            self._ws_clients[stream_name] = ws_client
    
    async def subscribe_candles(
        self, 
        trading_pair: str, 
        timeframe: TimeFrame,
        callback: Callable[[Candle], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de velas em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
            callback: Função assíncrona a ser chamada com cada nova vela
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Converte o timeframe para o formato da Binance
        binance_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not binance_timeframe:
            raise ValueError(f"Timeframe não suportado pela Binance: {timeframe}")
        
        # Converte o par de negociação para o formato da Binance
        symbol = trading_pair.replace('/', '').lower()
        stream_name = f"{symbol}@kline_{binance_timeframe}"
        
        # Armazena o callback
        if trading_pair not in self._candle_callbacks:
            self._candle_callbacks[trading_pair] = {}
            
        self._candle_callbacks[trading_pair][timeframe] = callback
        
        # Cria um WebSocketClient se ainda não existir
        if stream_name not in self._ws_clients:
            url = f"{self._ws_base_url}/{stream_name}"
            
            ws_client = WebSocketClient(url)
            ws_client.on_message(self._handle_candle_message)
            
            await ws_client.connect()
            self._ws_clients[stream_name] = ws_client
    
    async def unsubscribe_orderbook(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de orderbook.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        if not self._initialized:
            return
            
        # Converte o par de negociação para o formato da Binance
        symbol = trading_pair.replace('/', '').lower()
        stream_name = f"{symbol}@depth"
        
        # Remove o callback
        if trading_pair in self._orderbook_callbacks:
            del self._orderbook_callbacks[trading_pair]
        
        # Fecha o WebSocketClient se existir
        if stream_name in self._ws_clients:
            await self._ws_clients[stream_name].close()
            del self._ws_clients[stream_name]
        
        # Cancela a task de manutenção do orderbook
        if trading_pair in self._orderbook_maintenance_tasks:
            task = self._orderbook_maintenance_tasks[trading_pair]
            if not task.done():
                task.cancel()
            del self._orderbook_maintenance_tasks[trading_pair]
        
        # Remove o orderbook local
        if trading_pair in self._local_orderbooks:
            del self._local_orderbooks[trading_pair]
    
    async def unsubscribe_trades(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de trades.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        if not self._initialized:
            return
            
        # Converte o par de negociação para o formato da Binance
        symbol = trading_pair.replace('/', '').lower()
        stream_name = f"{symbol}@trade"
        
        # Remove o callback
        if trading_pair in self._trade_callbacks:
            del self._trade_callbacks[trading_pair]
        
        # Fecha o WebSocketClient se existir
        if stream_name in self._ws_clients:
            await self._ws_clients[stream_name].close()
            del self._ws_clients[stream_name]
    
    async def unsubscribe_candles(self, trading_pair: str, timeframe: TimeFrame) -> None:
        """
        Cancela a subscrição para atualizações de velas.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
        """
        if not self._initialized:
            return
            
        # Converte o timeframe para o formato da Binance
        binance_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not binance_timeframe:
            return
        
        # Converte o par de negociação para o formato da Binance
        symbol = trading_pair.replace('/', '').lower()
        stream_name = f"{symbol}@kline_{binance_timeframe}"
        
        # Remove o callback
        if trading_pair in self._candle_callbacks and timeframe in self._candle_callbacks[trading_pair]:
            del self._candle_callbacks[trading_pair][timeframe]
            
            if not self._candle_callbacks[trading_pair]:
                del self._candle_callbacks[trading_pair]
        
        # Fecha o WebSocketClient se existir
        if stream_name in self._ws_clients:
            await self._ws_clients[stream_name].close()
            del self._ws_clients[stream_name]
    
