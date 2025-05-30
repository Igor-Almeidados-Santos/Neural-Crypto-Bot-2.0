"""
Adaptador para a exchange Kraken.

Este módulo implementa o adaptador para a exchange Kraken, seguindo
a interface comum definida para todos os adaptadores de exchanges.
"""
import asyncio
import base64
import hmac
import hashlib
import json
import logging
import time
import urllib.parse
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable

import ccxt.async_support as ccxt
from websockets.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed
import orjson

from src.data_collection.adapters.exchange_adapter_interface import ExchangeAdapterInterface
from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from src.data_collection.domain.entities.trade import Trade, TradeSide
from src.data_collection.domain.entities.funding_rate import FundingRate
from src.data_collection.domain.entities.liquidation import Liquidation
from src.data_collection.infrastructure.websocket_client import WebSocketClient


logger = logging.getLogger(__name__)


class KrakenAdapter(ExchangeAdapterInterface):
    """
    Adaptador para a exchange Kraken.
    
    Implementa a interface ExchangeAdapterInterface para a exchange Kraken,
    utilizando a biblioteca ccxt para requisições REST e WebSockets nativos
    da Kraken para streaming de dados.
    """
    
    # Mapeamento de timeframes da Kraken para o formato padronizado
    TIMEFRAME_MAP = {
        "1": TimeFrame.MINUTE_1,
        "5": TimeFrame.MINUTE_5,
        "15": TimeFrame.MINUTE_15,
        "30": TimeFrame.MINUTE_30,
        "60": TimeFrame.HOUR_1,
        "240": TimeFrame.HOUR_4,
        "1440": TimeFrame.DAY_1,
        "10080": TimeFrame.WEEK_1,
        "21600": TimeFrame.MONTH_1
    }
    
    # Mapeamento reverso, do formato padronizado para o formato da Kraken
    REVERSE_TIMEFRAME_MAP = {v: k for k, v in TIMEFRAME_MAP.items()}
    
    # Mapeamento de pares de moedas da Kraken para formato padronizado
    ASSET_MAPPING = {
        'XXBT': 'BTC',
        'XETH': 'ETH',
        'XXRP': 'XRP',
        'XXLM': 'XLM',
        'XLTC': 'LTC',
        'XETC': 'ETC',
        'XZEC': 'ZEC',
        'XXMR': 'XMR',
        'DASH': 'DASH',
        'ZUSD': 'USD',
        'ZEUR': 'EUR',
        'ZJPY': 'JPY',
        'ZGBP': 'GBP',
        'ZCAD': 'CAD',
        'ZAUD': 'AUD'
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        testnet: bool = False
    ):
        """
        Inicializa o adaptador para a Kraken.
        
        Args:
            api_key: Chave de API da Kraken (opcional)
            api_secret: Chave secreta da API da Kraken (opcional)
            testnet: Se True, utiliza o ambiente de testes da Kraken
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet
        
        # Inicializa o cliente ccxt para a Kraken
        self._exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True
            }
        })
        
        # Define os URLs dos WebSockets
        if testnet:
            self._ws_public_url = "wss://beta-ws.kraken.com"
            self._ws_private_url = "wss://beta-ws-auth.kraken.com"
        else:
            self._ws_public_url = "wss://ws.kraken.com"
            self._ws_private_url = "wss://ws-auth.kraken.com"
        
        # WebSocket clients
        self._public_ws: Optional[WebSocketClient] = None
        self._private_ws: Optional[WebSocketClient] = None
        
        # Callbacks para eventos WebSocket
        self._orderbook_callbacks: Dict[str, Callable[[OrderBook], Awaitable[None]]] = {}
        self._trade_callbacks: Dict[str, Callable[[Trade], Awaitable[None]]] = {}
        self._candle_callbacks: Dict[str, Dict[TimeFrame, Callable[[Candle], Awaitable[None]]]] = {}
        self._funding_rate_callbacks: Dict[str, Callable[[FundingRate], Awaitable[None]]] = {}
        self._liquidation_callbacks: Dict[str, Callable[[Liquidation], Awaitable[None]]] = {}
        
        # Cache de dados de mercado
        self._markets: Dict[str, Dict] = {}
        self._trading_pairs: Set[str] = set()
        self._local_orderbooks: Dict[str, Dict] = {}
        
        # Subscriptions ativas
        self._subscriptions: Dict[str, List[Dict[str, Any]]] = {}
        
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
        return "kraken"
    
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
            logger.info("Inicializando adaptador Kraken")
            
            # Carrega os mercados via ccxt
            self._markets = await self._exchange.load_markets(reload=True)
            
            # Extrai os pares de negociação padronizados
            self._trading_pairs = {symbol for symbol in self._markets.keys() if '/' in symbol}
            
            # Inicializa o WebSocket público
            await self._initialize_public_websocket()
            
            # Inicializa o WebSocket privado se as credenciais foram fornecidas
            if self._api_key and self._api_secret:
                await self._initialize_private_websocket()
            
            logger.info(f"Adaptador Kraken inicializado com {len(self._trading_pairs)} pares de negociação")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar adaptador Kraken: {str(e)}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """
        Finaliza o adaptador de exchange.
        
        Libera recursos e fecha conexões.
        """
        if not self._initialized:
            return
            
        try:
            logger.info("Finalizando adaptador Kraken")
            
            # Fecha o WebSocket público
            if self._public_ws:
                await self._public_ws.close()
                self._public_ws = None
            
            # Fecha o WebSocket privado
            if self._private_ws:
                await self._private_ws.close()
                self._private_ws = None
            
            # Fecha o cliente ccxt
            await self._exchange.close()
            
            # Limpa estruturas de dados
            self._orderbook_callbacks.clear()
            self._trade_callbacks.clear()
            self._candle_callbacks.clear()
            self._funding_rate_callbacks.clear()
            self._liquidation_callbacks.clear()
            self._local_orderbooks.clear()
            
            self._initialized = False
            self._ws_connected = False
            
            logger.info("Adaptador Kraken finalizado")
            
        except Exception as e:
            logger.error(f"Erro ao finalizar adaptador Kraken: {str(e)}", exc_info=True)
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
            trading_pair: Par de negociação (ex: BTC/USD)
            
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
            "last_trade_id": ticker.get('info', {}).get('last_trade_id'),
            "trades": None  # Kraken não fornece o número de trades no ticker
        }
    
    async def fetch_orderbook(self, trading_pair: str, depth: int = 20) -> OrderBook:
        """
        Obtém o livro de ofertas para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
            depth: Profundidade do orderbook a ser obtido
            
        Returns:
            OrderBook: Entidade OrderBook com os dados do livro de ofertas
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Obtém o orderbook via ccxt
        orderbook_data = await self._exchange.fetch_order_book(trading_pair, limit=depth)
        
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
            trading_pair: Par de negociação (ex: BTC/USD)
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
                taker=True,  # Kraken não distingue entre taker e maker nas transações públicas
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
            trading_pair: Par de negociação (ex: BTC/USD)
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
        
        # Converte o timeframe para o formato da Kraken
        kraken_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not kraken_timeframe:
            raise ValueError(f"Timeframe não suportado pela Kraken: {timeframe}")
        
        # Converte datetime para timestamp em milissegundos
        since_ts = int(since.timestamp() * 1000) if since else None
        
        # Obtém as velas via ccxt
        candles_data = await self._exchange.fetch_ohlcv(
            trading_pair, 
            timeframe=kraken_timeframe, 
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
    
    async def fetch_funding_rates(
        self,
        trading_pair: Optional[str] = None
    ) -> List[FundingRate]:
        """
        Obtém taxas de financiamento para pares de negociação perpétuos.
        
        Args:
            trading_pair: Par de negociação específico (opcional)
            
        Returns:
            List[FundingRate]: Lista de entidades FundingRate
        """
        if not self._initialized:
            await self.initialize()
            
        # Kraken tem um endpoint específico para taxas de financiamento
        try:
            if trading_pair:
                # Verifica se o par é suportado
                if not self.validate_trading_pair(trading_pair):
                    raise ValueError(f"Par de negociação inválido: {trading_pair}")
                
                # Converte para o formato da Kraken (ex: BTC/USD -> XBTUSD)
                kraken_symbol = self._get_kraken_symbol(trading_pair)
                
                # Obtém as taxas de financiamento apenas para este par
                response = await self._exchange.fetch_funding_rate(kraken_symbol)
                funding_rates = [self._parse_funding_rate(response, trading_pair)]
            else:
                # Obtém as taxas de financiamento para todos os pares
                response = await self._exchange.fetch_funding_rates()
                funding_rates = [
                    self._parse_funding_rate(data, pair)
                    for pair, data in response.items()
                    if data.get('fundingRate') is not None  # Alguns pares podem não ter taxa de financiamento
                ]
            
            return funding_rates
        except Exception as e:
            logger.error(f"Erro ao obter taxas de financiamento: {str(e)}", exc_info=True)
            return []
    
    async def fetch_liquidations(
        self,
        trading_pair: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Liquidation]:
        """
        Obtém eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação específico (opcional)
            since: Timestamp a partir do qual obter liquidações (opcional)
            limit: Número máximo de liquidações a retornar (opcional)
            
        Returns:
            List[Liquidation]: Lista de entidades Liquidation
        """
        # Kraken não fornece um endpoint público para liquidações
        # Esta é uma implementação simulada para manter a interface consistente
        logger.warning("Kraken não fornece dados públicos de liquidação via API")
        return []
    
    async def subscribe_orderbook(
        self, 
        trading_pair: str, 
        callback: Callable[[OrderBook], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de orderbook em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
            callback: Função assíncrona a ser chamada com cada atualização de orderbook
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._orderbook_callbacks[trading_pair] = callback
        
        # Converte para o formato da Kraken
        kraken_symbol = self._get_kraken_symbol(trading_pair)
        
        # Prepara a mensagem de subscrição
        subscription = {
            "name": "book",
            "depth": 25  # Kraken suporta 10, 25, 100, 500, 1000
        }
        
        # Envia a subscrição
        await self._subscribe_public(subscription, [kraken_symbol])
        
        # Inicializa o orderbook local
        await self._initialize_orderbook(trading_pair)
    
    async def subscribe_trades(
        self, 
        trading_pair: str, 
        callback: Callable[[Trade], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de trades em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
            callback: Função assíncrona a ser chamada com cada nova transação
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._trade_callbacks[trading_pair] = callback
        
        # Converte para o formato da Kraken
        kraken_symbol = self._get_kraken_symbol(trading_pair)
        
        # Prepara a mensagem de subscrição
        subscription = {
            "name": "trade"
        }
        
        # Envia a subscrição
        await self._subscribe_public(subscription, [kraken_symbol])
    
    async def subscribe_candles(
        self, 
        trading_pair: str, 
        timeframe: TimeFrame,
        callback: Callable[[Candle], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de velas em tempo real.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
            timeframe: Intervalo de tempo das velas
            callback: Função assíncrona a ser chamada com cada nova vela
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Verifica se o timeframe é suportado
        kraken_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not kraken_timeframe:
            raise ValueError(f"Timeframe não suportado pela Kraken: {timeframe}")
        
        # Armazena o callback
        if trading_pair not in self._candle_callbacks:
            self._candle_callbacks[trading_pair] = {}
            
        self._candle_callbacks[trading_pair][timeframe] = callback
        
        # Converte para o formato da Kraken
        kraken_symbol = self._get_kraken_symbol(trading_pair)
        
        # Prepara a mensagem de subscrição
        subscription = {
            "name": "ohlc",
            "interval": int(kraken_timeframe)
        }
        
        # Envia a subscrição
        await self._subscribe_public(subscription, [kraken_symbol])
    
    async def subscribe_funding_rates(
        self,
        trading_pair: str,
        callback: Callable[[FundingRate], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
            callback: Função assíncrona a ser chamada com cada atualização
        """
        if not self._initialized:
            await self.initialize()
            
        if not self.validate_trading_pair(trading_pair):
            raise ValueError(f"Par de negociação inválido: {trading_pair}")
        
        # Armazena o callback
        self._funding_rate_callbacks[trading_pair] = callback
        
        # Kraken não oferece WebSocket para taxas de financiamento
        # Vamos fazer polling do endpoint REST periodicamente
        logger.info(f"Kraken não oferece streaming de taxas de financiamento. Configurando polling periódico para {trading_pair}")
        
        # Inicia uma task em background para fazer polling
        asyncio.create_task(self._funding_rate_polling_loop(trading_pair))
    
    async def subscribe_liquidations(
        self,
        trading_pair: Optional[str] = None,
        callback: Callable[[Liquidation], Awaitable[None]] = None
    ) -> None:
        """
        Subscreve para eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação (opcional)
            callback: Função assíncrona a ser chamada com cada liquidação
        """
        # Kraken não fornece dados de liquidação via WebSocket
        logger.warning("Kraken não fornece dados de liquidação via WebSocket")
    
    async def unsubscribe_orderbook(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de orderbook.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
        """
        if not self._initialized:
            return
            
        # Remove o callback
        if trading_pair in self._orderbook_callbacks:
            del self._orderbook_callbacks[trading_pair]
        
        # Converte para o formato da Kraken
        kraken_symbol = self._get_kraken_symbol(trading_pair)
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "name": "book",
            "depth": 25
        }
        
        # Envia o cancelamento
        await self._unsubscribe_public(subscription, [kraken_symbol])
        
        # Remove o orderbook local
        if trading_pair in self._local_orderbooks:
            del self._local_orderbooks[trading_pair]
    
    async def unsubscribe_trades(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de trades.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
        """
        if not self._initialized:
            return
            
        # Remove o callback
        if trading_pair in self._trade_callbacks:
            del self._trade_callbacks[trading_pair]
        
        # Converte para o formato da Kraken
        kraken_symbol = self._get_kraken_symbol(trading_pair)
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "name": "trade"
        }
        
        # Envia o cancelamento
        await self._unsubscribe_public(subscription, [kraken_symbol])
    
    async def unsubscribe_candles(self, trading_pair: str, timeframe: TimeFrame) -> None:
        """
        Cancela a subscrição para atualizações de velas.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
            timeframe: Intervalo de tempo das velas
        """
        if not self._initialized:
            return
            
        # Verifica se o timeframe é suportado
        kraken_timeframe = self.REVERSE_TIMEFRAME_MAP.get(timeframe)
        if not kraken_timeframe:
            logger.warning(f"Timeframe não suportado para cancelamento: {timeframe}")
            return
        
        # Remove o callback
        if trading_pair in self._candle_callbacks and timeframe in self._candle_callbacks[trading_pair]:
            del self._candle_callbacks[trading_pair][timeframe]
            
            if not self._candle_callbacks[trading_pair]:
                del self._candle_callbacks[trading_pair]
        
        # Converte para o formato da Kraken
        kraken_symbol = self._get_kraken_symbol(trading_pair)
        
        # Prepara a mensagem de cancelamento
        subscription = {
            "name": "ohlc",
            "interval": int(kraken_timeframe)
        }
        
        # Envia o cancelamento
        await self._unsubscribe_public(subscription, [kraken_symbol])
    
    async def unsubscribe_funding_rates(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USD)
        """
        if not self._initialized:
            return
            
        # Remove o callback
        if trading_pair in self._funding_rate_callbacks:
            del self._funding_rate_callbacks[trading_pair]
    
    async def unsubscribe_liquidations(self, trading_pair: Optional[str] = None) -> None:
        """
        Cancela a subscrição para eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação (opcional)
        """
        # Kraken não fornece dados de liquidação via WebSocket
        pass
    
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
    
    def _get_kraken_symbol(self, trading_pair: str) -> str:
        """
        Converte um par de negociação do formato padronizado para o formato da Kraken.
        
        Args:
            trading_pair: Par de negociação no formato padronizado (ex: BTC/USD)
            
        Returns:
            str: Par de negociação no formato da Kraken (ex: XBT/USD)
        """
        if trading_pair in self._markets:
            return self._markets[trading_pair]['id']
            
        # Tenta converter manualmente se não encontrar no mercado
        base, quote = trading_pair.split('/')
        
        # Mapeamento especial para BTC
        if base == 'BTC':
            base = 'XBT'
            
        # Mapeamento para pares especiais da Kraken
        for kraken_symbol, standard_symbol in self.ASSET_MAPPING.items():
            base = base.replace(standard_symbol, kraken_symbol)
            quote = quote.replace(standard_symbol, kraken_symbol)
            
        return f"{base}{quote}"
    
    def _parse_kraken_symbol(self, kraken_symbol: str) -> str:
        """
        Converte um par de negociação do formato da Kraken para o formato padronizado.
        
        Args:
            kraken_symbol: Par de negociação no formato da Kraken (ex: XXBTZUSD)
            
        Returns:
            str: Par de negociação no formato padronizado (ex: BTC/USD)
        """
        # Primeiro, verifica se o símbolo existe nos mercados
        for symbol, market in self._markets.items():
            if market['id'] == kraken_symbol:
                return symbol
                
        # Se não encontrar, tenta converter manualmente
        # Isso é complexo para a Kraken devido aos prefixos X e Z
        
        # Tenta identificar padrões conhecidos
        for base_asset, standard_base in self.ASSET_MAPPING.items():
            if kraken_symbol.startswith(base_asset):
                remaining = kraken_symbol[len(base_asset):]
                
                for quote_asset, standard_quote in self.ASSET_MAPPING.items():
                    if remaining == quote_asset:
                        return f"{standard_base}/{standard_quote}"
        
        # Se não conseguir converter, retorna o símbolo original
        logger.warning(f"Não foi possível converter o símbolo Kraken: {kraken_symbol}")
        return kraken_symbol
    
    async def _initialize_public_websocket(self) -> None:
        """
        Inicializa o WebSocket público para a Kraken.
        """
        if self._public_ws:
            return
            
        try:
            # Cria um cliente WebSocket
            self._public_ws = WebSocketClient(self._ws_public_url)
            self._public_ws.on_message(self._handle_public_message)
            self._public_ws.on_connect(self._handle_public_connect)
            self._public_ws.on_close(self._handle_public_close)
            
            # Conecta ao WebSocket
            await self._public_ws.connect()
            
            # A conexão inicial não requer autenticação
            
        except Exception as e:
            logger.error(f"Erro ao inicializar WebSocket público para Kraken: {str(e)}", exc_info=True)
            raise
    
    async def _initialize_private_websocket(self) -> None:
        """
        Inicializa o WebSocket privado para a Kraken.
        
        Requer API key e secret.
        """
        if not self._api_key or not self._api_secret:
            logger.warning("API key e secret são necessários para WebSocket privado")
            return
            
        if self._private_ws:
            return
            
        try:
            # Cria um cliente WebSocket
            self._private_ws = WebSocketClient(self._ws_private_url)
            self._private_ws.on_message(self._handle_private_message)
            self._private_ws.on_connect(self._handle_private_connect)
            self._private_ws.on_close(self._handle_private_close)
            
            # Conecta ao WebSocket
            await self._private_ws.connect()
            
            # Autentica após a conexão
            await self._authenticate_websocket()
            
        except Exception as e:
            logger.error(f"Erro ao inicializar WebSocket privado para Kraken: {str(e)}", exc_info=True)
            raise
    
    async def _authenticate_websocket(self) -> None:
        """
        Autentica o WebSocket privado.
        """
        if not self._private_ws or not self._api_key or not self._api_secret:
            return
            
        try:
            # Obtém um token do endpoint REST
            nonce = str(int(time.time() * 1000))
            
            # Gera a assinatura
            api_path = '/0/private/GetWebSocketsToken'
            api_nonce = nonce
            
            # Concatena os dados para assinatura
            sign_data = api_nonce + api_path
            
            # Calcula a assinatura
            signature = hmac.new(
                base64.b64decode(self._api_secret),
                sign_data.encode('utf-8'),
                hashlib.sha256
            ).digest()
            
            # Converte para base64
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            
            # Faz a requisição REST para obter o token
            headers = {
                'API-Key': self._api_key,
                'API-Sign': signature_b64
            }
            
            params = {
                'nonce': api_nonce
            }
            
            response = await self._exchange.fetch_json(
                'https://api.kraken.com' + api_path,
                params=params,
                headers=headers,
                method='POST'
            )
            
            # Extrai o token
            token = response.get('result', {}).get('token')
            
            if not token:
                raise ValueError("Não foi possível obter o token de autenticação")
                
            # Envia o token para o WebSocket
            auth_message = {
                "name": "authorize",
                "token": token
            }
            
            await self._private_ws.send(json.dumps(auth_message))
            logger.info("Token de autenticação enviado para o WebSocket privado")
            
        except Exception as e:
            logger.error(f"Erro ao autenticar WebSocket privado: {str(e)}", exc_info=True)
            raise
    
    async def _handle_public_connect(self) -> None:
        """
        Callback chamado quando o WebSocket público se conecta.
        """
        logger.info("WebSocket público da Kraken conectado")
        self._ws_connected = True
    
    async def _handle_public_close(self, code: int, reason: str) -> None:
        """
        Callback chamado quando o WebSocket público se desconecta.
        """
        logger.info(f"WebSocket público da Kraken desconectado: {code}, {reason}")
        self._ws_connected = False
        
        # Tenta reconectar após um tempo
        if self._initialized:
            await asyncio.sleep(5)
            try:
                logger.info("Tentando reconectar o WebSocket público")
                await self._initialize_public_websocket()
            except Exception as e:
                logger.error(f"Erro ao reconectar WebSocket público: {str(e)}", exc_info=True)
    
    async def _handle_private_connect(self) -> None:
        """
        Callback chamado quando o WebSocket privado se conecta.
        """
        logger.info("WebSocket privado da Kraken conectado")
    
    async def _handle_private_close(self, code: int, reason: str) -> None:
        """
        Callback chamado quando o WebSocket privado se desconecta.
        """
        logger.info(f"WebSocket privado da Kraken desconectado: {code}, {reason}")
        
        # Tenta reconectar após um tempo
        if self._initialized and self._api_key and self._api_secret:
            await asyncio.sleep(5)
            try:
                logger.info("Tentando reconectar o WebSocket privado")
                await self._initialize_private_websocket()
            except Exception as e:
                logger.error(f"Erro ao reconectar WebSocket privado: {str(e)}", exc_info=True)
    
    async def _handle_public_message(self, message: str) -> None:
        """
        Processa mensagens recebidas via WebSocket público.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = json.loads(message)
            
            # Verifica se é uma mensagem de sistema ou de evento
            if isinstance(data, dict):
                # Mensagem de sistema (resposta de subscribe/unsubscribe, erro, etc.)
                if 'event' in data:
                    event = data.get('event')
                    
                    if event == 'subscriptionStatus':
                        status = data.get('status')
                        subscription = data.get('subscription', {})
                        pair = data.get('pair')
                        
                        if status == 'subscribed':
                            logger.info(f"Subscrito com sucesso em {subscription.get('name')} para {pair}")
                        elif status == 'unsubscribed':
                            logger.info(f"Cancelada subscrição em {subscription.get('name')} para {pair}")
                        elif status == 'error':
                            logger.error(f"Erro na subscrição: {data.get('errorMessage')}")
                    
                    elif event == 'systemStatus':
                        status = data.get('status')
                        logger.info(f"Status do sistema Kraken: {status}")
                        
            elif isinstance(data, list) and len(data) >= 2:
                # Mensagem de dados (orderbook, trade, candle, etc.)
                channel_data = data[1]  # Dados do canal
                channel_name = data[2]  # Nome do canal
                pair = data[3]          # Par de negociação
                
                # Converte o par para o formato padronizado
                std_pair = self._parse_kraken_symbol(pair)
                
                # Processa os diferentes tipos de dados
                if channel_name == 'book':
                    await self._process_orderbook_update(channel_data, std_pair)
                elif channel_name == 'trade':
                    await self._process_trades_update(channel_data, std_pair)
                elif channel_name == 'ohlc':
                    await self._process_candle_update(channel_data, std_pair, data[0])  # data[0] contém o timeframe
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem pública: {str(e)}", exc_info=True)
    
    async def _handle_private_message(self, message: str) -> None:
        """
        Processa mensagens recebidas via WebSocket privado.
        
        Args:
            message: Mensagem WebSocket em formato JSON
        """
        try:
            data = json.loads(message)
            
            # Verifica se é uma mensagem de sistema ou de evento
            if isinstance(data, dict):
                # Mensagem de sistema ou resposta de autenticação
                if 'event' in data:
                    event = data.get('event')
                    
                    if event == 'subscriptionStatus':
                        status = data.get('status')
                        subscription = data.get('subscription', {})
                        
                        if status == 'subscribed':
                            logger.info(f"Subscrito com sucesso em {subscription.get('name')} (privado)")
                        elif status == 'unsubscribed':
                            logger.info(f"Cancelada subscrição em {subscription.get('name')} (privado)")
                        elif status == 'error':
                            logger.error(f"Erro na subscrição privada: {data.get('errorMessage')}")
                    
                    elif event == 'systemStatus':
                        status = data.get('status')
                        logger.info(f"Status do sistema Kraken (privado): {status}")
                        
                    elif event == 'authenticationStatus':
                        status = data.get('status')
                        
                        if status == 'success':
                            logger.info("Autenticação no WebSocket privado bem-sucedida")
                        else:
                            logger.error(f"Falha na autenticação do WebSocket privado: {data.get('errorMessage')}")
                
            elif isinstance(data, list) and len(data) >= 2:
                # Mensagem de dados privados
                # Não implementado neste adaptador, mas poderia processar ordens, posições, etc.
                pass
                
        except Exception as e:
            logger.error(f"Erro ao processar mensagem privada: {str(e)}", exc_info=True)
    
    async def _subscribe_public(self, subscription: Dict[str, Any], pairs: List[str]) -> None:
        """
        Envia uma mensagem de subscrição para o WebSocket público.
        
        Args:
            subscription: Detalhes da subscrição
            pairs: Lista de pares de negociação (no formato da Kraken)
        """
        if not self._public_ws or not self._ws_connected:
            await self._initialize_public_websocket()
            
        try:
            # Constrói a mensagem de subscrição
            message = {
                "name": "subscribe",
                "reqid": int(time.time() * 1000),
                "subscription": subscription,
                "pair": pairs
            }
            
            # Envia a mensagem
            await self._public_ws.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Erro ao enviar subscrição pública: {str(e)}", exc_info=True)
            raise
    
    async def _unsubscribe_public(self, subscription: Dict[str, Any], pairs: List[str]) -> None:
        """
        Envia uma mensagem de cancelamento de subscrição para o WebSocket público.
        
        Args:
            subscription: Detalhes da subscrição
            pairs: Lista de pares de negociação (no formato da Kraken)
        """
        if not self._public_ws or not self._ws_connected:
            return
            
        try:
            # Constrói a mensagem de cancelamento
            message = {
                "name": "unsubscribe",
                "reqid": int(time.time() * 1000),
                "subscription": subscription,
                "pair": pairs
            }
            
            # Envia a mensagem
            await self._public_ws.send(json.dumps(message))
            
        except Exception as e:
            logger.error(f"Erro ao enviar cancelamento de subscrição pública: {str(e)}", exc_info=True)
    
    async def _initialize_orderbook(self, trading_pair: str) -> None:
        """
        Inicializa o orderbook local com snapshot da Kraken.
        
        Args:
            trading_pair: Par de negociação no formato padronizado
        """
        try:
            # Obtém uma snapshot do orderbook via REST API
            orderbook_data = await self._exchange.fetch_order_book(trading_pair, limit=100)
            
            # Inicializa o orderbook local
            self._local_orderbooks[trading_pair] = {
                "bids": {Decimal(str(price)): Decimal(str(amount)) for price, amount in orderbook_data['bids']},
                "asks": {Decimal(str(price)): Decimal(str(amount)) for price, amount in orderbook_data['asks']},
                "timestamp": datetime.fromtimestamp(orderbook_data['timestamp'] / 1000) if orderbook_data['timestamp'] else datetime.utcnow()
            }
            
            logger.debug(f"Orderbook inicializado para {trading_pair} com {len(orderbook_data['bids'])} bids e {len(orderbook_data['asks'])} asks")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar orderbook para {trading_pair}: {str(e)}", exc_info=True)
            raise
    
    async def _process_orderbook_update(self, data: Dict[str, Any], trading_pair: str) -> None:
        """
        Processa uma atualização de orderbook recebida via WebSocket.
        
        Args:
            data: Dados da atualização
            trading_pair: Par de negociação no formato padronizado
        """
        try:
            # Verifica se temos um orderbook local para este par
            if trading_pair not in self._local_orderbooks:
                await self._initialize_orderbook(trading_pair)
            
            local_book = self._local_orderbooks[trading_pair]
            
            # Atualiza o timestamp
            local_book['timestamp'] = datetime.utcnow()
            
            # Processa as atualizações
            if 'bs' in data:  # Bids
                for bid in data['bs']:
                    price = Decimal(str(bid[0]))
                    amount = Decimal(str(bid[1]))
                    
                    if amount == 0:
                        local_book['bids'].pop(price, None)
                    else:
                        local_book['bids'][price] = amount
            
            if 'as' in data:  # Asks
                for ask in data['as']:
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
            
            # Cria a entidade OrderBook
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
            logger.error(f"Erro ao processar atualização de orderbook: {str(e)}", exc_info=True)
    
    async def _process_trades_update(self, data: List[List], trading_pair: str) -> None:
        """
        Processa uma atualização de trades recebida via WebSocket.
        
        Args:
            data: Dados dos trades
            trading_pair: Par de negociação no formato padronizado
        """
        try:
            # Processa cada trade na atualização
            for trade_data in data:
                # Formato da Kraken: [price, volume, time, side, orderType, misc]
                price = Decimal(str(trade_data[0]))
                amount = Decimal(str(trade_data[1]))
                timestamp = datetime.fromtimestamp(float(trade_data[2]))
                side_str = trade_data[3]  # 'b' para buy, 's' para sell
                
                # Converte o lado para o formato padronizado
                side = TradeSide.BUY if side_str == 'b' else TradeSide.SELL
                
                # Calcula o custo
                cost = price * amount
                
                # Gera um ID baseado nos dados (Kraken não fornece IDs de trade no WebSocket)
                trade_id = f"{trading_pair}-{timestamp.timestamp()}-{price}-{amount}"
                
                # Cria a entidade Trade
                trade = Trade(
                    id=trade_id,
                    exchange=self.name,
                    trading_pair=trading_pair,
                    price=price,
                    amount=amount,
                    cost=cost,
                    timestamp=timestamp,
                    side=side,
                    taker=True,  # Kraken WebSocket só envia trades de takers
                    raw_data=trade_data
                )
                
                # Chama o callback registrado para este par
                if trading_pair in self._trade_callbacks:
                    callback = self._trade_callbacks[trading_pair]
                    asyncio.create_task(callback(trade))
            
        except Exception as e:
            logger.error(f"Erro ao processar atualização de trades: {str(e)}", exc_info=True)
    
    async def _process_candle_update(self, data: List, trading_pair: str, interval: int) -> None:
        """
        Processa uma atualização de candle recebida via WebSocket.
        
        Args:
            data: Dados da candle
            trading_pair: Par de negociação no formato padronizado
            interval: Intervalo da candle em minutos
        """
        try:
            # Converte o intervalo para o timeframe padronizado
            if str(interval) in self.TIMEFRAME_MAP:
                timeframe = self.TIMEFRAME_MAP[str(interval)]
            else:
                logger.warning(f"Intervalo de candle não reconhecido: {interval}")
                return
                
            # Formato da Kraken: [time, open, high, low, close, vwap, volume, count]
            timestamp = datetime.fromtimestamp(float(data[1]))
            open_price = Decimal(str(data[2]))
            high_price = Decimal(str(data[3]))
            low_price = Decimal(str(data[4]))
            close_price = Decimal(str(data[5]))
            volume = Decimal(str(data[7]))
            count = int(data[8])
            
            # Cria a entidade Candle
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
                trades=count,
                raw_data=data
            )
            
            # Chama o callback registrado para este par e timeframe
            if (trading_pair in self._candle_callbacks and 
                timeframe in self._candle_callbacks[trading_pair]):
                callback = self._candle_callbacks[trading_pair][timeframe]
                asyncio.create_task(callback(candle))
            
        except Exception as e:
            logger.error(f"Erro ao processar atualização de candle: {str(e)}", exc_info=True)
    
    def _parse_funding_rate(self, data: Dict[str, Any], trading_pair: str) -> FundingRate:
        """
        Converte dados brutos de taxa de financiamento para a entidade FundingRate.
        
        Args:
            data: Dados brutos da taxa de financiamento
            trading_pair: Par de negociação
            
        Returns:
            FundingRate: Entidade FundingRate
        """
        # Extrai os dados
        rate = Decimal(str(data.get('fundingRate', 0)))
        timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()) / 1000)
        
        # Próximo timestamp de financiamento (normalmente a cada 8 horas)
        next_timestamp = timestamp + timedelta(hours=8)
        
        return FundingRate(
            exchange=self.name,
            trading_pair=trading_pair,
            timestamp=timestamp,
            rate=rate,
            next_timestamp=next_timestamp,
            raw_data=data
        )
    
    async def _funding_rate_polling_loop(self, trading_pair: str) -> None:
        """
        Loop para polling de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação
        """
        while trading_pair in self._funding_rate_callbacks:
            try:
                # Obtém a taxa de financiamento atual
                funding_rates = await self.fetch_funding_rates(trading_pair)
                
                if funding_rates:
                    # Chama o callback registrado para este par
                    callback = self._funding_rate_callbacks[trading_pair]
                    await callback(funding_rates[0])
                
                # Aguarda um intervalo antes de verificar novamente (30 minutos)
                await asyncio.sleep(1800)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Erro no polling de taxa de financiamento para {trading_pair}: {str(e)}", exc_info=True)
                await asyncio.sleep(60)  # Aguarda um minuto antes de tentar novamente