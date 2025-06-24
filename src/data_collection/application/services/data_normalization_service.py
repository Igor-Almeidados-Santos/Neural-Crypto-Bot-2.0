"""
Serviço de normalização de dados.

Este módulo fornece funções para normalizar dados obtidos de diferentes
exchanges para um formato padronizado, garantindo consistência nos dados
utilizados pelo sistema.
"""
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple

from data_collection.domain.entities.candle import Candle, TimeFrame
from data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from data_collection.domain.entities.trade import Trade, TradeSide


logger = logging.getLogger(__name__)


class DataNormalizationService:
    """
    Serviço responsável por normalizar dados de mercado.
    
    Implementa funções para converter dados brutos de diferentes exchanges
    para o formato padronizado das entidades do domínio.
    """
    
    # Mapeamento de nomes de exchanges para formatos padronizados
    EXCHANGE_NAME_MAP = {
        "binance": "binance",
        "binanceus": "binance_us",
        "coinbasepro": "coinbase",
        "coinbase": "coinbase",
        "kraken": "kraken",
        "kucoin": "kucoin",
        "ftx": "ftx",
        "bitmex": "bitmex",
        "bybit": "bybit",
        "huobi": "huobi",
        "okx": "okex",
        "gateio": "gate_io",
        "bitfinex": "bitfinex",
        "bitstamp": "bitstamp"
    }
    
    @staticmethod
    def normalize_exchange_name(exchange_name: str) -> str:
        """
        Normaliza o nome da exchange para um formato padronizado.
        
        Args:
            exchange_name: Nome da exchange a ser normalizado
            
        Returns:
            str: Nome da exchange no formato padronizado
        """
        exchange_name = exchange_name.lower()
        return DataNormalizationService.EXCHANGE_NAME_MAP.get(exchange_name, exchange_name)
    
    @staticmethod
    def normalize_trading_pair(trading_pair: str, exchange: str) -> str:
        """
        Normaliza um par de negociação para o formato padronizado.
        
        Converte formatos específicos de exchanges (como 'BTCUSDT') para
        o formato padronizado (como 'BTC/USDT').
        
        Args:
            trading_pair: Par de negociação a ser normalizado
            exchange: Nome da exchange de origem
            
        Returns:
            str: Par de negociação no formato padronizado
        """
        # Verifica se já está no formato padronizado
        if '/' in trading_pair:
            return trading_pair
            
        exchange = exchange.lower()
        
        # Binance e similares: 'BTCUSDT' -> 'BTC/USDT'
        if exchange in ['binance', 'binanceus', 'kucoin', 'huobi', 'gateio', 'okx']:
            # Identificação de stablecoins e outras moedas fiat comuns
            fiat_tokens = ['USDT', 'BUSD', 'USDC', 'DAI', 'USD', 'EUR', 'GBP', 'JPY', 'AUD', 
                          'CAD', 'CHF', 'CNY', 'HKD', 'NZD', 'SGD', 'TRY', 'ZAR', 'RUB', 'UAH']
            
            for token in sorted(fiat_tokens, key=len, reverse=True):
                if trading_pair.endswith(token):
                    base = trading_pair[:-len(token)]
                    quote = token
                    return f"{base}/{quote}"
            
            # Se não for identificado um padrão claro, tenta dividir no meio
            if len(trading_pair) > 3:
                for i in range(3, len(trading_pair) - 2):
                    base = trading_pair[:i]
                    quote = trading_pair[i:]
                    if base in ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOT']:
                        return f"{base}/{quote}"
            
            # Caso não seja possível identificar um padrão claro
            logger.warning(f"Não foi possível normalizar o par {trading_pair} da exchange {exchange}")
            return trading_pair
            
        # Coinbase: 'BTC-USD' -> 'BTC/USD'
        elif exchange in ['coinbase', 'coinbasepro']:
            return trading_pair.replace('-', '/')
            
        # Kraken: 'XXBTZUSD' -> 'BTC/USD'
        elif exchange == 'kraken':
            kraken_assets = {
                'XXBT': 'BTC', 'XETH': 'ETH', 'XXRP': 'XRP', 'XXLM': 'XLM',
                'XLTC': 'LTC', 'XETC': 'ETC', 'XZEC': 'ZEC', 'XXMR': 'XMR',
                'DASH': 'DASH', 'ZUSD': 'USD', 'ZEUR': 'EUR', 'ZJPY': 'JPY',
                'ZGBP': 'GBP', 'ZCAD': 'CAD', 'ZAUD': 'AUD'
            }
            
            # Tenta identificar padrões conhecidos do Kraken
            if trading_pair.startswith('X') and 'Z' in trading_pair:
                for base, quote in kraken_assets.items():
                    if trading_pair.startswith(base):
                        remaining = trading_pair[len(base):]
                        for base2, quote2 in kraken_assets.items():
                            if remaining == base2:
                                return f"{quote}/{quote2}"
            
            # Se não for possível identificar um padrão claro
            logger.warning(f"Não foi possível normalizar o par {trading_pair} da exchange {exchange}")
            return trading_pair
            
        # FTX: 'BTC/USD' -> já no formato padronizado
        # Bitfinex: 'tBTCUSD' -> 'BTC/USD'
        elif exchange == 'bitfinex':
            if trading_pair.startswith('t'):
                pair = trading_pair[1:]
                if len(pair) >= 6:
                    base = pair[:3]
                    quote = pair[3:]
                    return f"{base}/{quote}"
            
            # Se não for possível identificar um padrão claro
            logger.warning(f"Não foi possível normalizar o par {trading_pair} da exchange {exchange}")
            return trading_pair
            
        # Bitstamp: 'btcusd' -> 'BTC/USD'
        elif exchange == 'bitstamp':
            if len(trading_pair) >= 6:
                base = trading_pair[:3].upper()
                quote = trading_pair[3:].upper()
                return f"{base}/{quote}"
            
            # Se não for possível identificar um padrão claro
            logger.warning(f"Não foi possível normalizar o par {trading_pair} da exchange {exchange}")
            return trading_pair
            
        # Para outras exchanges, retorna o par original
        return trading_pair
    
    @staticmethod
    def normalize_timestamp(timestamp: Union[int, float, str, datetime]) -> datetime:
        """
        Normaliza um timestamp para o formato datetime UTC.
        
        Args:
            timestamp: Timestamp a ser normalizado, pode ser um número (ms ou s),
                      uma string ISO ou um objeto datetime
            
        Returns:
            datetime: Timestamp normalizado como datetime UTC
        """
        if isinstance(timestamp, datetime):
            # Assegura que o datetime está em UTC
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            return timestamp.astimezone(timezone.utc)
            
        elif isinstance(timestamp, (int, float)):
            # Determina se o timestamp está em segundos ou milissegundos
            if timestamp > 2**32:  # Provavelmente em milissegundos
                timestamp = timestamp / 1000
                
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            
        elif isinstance(timestamp, str):
            # Tenta converter a string para datetime
            try:
                # Para strings ISO 8601
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.astimezone(timezone.utc)
            except ValueError:
                # Tenta interpretar como timestamp em segundos
                try:
                    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
                except ValueError:
                    raise ValueError(f"Formato de timestamp não reconhecido: {timestamp}")
        
        raise TypeError(f"Tipo de timestamp não suportado: {type(timestamp)}")
    
    @staticmethod
    def normalize_decimal(value: Union[int, float, str, Decimal]) -> Decimal:
        """
        Normaliza um valor para o formato Decimal.
        
        Args:
            value: Valor a ser normalizado
            
        Returns:
            Decimal: Valor normalizado como Decimal
        """
        if isinstance(value, Decimal):
            return value
            
        if isinstance(value, (int, float)):
            return Decimal(str(value))
            
        if isinstance(value, str):
            return Decimal(value)
            
        raise TypeError(f"Tipo de valor não suportado: {type(value)}")
    
    @staticmethod
    def normalize_trade_side(side: str) -> TradeSide:
        """
        Normaliza o lado de uma transação para o formato padronizado.
        
        Args:
            side: Lado da transação (ex: 'buy', 'sell', 'bid', 'ask')
            
        Returns:
            TradeSide: Lado da transação no formato padronizado
        """
        side = side.lower()
        
        if side in ['buy', 'bid', 'long']:
            return TradeSide.BUY
        elif side in ['sell', 'ask', 'short']:
            return TradeSide.SELL
        else:
            return TradeSide.UNKNOWN
    
    @staticmethod
    def normalize_timeframe(timeframe: str, exchange: str) -> TimeFrame:
        """
        Normaliza um timeframe para o formato padronizado.
        
        Args:
            timeframe: Timeframe a ser normalizado (ex: '1m', '1h', '1d')
            exchange: Nome da exchange de origem
            
        Returns:
            TimeFrame: Timeframe no formato padronizado
        """
        exchange = exchange.lower()
        timeframe = timeframe.lower()
        
        # Mapeamento comum para a maioria das exchanges
        common_map = {
            '1m': TimeFrame.MINUTE_1,
            '3m': TimeFrame.MINUTE_3,
            '5m': TimeFrame.MINUTE_5,
            '15m': TimeFrame.MINUTE_15,
            '30m': TimeFrame.MINUTE_30,
            '1h': TimeFrame.HOUR_1,
            '2h': TimeFrame.HOUR_2,
            '4h': TimeFrame.HOUR_4,
            '6h': TimeFrame.HOUR_6,
            '8h': TimeFrame.HOUR_8,
            '12h': TimeFrame.HOUR_12,
            '1d': TimeFrame.DAY_1,
            '3d': TimeFrame.DAY_3,
            '1w': TimeFrame.WEEK_1,
            '1M': TimeFrame.MONTH_1
        }
        
        # Tenta o mapeamento comum
        if timeframe in common_map:
            return common_map[timeframe]
        
        # Mapeamentos específicos para exchanges
        if exchange == 'binance':
            return common_map.get(timeframe)
            
        elif exchange in ['coinbase', 'coinbasepro']:
            coinbase_map = {
                '60': TimeFrame.MINUTE_1,
                '300': TimeFrame.MINUTE_5,
                '900': TimeFrame.MINUTE_15,
                '3600': TimeFrame.HOUR_1,
                '21600': TimeFrame.HOUR_6,
                '86400': TimeFrame.DAY_1
            }
            return coinbase_map.get(timeframe)
            
        elif exchange == 'kraken':
            kraken_map = {
                '1': TimeFrame.MINUTE_1,
                '5': TimeFrame.MINUTE_5,
                '15': TimeFrame.MINUTE_15,
                '30': TimeFrame.MINUTE_30,
                '60': TimeFrame.HOUR_1,
                '240': TimeFrame.HOUR_4,
                '1440': TimeFrame.DAY_1,
                '10080': TimeFrame.WEEK_1,
                '21600': TimeFrame.MONTH_1
            }
            return kraken_map.get(timeframe)
        
        # Se não for possível mapear, retorna None
        logger.warning(f"Timeframe não reconhecido: {timeframe} da exchange {exchange}")
        return None
    
    @staticmethod
    def normalize_trade(
        raw_trade: Dict[str, Any], 
        exchange: str, 
        trading_pair: Optional[str] = None
    ) -> Trade:
        """
        Normaliza dados brutos de trade para a entidade Trade.
        
        Args:
            raw_trade: Dados brutos do trade
            exchange: Nome da exchange de origem
            trading_pair: Par de negociação (opcional, se não estiver nos dados brutos)
            
        Returns:
            Trade: Entidade Trade normalizada
        """
        exchange = DataNormalizationService.normalize_exchange_name(exchange)
        
        # Extrai e normaliza os campos comuns
        trade_id = str(raw_trade.get('id', raw_trade.get('trade_id', raw_trade.get('tid', ''))))
        
        # Extrai o par de negociação se não fornecido
        if trading_pair is None:
            trading_pair = raw_trade.get('symbol', raw_trade.get('market', raw_trade.get('pair', '')))
            trading_pair = DataNormalizationService.normalize_trading_pair(trading_pair, exchange)
        
        # Extrai e normaliza o timestamp
        timestamp = raw_trade.get('timestamp', raw_trade.get('time', raw_trade.get('date', datetime.utcnow())))
        timestamp = DataNormalizationService.normalize_timestamp(timestamp)
        
        # Extrai e normaliza o preço e quantidade
        price = DataNormalizationService.normalize_decimal(raw_trade.get('price', 0))
        amount = DataNormalizationService.normalize_decimal(raw_trade.get('amount', raw_trade.get('size', raw_trade.get('volume', 0))))
        
        # Calcula o custo se não fornecido
        cost = raw_trade.get('cost', raw_trade.get('total', price * amount))
        cost = DataNormalizationService.normalize_decimal(cost)
        
        # Extrai e normaliza o lado
        side_raw = raw_trade.get('side', raw_trade.get('type', 'unknown'))
        side = DataNormalizationService.normalize_trade_side(side_raw)
        
        # Determina se é taker ou maker
        taker = raw_trade.get('taker', raw_trade.get('liquidity', 'taker')) == 'taker'
        
        # Cria a entidade Trade
        return Trade(
            id=trade_id,
            exchange=exchange,
            trading_pair=trading_pair,
            price=price,
            amount=amount,
            cost=cost,
            timestamp=timestamp,
            side=side,
            taker=taker,
            raw_data=raw_trade
        )
    
    @staticmethod
    def normalize_orderbook(
        raw_orderbook: Dict[str, Any], 
        exchange: str, 
        trading_pair: Optional[str] = None
    ) -> OrderBook:
        """
        Normaliza dados brutos de orderbook para a entidade OrderBook.
        
        Args:
            raw_orderbook: Dados brutos do orderbook
            exchange: Nome da exchange de origem
            trading_pair: Par de negociação (opcional, se não estiver nos dados brutos)
            
        Returns:
            OrderBook: Entidade OrderBook normalizada
        """
        exchange = DataNormalizationService.normalize_exchange_name(exchange)
        
        # Extrai o par de negociação se não fornecido
        if trading_pair is None:
            trading_pair = raw_orderbook.get('symbol', raw_orderbook.get('market', raw_orderbook.get('pair', '')))
            trading_pair = DataNormalizationService.normalize_trading_pair(trading_pair, exchange)
        
        # Extrai e normaliza o timestamp
        timestamp = raw_orderbook.get('timestamp', raw_orderbook.get('time', raw_orderbook.get('date', datetime.utcnow())))
        timestamp = DataNormalizationService.normalize_timestamp(timestamp)
        
        # Extrai e normaliza as ofertas de compra (bids)
        bids_raw = raw_orderbook.get('bids', [])
        bids = []
        
        for bid_data in bids_raw:
            if isinstance(bid_data, list):
                # Formato [price, amount, count?]
                price = DataNormalizationService.normalize_decimal(bid_data[0])
                amount = DataNormalizationService.normalize_decimal(bid_data[1])
                count = int(bid_data[2]) if len(bid_data) > 2 else None
            elif isinstance(bid_data, dict):
                # Formato {price: float, amount: float, count?: int}
                price = DataNormalizationService.normalize_decimal(bid_data.get('price', 0))
                amount = DataNormalizationService.normalize_decimal(bid_data.get('amount', 0))
                count = bid_data.get('count')
            else:
                continue
                
            bids.append(OrderBookLevel(price=price, amount=amount, count=count))
        
        # Extrai e normaliza as ofertas de venda (asks)
        asks_raw = raw_orderbook.get('asks', [])
        asks = []
        
        for ask_data in asks_raw:
            if isinstance(ask_data, list):
                # Formato [price, amount, count?]
                price = DataNormalizationService.normalize_decimal(ask_data[0])
                amount = DataNormalizationService.normalize_decimal(ask_data[1])
                count = int(ask_data[2]) if len(ask_data) > 2 else None
            elif isinstance(ask_data, dict):
                # Formato {price: float, amount: float, count?: int}
                price = DataNormalizationService.normalize_decimal(ask_data.get('price', 0))
                amount = DataNormalizationService.normalize_decimal(ask_data.get('amount', 0))
                count = ask_data.get('count')
            else:
                continue
                
            asks.append(OrderBookLevel(price=price, amount=amount, count=count))
        
        # Ordena as ofertas corretamente
        bids.sort(key=lambda x: x.price, reverse=True)  # Preços maiores primeiro
        asks.sort(key=lambda x: x.price)  # Preços menores primeiro
        
        # Cria a entidade OrderBook
        return OrderBook(
            exchange=exchange,
            trading_pair=trading_pair,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            raw_data=raw_orderbook
        )
    
    @staticmethod
    def normalize_candle(
        raw_candle: Union[List, Dict[str, Any]], 
        exchange: str, 
        trading_pair: str,
        timeframe: Union[str, TimeFrame]
    ) -> Candle:
        """
        Normaliza dados brutos de candle para a entidade Candle.
        
        Args:
            raw_candle: Dados brutos da candle
            exchange: Nome da exchange de origem
            trading_pair: Par de negociação
            timeframe: Timeframe da candle
            
        Returns:
            Candle: Entidade Candle normalizada
        """
        exchange = DataNormalizationService.normalize_exchange_name(exchange)
        trading_pair = DataNormalizationService.normalize_trading_pair(trading_pair, exchange)
        
        # Normaliza o timeframe se for string
        if isinstance(timeframe, str):
            timeframe = DataNormalizationService.normalize_timeframe(timeframe, exchange)
        
        # Extrai os dados da candle
        if isinstance(raw_candle, list):
            # Formato OHLCV: [timestamp, open, high, low, close, volume, ...]
            timestamp = DataNormalizationService.normalize_timestamp(raw_candle[0])
            open_price = DataNormalizationService.normalize_decimal(raw_candle[1])
            high_price = DataNormalizationService.normalize_decimal(raw_candle[2])
            low_price = DataNormalizationService.normalize_decimal(raw_candle[3])
            close_price = DataNormalizationService.normalize_decimal(raw_candle[4])
            volume = DataNormalizationService.normalize_decimal(raw_candle[5])
            trades = int(raw_candle[6]) if len(raw_candle) > 6 else None
            
        elif isinstance(raw_candle, dict):
            # Formato com chaves
            timestamp = raw_candle.get('timestamp', raw_candle.get('time', raw_candle.get('date')))
            timestamp = DataNormalizationService.normalize_timestamp(timestamp)
            
            open_price = DataNormalizationService.normalize_decimal(raw_candle.get('open', raw_candle.get('o')))
            high_price = DataNormalizationService.normalize_decimal(raw_candle.get('high', raw_candle.get('h')))
            low_price = DataNormalizationService.normalize_decimal(raw_candle.get('low', raw_candle.get('l')))
            close_price = DataNormalizationService.normalize_decimal(raw_candle.get('close', raw_candle.get('c')))
            volume = DataNormalizationService.normalize_decimal(raw_candle.get('volume', raw_candle.get('v')))
            trades = raw_candle.get('trades', raw_candle.get('n'))
            
        else:
            raise ValueError(f"Formato de candle não suportado: {type(raw_candle)}")
        
        # Cria a entidade Candle
        return Candle(
            exchange=exchange,
            trading_pair=trading_pair,
            timestamp=timestamp,
            timeframe=timeframe,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            trades=trades,
            raw_data=raw_candle
        )