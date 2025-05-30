"""
Cliente para comunicação com exchanges de criptomoedas.

Este módulo implementa o cliente para comunicação com exchanges
de criptomoedas, usando a biblioteca CCXT.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import ccxt.async_support as ccxt
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class ExchangeClient:
    """
    Cliente para comunicação com exchanges de criptomoedas.
    
    Esta classe encapsula a comunicação com exchanges de criptomoedas
    através da biblioteca CCXT, fornecendo uma interface unificada para
    operações como obtenção de dados de mercado e execução de ordens.
    """
    
    def __init__(self, config: Dict[str, Dict[str, str]]):
        """
        Inicializa o cliente de exchange.
        
        Args:
            config: Dicionário de configuração com as credenciais para cada exchange.
                   Formato: {'exchange_id': {'api_key': '...', 'secret': '...', 'password': '...'}}
        """
        self.exchanges = {}
        self.config = config
        self._market_cache = {}
        self._last_market_update = {}
        self._cache_ttl = 60  # Tempo de vida do cache em segundos
    
    async def initialize(self):
        """
        Inicializa as conexões com as exchanges configuradas.
        """
        for exchange_id, credentials in self.config.items():
            try:
                # Verificar se a exchange é suportada pela CCXT
                if not hasattr(ccxt, exchange_id):
                    logger.error(f"Exchange não suportada: {exchange_id}")
                    continue
                
                # Criar instância da exchange
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class({
                    'apiKey': credentials.get('api_key'),
                    'secret': credentials.get('secret'),
                    'password': credentials.get('password'),
                    'enableRateLimit': True,
                    'options': {
                        'adjustForTimeDifference': True,
                        'recvWindow': 60000,
                    }
                })
                
                # Carregar mercados
                await exchange.load_markets()
                
                self.exchanges[exchange_id] = exchange
                logger.info(f"Exchange {exchange_id} inicializada com sucesso")
                
            except Exception as e:
                logger.error(f"Erro ao inicializar exchange {exchange_id}: {str(e)}")
    
    async def close(self):
        """
        Fecha todas as conexões com as exchanges.
        """
        close_tasks = []
        for exchange_id, exchange in self.exchanges.items():
            close_tasks.append(exchange.close())
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
            logger.info("Todas as conexões com exchanges foram fechadas")
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get_market_data(
        self, exchange: str, trading_pair: str, cache: bool = True
    ) -> Dict[str, Any]:
        """
        Obtém dados de mercado para um par de trading em uma exchange.
        
        Args:
            exchange: ID da exchange.
            trading_pair: Par de trading.
            cache: Se True, usa dados em cache se disponíveis e não expirados.
            
        Returns:
            Dict: Dados de mercado.
            
        Raises:
            ValueError: Se a exchange não estiver configurada.
            ccxt.ExchangeError: Se houver um erro na comunicação com a exchange.
        """
        # Verificar se a exchange está configurada
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange não configurada: {exchange}")
        
        # Verificar cache
        cache_key = f"{exchange}:{trading_pair}"
        current_time = time.time()
        
        if (
            cache and 
            cache_key in self._market_cache and 
            current_time - self._last_market_update.get(cache_key, 0) < self._cache_ttl
        ):
            return self._market_cache[cache_key]
        
        # Obter dados da exchange
        try:
            ccxt_exchange = self.exchanges[exchange]
            
            # Padronizar formato do par de trading
            market_pair = self._format_trading_pair(trading_pair, exchange)
            
            # Obter ticker
            ticker = await ccxt_exchange.fetch_ticker(market_pair)
            
            # Obter livro de ordens
            order_book = await ccxt_exchange.fetch_order_book(market_pair, limit=20)
            
            # Calcular profundidade do livro
            bid_depth = sum(bid[1] for bid in order_book['bids']) if order_book['bids'] else 0
            ask_depth = sum(ask[1] for ask in order_book['asks']) if order_book['asks'] else 0
            
            # Obter taxas
            fees = await self._get_trading_fees(ccxt_exchange, market_pair)
            
            # Montar resultado
            result = {
                "trading_pair": trading_pair,
                "exchange": exchange,
                "last_price": ticker['last'],
                "bid": ticker['bid'],
                "ask": ticker['ask'],
                "volume_24h": ticker['baseVolume'],
                "quote_volume_24h": ticker['quoteVolume'],
                "high_24h": ticker['high'],
                "low_24h": ticker['low'],
                "timestamp": ticker['timestamp'],
                "spread": (ticker['ask'] - ticker['bid']) / ticker['bid'] if ticker['bid'] > 0 else 0,
                "bid_depth": bid_depth,
                "ask_depth": ask_depth,
                "book_depth": {
                    "bids": order_book['bids'],
                    "asks": order_book['asks'],
                },
                "maker_fee": fees.get('maker', 0),
                "taker_fee": fees.get('taker', 0),
            }
            
            # Adicionar volatilidade se disponível
            if 'info' in ticker and 'priceChangePercent' in ticker['info']:
                result['volatility_24h'] = float(ticker['info']['priceChangePercent']) / 100
            else:
                # Estimar volatilidade como (high - low) / low
                result['volatility_24h'] = (ticker['high'] - ticker['low']) / ticker['low'] if ticker['low'] > 0 else 0
            
            # Atualizar cache
            self._market_cache[cache_key] = result
            self._last_market_update[cache_key] = current_time
            
            return result
            
        except ccxt.BaseError as e:
            logger.error(f"Erro ao obter dados de mercado para {trading_pair} em {exchange}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def place_order(
        self,
        exchange: str,
        trading_pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Coloca uma ordem em uma exchange.
        
        Args:
            exchange: ID da exchange.
            trading_pair: Par de trading.
            side: Lado da ordem ('buy' ou 'sell').
            order_type: Tipo da ordem ('market', 'limit', etc.).
            quantity: Quantidade a ser comprada/vendida.
            price: Preço para ordens limit (opcional).
            
        Returns:
            Dict: Resultado da execução da ordem.
            
        Raises:
            ValueError: Se a exchange não estiver configurada ou parâmetros inválidos.
            ccxt.ExchangeError: Se houver um erro na comunicação com a exchange.
        """
        # Verificar se a exchange está configurada
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange não configurada: {exchange}")
        
        # Validar parâmetros
        if side not in ['buy', 'sell']:
            raise ValueError(f"Side inválido: {side}. Deve ser 'buy' ou 'sell'")
        
        if order_type not in ['market', 'limit']:
            raise ValueError(f"Tipo de ordem inválido: {order_type}. Deve ser 'market' ou 'limit'")
        
        if order_type == 'limit' and price is None:
            raise ValueError("Preço é obrigatório para ordens limit")
        
        # Preparar ordem
        try:
            ccxt_exchange = self.exchanges[exchange]
            
            # Padronizar formato do par de trading
            market_pair = self._format_trading_pair(trading_pair, exchange)
            
            # Obter informações do mercado para validação e ajustes
            market_info = ccxt_exchange.markets[market_pair]
            
            # Ajustar quantidade e preço de acordo com os limites da exchange
            adjusted_quantity = self._adjust_quantity(quantity, market_info)
            adjusted_price = None
            if price is not None:
                adjusted_price = self._adjust_price(price, market_info)
            
            logger.info(
                f"Colocando ordem em {exchange}: {side} {order_type} {adjusted_quantity} {trading_pair} "
                f"{'@ ' + str(adjusted_price) if adjusted_price else ''}"
            )
            
            # Enviar ordem para a exchange
            ccxt_order_type = order_type
            
            # Algumas exchanges usam nomenclatura diferente
            if ccxt_exchange.id in ['bittrex', 'kucoin']:
                if order_type == 'market':
                    ccxt_order_type = 'market'
            
            order_params = {}
            
            # Parâmetros especiais para algumas exchanges
            if ccxt_exchange.id == 'binance':
                order_params['newClientOrderId'] = f"bot_{int(time.time() * 1000)}"
            
            # Colocar a ordem
            order_result = await ccxt_exchange.create_order(
                symbol=market_pair,
                type=ccxt_order_type,
                side=side,
                amount=adjusted_quantity,
                price=adjusted_price,
                params=order_params
            )
            
            # Obter detalhes da ordem
            try:
                order_details = await ccxt_exchange.fetch_order(order_result['id'], market_pair)
            except Exception as e:
                logger.warning(f"Não foi possível obter detalhes da ordem: {str(e)}")
                order_details = order_result
            
            # Processar resultado
            filled = order_details.get('filled', 0)
            status = order_details.get('status', 'open')
            
            # Padronizar status
            if status in ['closed', 'filled']:
                status = 'filled'
            elif status in ['open', 'open', 'partially_filled']:
                status = 'partial' if filled > 0 else 'pending'
            elif status in ['canceled', 'cancelled', 'expired', 'rejected']:
                status = 'cancelled'
            
            # Calcular preço médio
            cost = order_details.get('cost', 0)
            average_price = cost / filled if filled > 0 else None
            
            # Calcular taxas
            fees = 0
            if 'fee' in order_details and order_details['fee'] is not None:
                if 'cost' in order_details['fee']:
                    fees = order_details['fee']['cost']
            
            result = {
                "order_id": order_details['id'],
                "status": status,
                "filled_quantity": filled,
                "average_price": average_price,
                "fees": fees,
                "timestamp": order_details.get('timestamp', int(time.time() * 1000)),
                "raw_response": order_details,
            }
            
            return result
            
        except ccxt.InsufficientFunds as e:
            logger.error(f"Fundos insuficientes na exchange {exchange}: {str(e)}")
            raise
        except ccxt.InvalidOrder as e:
            logger.error(f"Ordem inválida para {trading_pair} em {exchange}: {str(e)}")
            raise
        except ccxt.BaseError as e:
            logger.error(f"Erro ao colocar ordem para {trading_pair} em {exchange}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def get_order_status(
        self, exchange: str, order_id: str, trading_pair: str
    ) -> Dict[str, Any]:
        """
        Obtém o status de uma ordem em uma exchange.
        
        Args:
            exchange: ID da exchange.
            order_id: ID da ordem na exchange.
            trading_pair: Par de trading.
            
        Returns:
            Dict: Status da ordem.
            
        Raises:
            ValueError: Se a exchange não estiver configurada.
            ccxt.ExchangeError: Se houver um erro na comunicação com a exchange.
        """
        # Verificar se a exchange está configurada
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange não configurada: {exchange}")
        
        try:
            ccxt_exchange = self.exchanges[exchange]
            
            # Padronizar formato do par de trading
            market_pair = self._format_trading_pair(trading_pair, exchange)
            
            # Obter detalhes da ordem
            order_details = await ccxt_exchange.fetch_order(order_id, market_pair)
            
            # Processar resultado
            filled = order_details.get('filled', 0)
            status = order_details.get('status', 'open')
            
            # Padronizar status
            if status in ['closed', 'filled']:
                status = 'filled'
            elif status in ['open', 'open', 'partially_filled']:
                status = 'partial' if filled > 0 else 'pending'
            elif status in ['canceled', 'cancelled', 'expired', 'rejected']:
                status = 'cancelled'
            
            # Calcular preço médio
            cost = order_details.get('cost', 0)
            average_price = cost / filled if filled > 0 else None
            
            # Calcular taxas
            fees = 0
            if 'fee' in order_details and order_details['fee'] is not None:
                if 'cost' in order_details['fee']:
                    fees = order_details['fee']['cost']
            
            result = {
                "order_id": order_details['id'],
                "status": status,
                "filled_quantity": filled,
                "average_price": average_price,
                "fees": fees,
                "timestamp": order_details.get('timestamp', int(time.time() * 1000)),
                "raw_response": order_details,
            }
            
            return result
            
        except ccxt.OrderNotFound as e:
            logger.error(f"Ordem {order_id} não encontrada na exchange {exchange}: {str(e)}")
            raise
        except ccxt.BaseError as e:
            logger.error(f"Erro ao obter status da ordem {order_id} na exchange {exchange}: {str(e)}")
            raise
    
    @retry(
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeNotAvailable)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def cancel_order(
        self, exchange: str, order_id: str, trading_pair: str
    ) -> Dict[str, Any]:
        """
        Cancela uma ordem em uma exchange.
        
        Args:
            exchange: ID da exchange.
            order_id: ID da ordem na exchange.
            trading_pair: Par de trading.
            
        Returns:
            Dict: Resultado do cancelamento.
            
        Raises:
            ValueError: Se a exchange não estiver configurada.
            ccxt.ExchangeError: Se houver um erro na comunicação com a exchange.
        """
        # Verificar se a exchange está configurada
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange não configurada: {exchange}")
        
        try:
            ccxt_exchange = self.exchanges[exchange]
            
            # Padronizar formato do par de trading
            market_pair = self._format_trading_pair(trading_pair, exchange)
            
            # Cancelar a ordem
            cancel_result = await ccxt_exchange.cancel_order(order_id, market_pair)
            
            # Obter detalhes da ordem após cancelamento
            try:
                order_details = await ccxt_exchange.fetch_order(order_id, market_pair)
            except Exception as e:
                logger.warning(f"Não foi possível obter detalhes da ordem após cancelamento: {str(e)}")
                order_details = cancel_result
            
            # Processar resultado
            filled = order_details.get('filled', 0)
            
            result = {
                "order_id": order_id,
                "status": "cancelled",
                "filled_quantity": filled,
                "timestamp": int(time.time() * 1000),
                "raw_response": order_details,
            }
            
            return result
            
        except ccxt.OrderNotFound as e:
            logger.error(f"Ordem {order_id} não encontrada na exchange {exchange}: {str(e)}")
            raise
        except ccxt.BaseError as e:
            logger.error(f"Erro ao cancelar ordem {order_id} na exchange {exchange}: {str(e)}")
            raise
    
    async def get_balance(self, exchange: str, currency: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém o saldo de uma ou todas as moedas em uma exchange.
        
        Args:
            exchange: ID da exchange.
            currency: Código da moeda (opcional, se não fornecido retorna todas).
            
        Returns:
            Dict: Saldo da moeda ou de todas as moedas.
            
        Raises:
            ValueError: Se a exchange não estiver configurada.
            ccxt.ExchangeError: Se houver um erro na comunicação com a exchange.
        """
        # Verificar se a exchange está configurada
        if exchange not in self.exchanges:
            raise ValueError(f"Exchange não configurada: {exchange}")
        
        try:
            ccxt_exchange = self.exchanges[exchange]
            
            # Obter saldo
            balance = await ccxt_exchange.fetch_balance()
            
            if currency:
                if currency in balance:
                    return {
                        "currency": currency,
                        "free": balance[currency]['free'],
                        "used": balance[currency]['used'],
                        "total": balance[currency]['total'],
                    }
                else:
                    return {
                        "currency": currency,
                        "free": 0,
                        "used": 0,
                        "total": 0,
                    }
            else:
                result = {}
                for curr, amount in balance.items():
                    if isinstance(amount, dict) and 'free' in amount:
                        result[curr] = {
                            "free": amount['free'],
                            "used": amount['used'],
                            "total": amount['total'],
                        }
                return result
            
        except ccxt.BaseError as e:
            logger.error(f"Erro ao obter saldo na exchange {exchange}: {str(e)}")
            raise
    
    def _format_trading_pair(self, trading_pair: str, exchange: str) -> str:
        """
        Formata um par de trading para o formato aceito pela exchange.
        
        Args:
            trading_pair: Par de trading no formato padrão (ex: BTC/USDT).
            exchange: ID da exchange.
            
        Returns:
            str: Par de trading no formato aceito pela exchange.
        """
        if exchange not in self.exchanges:
            return trading_pair
        
        ccxt_exchange = self.exchanges[exchange]
        
        # Se a exchange aceita o formato padrão, retorná-lo
        if trading_pair in ccxt_exchange.markets:
            return trading_pair
        
        # Tentar converter para o formato da exchange
        # Algumas exchanges usam formatos diferentes, como BTCUSDT em vez de BTC/USDT
        if '/' in trading_pair:
            base, quote = trading_pair.split('/')
            alternative_format = f"{base}{quote}"
            if alternative_format in ccxt_exchange.markets:
                return alternative_format
        
        # Se não encontrar, retornar o formato original
        return trading_pair
    
    async def _get_trading_fees(self, ccxt_exchange, trading_pair: str) -> Dict[str, float]:
        """
        Obtém as taxas de trading para um par.
        
        Args:
            ccxt_exchange: Instância da exchange CCXT.
            trading_pair: Par de trading.
            
        Returns:
            Dict[str, float]: Dicionário com taxas maker e taker.
        """
        try:
            # Tentar obter taxas específicas do par
            fees = await ccxt_exchange.fetch_trading_fee(trading_pair)
            return {
                "maker": fees.get('maker', 0),
                "taker": fees.get('taker', 0),
            }
        except Exception as e:
            # Em caso de erro, usar taxas padrão da exchange
            try:
                markets = ccxt_exchange.markets or {}
                market = markets.get(trading_pair, {})
                return {
                    "maker": market.get('maker', ccxt_exchange.fees.get('trading', {}).get('maker', 0.001)),
                    "taker": market.get('taker', ccxt_exchange.fees.get('trading', {}).get('taker', 0.001)),
                }
            except Exception:
                # Valores padrão caso não seja possível obter as taxas
                return {
                    "maker": 0.001,  # 0.1%
                    "taker": 0.001,  # 0.1%
                }
    
    def _adjust_quantity(self, quantity: float, market_info: Dict) -> float:
        """
        Ajusta a quantidade de acordo com os limites da exchange.
        
        Args:
            quantity: Quantidade original.
            market_info: Informações do mercado.
            
        Returns:
            float: Quantidade ajustada.
        """
        # Obter limites
        limits = market_info.get('limits', {})
        amount_limits = limits.get('amount', {})
        min_amount = amount_limits.get('min', 0)
        max_amount = amount_limits.get('max', float('inf'))
        
        # Obter precisão
        precision = market_info.get('precision', {})
        amount_precision = precision.get('amount', 8)
        
        # Ajustar para os limites
        adjusted = max(min_amount, min(quantity, max_amount))
        
        # Ajustar para a precisão
        factor = 10 ** amount_precision
        adjusted = int(adjusted * factor) / factor
        
        return adjusted
    
    def _adjust_price(self, price: float, market_info: Dict) -> float:
        """
        Ajusta o preço de acordo com os limites da exchange.
        
        Args:
            price: Preço original.
            market_info: Informações do mercado.
            
        Returns:
            float: Preço ajustado.
        """
        # Obter limites
        limits = market_info.get('limits', {})
        price_limits = limits.get('price', {})
        min_price = price_limits.get('min', 0)
        max_price = price_limits.get('max', float('inf'))
        
        # Obter precisão
        precision = market_info.get('precision', {})
        price_precision = precision.get('price', 8)
        
        # Ajustar para os limites
        adjusted = max(min_price, min(price, max_price))
        
        # Ajustar para a precisão
        factor = 10 ** price_precision
        adjusted = int(adjusted * factor) / factor
        
        return adjusted