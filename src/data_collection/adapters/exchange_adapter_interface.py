"""
Interface para adaptadores de exchanges de criptomoedas.

Este módulo define a interface comum para todos os adaptadores de exchanges,
seguindo o padrão de Adapter do Domain-Driven Design e o princípio de
inversão de dependência.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable

from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook
from src.data_collection.domain.entities.trade import Trade
from src.data_collection.domain.entities.funding_rate import FundingRate
from src.data_collection.domain.entities.liquidation import Liquidation


class ExchangeAdapterInterface(ABC):
    """
    Interface para adaptadores de exchanges de criptomoedas.
    
    Define as operações comuns que todos os adaptadores de exchanges
    devem implementar, independente da exchange específica.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Retorna o nome da exchange.
        
        Returns:
            str: Nome da exchange
        """
        pass
    
    @property
    @abstractmethod
    def supported_trading_pairs(self) -> Set[str]:
        """
        Retorna o conjunto de pares de negociação suportados pela exchange.
        
        Returns:
            Set[str]: Conjunto de pares de negociação suportados
        """
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Inicializa o adaptador de exchange.
        
        Este método deve ser chamado antes de qualquer outra operação
        para estabelecer conexões e carregar configurações necessárias.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Finaliza o adaptador de exchange.
        
        Este método deve ser chamado ao encerrar o uso do adaptador
        para liberar recursos e fechar conexões de forma adequada.
        """
        pass
    
    @abstractmethod
    async def fetch_trading_pairs(self) -> List[str]:
        """
        Obtém a lista de pares de negociação disponíveis na exchange.
        
        Returns:
            List[str]: Lista de pares de negociação no formato padronizado (ex: BTC/USDT)
        """
        pass
    
    @abstractmethod
    async def fetch_ticker(self, trading_pair: str) -> Dict[str, Any]:
        """
        Obtém informações de ticker para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            
        Returns:
            Dict[str, Any]: Informações de ticker no formato padronizado
        """
        pass
    
    @abstractmethod
    async def fetch_orderbook(self, trading_pair: str, depth: int = 20) -> OrderBook:
        """
        Obtém o livro de ofertas para um par de negociação.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            depth: Profundidade do orderbook a ser obtido
            
        Returns:
            OrderBook: Entidade OrderBook com os dados do livro de ofertas
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def subscribe_funding_rates(
        self,
        trading_pair: str,
        callback: Callable[[FundingRate], Awaitable[None]]
    ) -> None:
        """
        Subscreve para atualizações de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            callback: Função assíncrona a ser chamada com cada atualização
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def unsubscribe_orderbook(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de orderbook.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        pass
    
    @abstractmethod
    async def unsubscribe_trades(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de trades.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        pass
    
    @abstractmethod
    async def unsubscribe_candles(self, trading_pair: str, timeframe: TimeFrame) -> None:
        """
        Cancela a subscrição para atualizações de velas.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
            timeframe: Intervalo de tempo das velas
        """
        pass
    
    @abstractmethod
    async def unsubscribe_funding_rates(self, trading_pair: str) -> None:
        """
        Cancela a subscrição para atualizações de taxas de financiamento.
        
        Args:
            trading_pair: Par de negociação (ex: BTC/USDT)
        """
        pass
    
    @abstractmethod
    async def unsubscribe_liquidations(self, trading_pair: Optional[str] = None) -> None:
        """
        Cancela a subscrição para eventos de liquidação.
        
        Args:
            trading_pair: Par de negociação (opcional)
        """
        pass
    
    @abstractmethod
    def validate_trading_pair(self, trading_pair: str) -> bool:
        """
        Valida se um par de negociação é suportado pela exchange.
        
        Args:
            trading_pair: Par de negociação a ser validado
            
        Returns:
            bool: True se o par é suportado, False caso contrário
        """
        pass
    
    @abstractmethod
    def get_supported_timeframes(self) -> Dict[str, TimeFrame]:
        """
        Obtém os timeframes suportados pela exchange.
        
        Returns:
            Dict[str, TimeFrame]: Dicionário de timeframes suportados,
            mapeando o código da exchange para o TimeFrame padronizado
        """
        pass