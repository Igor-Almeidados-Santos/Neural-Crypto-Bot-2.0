"""
Repositório para persistência e recuperação de livros de ofertas (orderbooks).

Este módulo define a interface do repositório para orderbooks, seguindo
o padrão de Repository do Domain-Driven Design.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple

from src.data_collection.domain.entities.orderbook import OrderBook


class OrderBookRepository(ABC):
    """
    Interface para repositório de livros de ofertas (orderbooks).
    
    Define operações para persistência e recuperação de orderbooks,
    independente da tecnologia de armazenamento utilizada.
    """
    
    @abstractmethod
    async def save(self, orderbook: OrderBook) -> None:
        """
        Salva um orderbook no repositório.
        
        Args:
            orderbook: OrderBook a ser salvo
        """
        pass
    
    @abstractmethod
    async def save_batch(self, orderbooks: List[OrderBook]) -> None:
        """
        Salva um lote de orderbooks no repositório.
        
        Args:
            orderbooks: Lista de orderbooks a serem salvos
        """
        pass
    
    @abstractmethod
    async def get_orderbook(
        self, 
        exchange: str, 
        trading_pair: str, 
        timestamp: datetime
    ) -> Optional[OrderBook]:
        """
        Recupera um orderbook específico do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timestamp: Timestamp do orderbook
            
        Returns:
            Optional[OrderBook]: O orderbook encontrado ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_latest_orderbook(
        self, 
        exchange: str, 
        trading_pair: str
    ) -> Optional[OrderBook]:
        """
        Recupera o orderbook mais recente do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            
        Returns:
            Optional[OrderBook]: O orderbook mais recente ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_orderbooks(
        self, 
        exchange: str, 
        trading_pair: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        interval_seconds: Optional[int] = None
    ) -> List[OrderBook]:
        """
        Recupera um conjunto de orderbooks do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de orderbooks a retornar (opcional)
            interval_seconds: Intervalo em segundos entre os orderbooks (opcional)
            
        Returns:
            List[OrderBook]: Lista de orderbooks encontrados
        """
        pass
    
    @abstractmethod
    async def get_orderbook_availability(
        self,
        exchange: str,
        trading_pair: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Verifica a disponibilidade de dados de orderbook.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            
        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: Tupla contendo o timestamp 
            do orderbook mais antigo e do mais recente, ou (None, None) se não houver dados
        """
        pass