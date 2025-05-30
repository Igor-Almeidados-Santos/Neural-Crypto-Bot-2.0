"""
Repositório para persistência e recuperação de velas (candles).

Este módulo define a interface do repositório para velas, seguindo
o padrão de Repository do Domain-Driven Design.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple

from src.data_collection.domain.entities.candle import Candle, TimeFrame


class CandleRepository(ABC):
    """
    Interface para repositório de velas (candles).
    
    Define operações para persistência e recuperação de velas,
    independente da tecnologia de armazenamento utilizada.
    """
    
    @abstractmethod
    async def save(self, candle: Candle) -> None:
        """
        Salva uma vela no repositório.
        
        Args:
            candle: Vela a ser salva
        """
        pass
    
    @abstractmethod
    async def save_batch(self, candles: List[Candle]) -> None:
        """
        Salva um lote de velas no repositório.
        
        Args:
            candles: Lista de velas a serem salvas
        """
        pass
    
    @abstractmethod
    async def get_candle(
        self, 
        exchange: str, 
        trading_pair: str, 
        timestamp: datetime, 
        timeframe: TimeFrame
    ) -> Optional[Candle]:
        """
        Recupera uma vela específica do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timestamp: Timestamp da vela
            timeframe: Timeframe da vela
            
        Returns:
            Optional[Candle]: A vela encontrada ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_candles(
        self, 
        exchange: str, 
        trading_pair: str, 
        timeframe: TimeFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Recupera um conjunto de velas do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timeframe: Timeframe das velas
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de velas a retornar (opcional)
            
        Returns:
            List[Candle]: Lista de velas encontradas
        """
        pass
    
    @abstractmethod
    async def get_latest_candle(
        self, 
        exchange: str, 
        trading_pair: str, 
        timeframe: TimeFrame
    ) -> Optional[Candle]:
        """
        Recupera a vela mais recente do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timeframe: Timeframe da vela
            
        Returns:
            Optional[Candle]: A vela mais recente ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_timeframe_availability(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Verifica a disponibilidade de dados para um determinado timeframe.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timeframe: Timeframe das velas
            
        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: Tupla contendo o timestamp 
            da vela mais antiga e da vela mais recente, ou (None, None) se não houver dados
        """
        pass