"""
Repositório para persistência e recuperação de taxas de financiamento.

Este módulo define a interface do repositório para taxas de financiamento,
seguindo o padrão de Repository do Domain-Driven Design.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple

from src.data_collection.domain.entities.funding_rate import FundingRate


class FundingRateRepository(ABC):
    """
    Interface para repositório de taxas de financiamento.
    
    Define operações para persistência e recuperação de taxas de financiamento,
    independente da tecnologia de armazenamento utilizada.
    """
    
    @abstractmethod
    async def save(self, funding_rate: FundingRate) -> None:
        """
        Salva uma taxa de financiamento no repositório.
        
        Args:
            funding_rate: Taxa de financiamento a ser salva
        """
        pass
    
    @abstractmethod
    async def save_batch(self, funding_rates: List[FundingRate]) -> None:
        """
        Salva um lote de taxas de financiamento no repositório.
        
        Args:
            funding_rates: Lista de taxas de financiamento a serem salvas
        """
        pass
    
    @abstractmethod
    async def get_funding_rate(
        self, 
        exchange: str, 
        trading_pair: str, 
        timestamp: datetime
    ) -> Optional[FundingRate]:
        """
        Recupera uma taxa de financiamento específica do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timestamp: Timestamp da taxa
            
        Returns:
            Optional[FundingRate]: A taxa de financiamento encontrada ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_funding_rates(
        self, 
        exchange: str, 
        trading_pair: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[FundingRate]:
        """
        Recupera um conjunto de taxas de financiamento do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de taxas a retornar (opcional)
            
        Returns:
            List[FundingRate]: Lista de taxas de financiamento encontradas
        """
        pass
    
    @abstractmethod
    async def get_latest_funding_rate(
        self, 
        exchange: str, 
        trading_pair: str
    ) -> Optional[FundingRate]:
        """
        Recupera a taxa de financiamento mais recente do repositório.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            
        Returns:
            Optional[FundingRate]: A taxa de financiamento mais recente ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_funding_rate_availability(
        self,
        exchange: str,
        trading_pair: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Verifica a disponibilidade de dados de taxas de financiamento.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            
        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: Tupla contendo o timestamp 
            da taxa mais antiga e da taxa mais recente, ou (None, None) se não houver dados
        """
        pass
    
    @abstractmethod
    async def get_average_funding_rate(
        self,
        exchange: str,
        trading_pair: str,
        period_days: int = 7
    ) -> Optional[float]:
        """
        Calcula a taxa média de financiamento para um período.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            period_days: Número de dias para o cálculo da média
            
        Returns:
            Optional[float]: Taxa média de financiamento para o período ou None se não houver dados
        """
        pass