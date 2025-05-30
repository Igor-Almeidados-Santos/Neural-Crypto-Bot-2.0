"""
Repositório para persistência e recuperação de eventos de liquidação.

Este módulo define a interface do repositório para eventos de liquidação,
seguindo o padrão de Repository do Domain-Driven Design.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from src.data_collection.domain.entities.liquidation import Liquidation, LiquidationSide


class LiquidationRepository(ABC):
    """
    Interface para repositório de eventos de liquidação.
    
    Define operações para persistência e recuperação de eventos de liquidação,
    independente da tecnologia de armazenamento utilizada.
    """
    
    @abstractmethod
    async def save(self, liquidation: Liquidation) -> None:
        """
        Salva um evento de liquidação no repositório.
        
        Args:
            liquidation: Evento de liquidação a ser salvo
        """
        pass
    
    @abstractmethod
    async def save_batch(self, liquidations: List[Liquidation]) -> None:
        """
        Salva um lote de eventos de liquidação no repositório.
        
        Args:
            liquidations: Lista de eventos de liquidação a serem salvos
        """
        pass
    
    @abstractmethod
    async def get_liquidations(
        self, 
        exchange: Optional[str] = None,
        trading_pair: Optional[str] = None,
        side: Optional[LiquidationSide] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Liquidation]:
        """
        Recupera eventos de liquidação do repositório com base em filtros.
        
        Args:
            exchange: Nome da exchange (opcional)
            trading_pair: Par de negociação (opcional)
            side: Lado da liquidação (opcional)
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de eventos a retornar (opcional)
            
        Returns:
            List[Liquidation]: Lista de eventos de liquidação encontrados
        """
        pass
    
    @abstractmethod
    async def get_liquidation_stats(
        self,
        exchange: Optional[str] = None,
        trading_pair: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calcula estatísticas sobre eventos de liquidação.
        
        Args:
            exchange: Nome da exchange (opcional)
            trading_pair: Par de negociação (opcional)
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            
        Returns:
            Dict[str, Any]: Dicionário com estatísticas, incluindo:
                - total_count: Número total de liquidações
                - total_value: Valor total das liquidações
                - long_count: Número de liquidações de posições longas
                - short_count: Número de liquidações de posições curtas
                - long_value: Valor total das liquidações de posições longas
                - short_value: Valor total das liquidações de posições curtas
                - largest_liquidation: Maior liquidação individual
                - most_liquidated_pair: Par com mais liquidações
        """
        pass
    
    @abstractmethod
    async def get_liquidation_timeline(
        self,
        exchange: Optional[str] = None,
        trading_pair: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        interval_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Retorna dados de liquidação agregados por intervalo de tempo.
        
        Args:
            exchange: Nome da exchange (opcional)
            trading_pair: Par de negociação (opcional)
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            interval_minutes: Intervalo de agregação em minutos
            
        Returns:
            List[Dict[str, Any]]: Lista de dicionários, cada um contendo:
                - timestamp: Início do intervalo
                - count: Número de liquidações no intervalo
                - value: Valor total das liquidações no intervalo
                - long_count: Número de liquidações de posições longas
                - short_count: Número de liquidações de posições curtas
                - long_value: Valor de liquidações de posições longas
                - short_value: Valor de liquidações de posições curtas
        """
        pass
    
    @abstractmethod
    async def get_recent_large_liquidations(
        self,
        exchange: Optional[str] = None,
        trading_pair: Optional[str] = None,
        min_value: Optional[float] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[Liquidation]:
        """
        Recupera liquidações recentes de grande valor.
        
        Args:
            exchange: Nome da exchange (opcional)
            trading_pair: Par de negociação (opcional)
            min_value: Valor mínimo para considerar uma liquidação grande (opcional)
            hours: Número de horas atrás para considerar
            limit: Número máximo de eventos a retornar
            
        Returns:
            List[Liquidation]: Lista de eventos de liquidação de grande valor
        """
        pass