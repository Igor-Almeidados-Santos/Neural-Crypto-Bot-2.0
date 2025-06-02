"""
src/feature_engineering/domain/repositories/feature_repository.py
Interface de repositório para gerenciar entidades Feature.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from uuid import UUID

from src.feature_engineering.domain.entities.feature import Feature, FeatureCategory, FeatureType


class FeatureRepository(ABC):
    """Interface para repositório de features."""
    
    @abstractmethod
    async def add(self, feature: Feature) -> Feature:
        """
        Adiciona uma nova feature ao repositório.
        
        Args:
            feature: Feature a ser adicionada.
            
        Returns:
            Feature: Feature adicionada com ID atualizado.
        """
        pass
    
    @abstractmethod
    async def update(self, feature: Feature) -> Feature:
        """
        Atualiza uma feature existente.
        
        Args:
            feature: Feature a ser atualizada.
            
        Returns:
            Feature: Feature atualizada.
        """
        pass
    
    @abstractmethod
    async def get_by_id(self, feature_id: UUID) -> Optional[Feature]:
        """
        Obtém uma feature pelo ID.
        
        Args:
            feature_id: ID da feature.
            
        Returns:
            Optional[Feature]: Feature encontrada ou None se não existir.
        """
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Feature]:
        """
        Obtém uma feature pelo nome.
        
        Args:
            name: Nome da feature.
            
        Returns:
            Optional[Feature]: Feature encontrada ou None se não existir.
        """
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Feature]:
        """
        Obtém todas as features.
        
        Returns:
            List[Feature]: Lista de todas as features.
        """
        pass
    
    @abstractmethod
    async def delete(self, feature_id: UUID) -> bool:
        """
        Remove uma feature pelo ID.
        
        Args:
            feature_id: ID da feature a ser removida.
            
        Returns:
            bool: True se a feature foi removida, False caso contrário.
        """
        pass
    
    @abstractmethod
    async def get_by_type(self, feature_type: FeatureType) -> List[Feature]:
        """
        Obtém features por tipo.
        
        Args:
            feature_type: Tipo de feature.
            
        Returns:
            List[Feature]: Lista de features do tipo especificado.
        """
        pass