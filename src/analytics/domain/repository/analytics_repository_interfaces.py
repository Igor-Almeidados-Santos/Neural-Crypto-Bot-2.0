"""
Interfaces de repositório para o domínio Analytics.

Este módulo define as interfaces de repositório que permitem
a persistência e recuperação de dados de análise.
"""
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.analytics.domain.value_objects.performance_metric import (
    PerformanceMetric, PerformanceMetricCollection, PerformanceAnalysis
)


class PerformanceRepository(ABC):
    """Interface para repositório de dados de desempenho."""
    
    @abstractmethod
    async def save_metric(self, metric: PerformanceMetric) -> str:
        """
        Salva uma métrica de desempenho.
        
        Args:
            metric: Métrica a ser salva
            
        Returns:
            ID da métrica salva
        """
        pass
    
    @abstractmethod
    async def save_metric_collection(self, collection: PerformanceMetricCollection) -> str:
        """
        Salva uma coleção de métricas.
        
        Args:
            collection: Coleção de métricas a ser salva
            
        Returns:
            ID da coleção salva
        """
        pass
    
    @abstractmethod
    async def save_performance_analysis(self, analysis: PerformanceAnalysis) -> str:
        """
        Salva uma análise de desempenho.
        
        Args:
            analysis: Análise a ser salva
            
        Returns:
            ID da análise salva
        """
        pass
    
    @abstractmethod
    async def get_metrics_by_strategy(
        self, 
        strategy_id: str, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> List[PerformanceMetric]:
        """
        Obtém métricas para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            start_date: Data de início (opcional)
            end_date: Data de fim (opcional)
            
        Returns:
            Lista de métricas
        """
        pass
    
    @abstractmethod
    async def get_performance_analysis(
        self, 
        strategy_id: str, 
        time_range: str
    ) -> Optional[PerformanceAnalysis]:
        """
        Obtém uma análise de desempenho para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            time_range: Intervalo de tempo
            
        Returns:
            Análise de desempenho ou None se não existir
        """
        pass
    
    @abstractmethod
    async def get_latest_metrics(
        self,
        strategy_id: str,
        metric_types: Optional[List[str]] = None
    ) -> Dict[str, PerformanceMetric]:
        """
        Obtém as métricas mais recentes para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            metric_types: Lista de tipos de métrica (opcional)
            
        Returns:
            Dicionário com as métricas mais recentes
        """
        pass


class AnalyticsReportRepository(ABC):
    """Interface para repositório de relatórios de análise."""
    
    @abstractmethod
    async def save_report(
        self, 
        user_id: str, 
        report_name: str, 
        report_type: str, 
        parameters: Dict[str, Any],
        url: str
    ) -> str:
        """
        Salva um relatório.
        
        Args:
            user_id: ID do usuário
            report_name: Nome do relatório
            report_type: Tipo do relatório
            parameters: Parâmetros usados para gerar o relatório
            url: URL para acessar o relatório
            
        Returns:
            ID do relatório salvo
        """
        pass
    
    @abstractmethod
    async def get_reports_by_user(
        self, 
        user_id: str, 
        page: int = 1, 
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Obtém relatórios para um usuário com paginação.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            
        Returns:
            Dicionário com os relatórios e informações de paginação
        """
        pass
    
    @abstractmethod
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém um relatório pelo ID.
        
        Args:
            report_id: ID do relatório
            
        Returns:
            Dicionário com os dados do relatório ou None se não existir
        """
        pass
    
    @abstractmethod
    async def delete_report(self, report_id: str) -> bool:
        """
        Exclui um relatório.
        
        Args:
            report_id: ID do relatório
            
        Returns:
            True se excluído com sucesso, False caso contrário
        """
        pass


class DashboardRepository(ABC):
    """Interface para repositório de dashboards."""
    
    @abstractmethod
    async def save_dashboard(
        self, 
        user_id: str, 
        dashboard_name: str, 
        layout: Dict[str, Any],
        is_default: bool = False
    ) -> str:
        """
        Salva um dashboard.
        
        Args:
            user_id: ID do usuário
            dashboard_name: Nome do dashboard
            layout: Layout do dashboard
            is_default: Se é o dashboard padrão
            
        Returns:
            ID do dashboard salvo
        """
        pass
    
    @abstractmethod
    async def get_dashboards_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Obtém dashboards para um usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de dashboards
        """
        pass
    
    @abstractmethod
    async def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém um dashboard pelo ID.
        
        Args:
            dashboard_id: ID do dashboard
            
        Returns:
            Dados do dashboard ou None se não existir
        """
        pass
    
    @abstractmethod
    async def update_dashboard(
        self, 
        dashboard_id: str, 
        dashboard_data: Dict[str, Any]
    ) -> bool:
        """
        Atualiza um dashboard.
        
        Args:
            dashboard_id: ID do dashboard
            dashboard_data: Dados atualizados
            
        Returns:
            True se atualizado com sucesso, False caso contrário
        """
        pass
    
    @abstractmethod
    async def delete_dashboard(self, dashboard_id: str) -> bool:
        """
        Exclui um dashboard.
        
        Args:
            dashboard_id: ID do dashboard
            
        Returns:
            True se excluído com sucesso, False caso contrário
        """
        pass


class AlertRepository(ABC):
    """Interface para repositório de alertas de analytics."""
    
    @abstractmethod
    async def save_alert_config(
        self, 
        user_id: str, 
        name: str, 
        metric_type: str,
        strategy_id: Optional[str],
        condition: str,
        threshold: float,
        level: str,
        notification_channels: List[str]
    ) -> str:
        """
        Salva uma configuração de alerta.
        
        Args:
            user_id: ID do usuário
            name: Nome do alerta
            metric_type: Tipo de métrica monitorada
            strategy_id: ID da estratégia (opcional)
            condition: Condição (>, <, ==, etc)
            threshold: Valor limiar
            level: Nível do alerta (info, warning, error)
            notification_channels: Canais de notificação
            
        Returns:
            ID da configuração salva
        """
        pass
    
    @abstractmethod
    async def get_alert_configs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Obtém configurações de alerta para um usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de configurações de alerta
        """
        pass
    
    @abstractmethod
    async def save_alert_instance(
        self, 
        config_id: str, 
        value: float,
        message: str,
        timestamp: datetime
    ) -> str:
        """
        Salva uma instância de alerta disparado.
        
        Args:
            config_id: ID da configuração
            value: Valor que disparou o alerta
            message: Mensagem do alerta
            timestamp: Timestamp de disparo
            
        Returns:
            ID da instância salva
        """
        pass
    
    @abstractmethod
    async def get_alert_history(
        self, 
        user_id: str, 
        page: int = 1, 
        size: int = 20, 
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Obtém histórico de alertas com paginação.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            filters: Filtros adicionais
            
        Returns:
            Dicionário com alertas e informações de paginação
        """
        pass
    
    @abstractmethod
    async def acknowledge_alert(self, alert_instance_id: str) -> bool:
        """
        Marca um alerta como reconhecido.
        
        Args:
            alert_instance_id: ID da instância de alerta
            
        Returns:
            True se reconhecido com sucesso, False caso contrário
        """
        pass