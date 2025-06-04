"""
Módulo principal para o componente de Analytics.

Este módulo inicializa e orquestra todos os componentes do módulo de Analytics,
incluindo serviços, repositórios e APIs.
"""
import os
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from src.common.infrastructure.config import (
    DatabaseConfig, ServiceConfig, load_config_from_env
)
from src.common.infrastructure.logging import setup_logger
from src.analytics.domain.value_objects.performance_metric import (
    PerformanceMetric, PerformanceMetricCollection, PerformanceAnalysis, MetricType
)
from src.analytics.application.services.performance_attribution_service import PerformanceAttributionService
from src.analytics.application.services.analytics_service import AnalyticsService
from src.analytics.application.use_cases.generate_analytics_report_use_case import GenerateAnalyticsReportUseCase
from src.analytics.infrastructure.dashboard_publisher import DashboardPublisher
from src.analytics.infrastructure.data_warehouse import DataWarehouse


class AnalyticsModule:
    """
    Classe principal do módulo de Analytics.
    
    Esta classe é responsável por inicializar e gerenciar todos os
    componentes do módulo de Analytics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o módulo de Analytics.
        
        Args:
            config_path: Caminho para o arquivo de configuração (opcional)
        """
        # Carregar configurações
        self.config = load_config_from_env(config_path)
        
        # Configurar logger
        self.logger = setup_logger(
            "analytics", 
            log_level=self.config.logging.level,
            log_file=self.config.logging.file
        )
        
        self.logger.info("Inicializando módulo de Analytics")
        
        # Inicializar componentes
        self._init_components()
    
    def _init_components(self) -> None:
        """Inicializa todos os componentes do módulo."""
        # Inicializar componentes de infraestrutura
        self._init_infrastructure()
        
        # Inicializar serviços
        self._init_services()
        
        # Inicializar casos de uso
        self._init_use_cases()
        
        self.logger.info("Todos os componentes do módulo de Analytics inicializados")
    
    def _init_infrastructure(self) -> None:
        """Inicializa componentes de infraestrutura."""
        self.logger.info("Inicializando componentes de infraestrutura")
        
        # Data Warehouse
        self.data_warehouse = DataWarehouse(
            config=self.config.database,
            logger=self.logger.getChild("data_warehouse")
        )
        
        # Dashboard Publisher
        reports_dir = os.path.join(
            self.config.services.output_dir, 
            "reports"
        )
        templates_dir = os.path.join(
            Path(__file__).parent, 
            "infrastructure", 
            "templates"
        )
        
        self.dashboard_publisher = DashboardPublisher(
            output_dir=reports_dir,
            template_dir=templates_dir,
            base_url=self.config.services.base_url + "/reports",
            logger=self.logger.getChild("dashboard_publisher")
        )
    
    def _init_services(self) -> None:
        """Inicializa serviços de aplicação."""
        self.logger.info("Inicializando serviços de aplicação")
        
        # Performance Attribution Service
        self.performance_service = PerformanceAttributionService(
            logger=self.logger.getChild("performance_service")
        )
        
        # Analytics Service
        self.analytics_service = AnalyticsService(
            logger=self.logger.getChild("analytics_service")
        )
    
    def _init_use_cases(self) -> None:
        """Inicializa casos de uso."""
        self.logger.info("Inicializando casos de uso")
        
        # Generate Analytics Report Use Case
        self.generate_report_use_case = GenerateAnalyticsReportUseCase(
            performance_service=self.performance_service,
            analytics_service=self.analytics_service,
            data_warehouse=self.data_warehouse,
            dashboard_publisher=self.dashboard_publisher,
            logger=self.logger.getChild("generate_report_use_case")
        )
    
    async def generate_report(
        self,
        user_id: str,
        report_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise.
        
        Args:
            user_id: ID do usuário
            report_type: Tipo de relatório
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        self.logger.info(f"Gerando relatório do tipo '{report_type}' para usuário {user_id}")
        
        return await self.generate_report_use_case.execute(
            user_id=user_id,
            report_type=report_type,
            parameters=parameters
        )
    
    async def get_performance_metrics(
        self,
        user_id: str,
        strategy_ids: Optional[list] = None,
        time_range: str = "month",
        **kwargs
    ) -> list:
        """
        Obtém métricas de desempenho para estratégias.
        
        Args:
            user_id: ID do usuário
            strategy_ids: Lista de IDs de estratégias (opcional)
            time_range: Intervalo de tempo (day, week, month, year, all)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Lista de métricas de desempenho por estratégia
        """
        self.logger.info(f"Obtendo métricas de desempenho para usuário {user_id}")
        
        return await self.analytics_service.get_performance_metrics(
            user_id=user_id,
            strategy_ids=strategy_ids,
            time_range=time_range,
            **kwargs
        )
    
    async def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Obtém resumo do portfolio.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Resumo do portfolio com principais métricas
        """
        self.logger.info(f"Obtendo resumo do portfolio para usuário {user_id}")
        
        return await self.analytics_service.get_portfolio_summary(user_id)
    
    async def get_exposure_analysis(self, user_id: str) -> Dict[str, Any]:
        """
        Obtém análise de exposição do portfolio.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Análise detalhada de exposição
        """
        self.logger.info(f"Obtendo análise de exposição para usuário {user_id}")
        
        return await self.analytics_service.get_exposure_analysis(user_id)
    
    async def get_trade_analysis(
        self,
        user_id: str,
        page: int = 1,
        size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Obtém análise de trades com paginação.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            filters: Filtros adicionais
            
        Returns:
            Análise de trades paginada
        """
        self.logger.info(f"Obtendo análise de trades para usuário {user_id}")
        
        return await self.analytics_service.get_trade_analysis(
            user_id=user_id,
            page=page,
            size=size,
            filters=filters
        )
    
    async def get_market_analysis(
        self,
        trading_pairs: list,
        time_range: str = "day",
        **kwargs
    ) -> list:
        """
        Obtém análise de mercado para pares de trading.
        
        Args:
            trading_pairs: Lista de pares de trading
            time_range: Intervalo de tempo
            **kwargs: Parâmetros adicionais
            
        Returns:
            Lista de análises por par de trading
        """
        self.logger.info(f"Obtendo análise de mercado para {len(trading_pairs)} pares")
        
        return await self.analytics_service.get_market_analysis(
            trading_pairs=trading_pairs,
            time_range=time_range,
            **kwargs
        )
    
    async def list_reports(
        self,
        user_id: str,
        page: int = 1,
        size: int = 10
    ) -> list:
        """
        Lista relatórios gerados anteriormente.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            
        Returns:
            Lista de relatórios
        """
        self.logger.info(f"Listando relatórios para usuário {user_id}")
        
        return await self.analytics_service.list_reports(
            user_id=user_id,
            page=page,
            size=size
        )
    
    async def list_dashboards(self, user_id: str) -> list:
        """
        Lista configurações de dashboards disponíveis.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de configurações de dashboard
        """
        self.logger.info(f"Listando dashboards para usuário {user_id}")
        
        return await self.analytics_service.list_dashboards(user_id)
    
    async def create_dashboard(
        self,
        user_id: str,
        dashboard: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cria um novo dashboard.
        
        Args:
            user_id: ID do usuário
            dashboard: Configuração do dashboard
            
        Returns:
            Dashboard criado
        """
        self.logger.info(f"Criando dashboard para usuário {user_id}")
        
        return await self.analytics_service.create_dashboard(
            user_id=user_id,
            dashboard=dashboard
        )
    
    async def list_alerts(self, user_id: str) -> list:
        """
        Lista configurações de alertas.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de configurações de alerta
        """
        self.logger.info(f"Listando alertas para usuário {user_id}")
        
        return await self.analytics_service.list_alerts(user_id)
    
    async def create_alert(
        self,
        user_id: str,
        alert_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cria uma nova configuração de alerta.
        
        Args:
            user_id: ID do usuário
            alert_config: Configuração do alerta
            
        Returns:
            Alerta criado
        """
        self.logger.info(f"Criando alerta para usuário {user_id}")
        
        return await self.analytics_service.create_alert(
            user_id=user_id,
            alert_config=alert_config
        )
    
    async def get_alert_history(
        self,
        user_id: str,
        page: int = 1,
        size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        Obtém histórico de alertas disparados.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            filters: Filtros adicionais
            
        Returns:
            Lista de instâncias de alerta
        """
        self.logger.info(f"Obtendo histórico de alertas para usuário {user_id}")
        
        return await self.analytics_service.get_alert_history(
            user_id=user_id,
            page=page,
            size=size,
            filters=filters
        )


# Função para inicializar o módulo
def initialize_analytics_module(config_path: Optional[str] = None) -> AnalyticsModule:
    """
    Inicializa o módulo de Analytics.
    
    Args:
        config_path: Caminho para o arquivo de configuração
        
    Returns:
        Instância do módulo de Analytics
    """
    return AnalyticsModule(config_path)


# Função para executar exemplo de uso
async def run_example():
    """Executa um exemplo de uso do módulo de Analytics."""
    # Inicializar módulo
    analytics = initialize_analytics_module()
    
    # Gerar um relatório de exemplo
    user_id = "test_user"
    result = await analytics.generate_report(
        user_id=user_id,
        report_type="performance_summary",
        parameters={
            "strategy_id": "strategy-123",
            "time_range": "month",
            "include_charts": True,
            "report_name": "Performance Summary - Test"
        }
    )
    
    print(f"Relatório gerado: {result['url']}")
    
    # Obter métricas de desempenho
    metrics = await analytics.get_performance_metrics(
        user_id=user_id,
        strategy_ids=["strategy-123"],
        time_range="month"
    )
    
    print(f"Métricas de desempenho: {len(metrics)} estratégias analisadas")
    
    # Obter resumo do portfolio
    portfolio = await analytics.get_portfolio_summary(user_id)
    
    print(f"Portfolio: {portfolio['total_capital']} total, {portfolio['allocated_capital']} alocado")


# Executar exemplo se o script for executado diretamente
if __name__ == "__main__":
    asyncio.run(run_example())