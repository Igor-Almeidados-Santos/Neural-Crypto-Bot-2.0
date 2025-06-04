from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union

from src.risk_management.application.services.circuit_breaker_service import CircuitBreakerService
from src.risk_management.application.services.exposure_service import ExposureService
from src.risk_management.application.services.portfolio_analytics_service import PortfolioAnalyticsService, PortfolioStats
from src.risk_management.domain.entities.risk_metric import RiskMetric
from src.risk_management.domain.entities.risk_profile import RiskProfile
from src.risk_management.domain.events.risk_threshold_breached_event import RiskThresholdBreachedEvent
from src.risk_management.infrastructure.risk_repository import RiskRepository


class ReportTimeframe(Enum):
    """Períodos de tempo para relatórios."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    CUSTOM = "custom"


@dataclass
class RiskReportRequest:
    """Solicitação para geração de relatório de risco."""
    portfolio_id: str
    timeframe: ReportTimeframe
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    include_metrics: List[str] = field(default_factory=list)
    include_positions: bool = True
    include_breaches: bool = True
    include_circuit_breakers: bool = True
    include_recommendations: bool = True
    format: str = "json"  # "json", "html", "pdf", "csv"


@dataclass
class MetricSummary:
    """Resumo de uma métrica de risco para um relatório."""
    name: str
    current_value: float
    average_value: float
    min_value: float
    max_value: float
    trend: str  # "up", "down", "stable"
    status: str  # "normal", "warning", "critical"
    threshold: Optional[float] = None


@dataclass
class RiskReport:
    """Relatório completo de risco."""
    portfolio_id: str
    generation_time: datetime
    timeframe: ReportTimeframe
    start_date: datetime
    end_date: datetime
    portfolio_stats: PortfolioStats
    metrics_summary: List[MetricSummary]
    risk_breaches: List[Dict[str, str]]
    circuit_breaker_events: List[Dict[str, str]]
    exposure_breakdown: Dict[str, Dict[str, float]]
    asset_risk_contribution: Dict[str, Dict[str, float]]
    correlation_matrix: Optional[Dict[str, Dict[str, float]]]
    recommendations: List[str]
    report_id: str


class GenerateRiskReportUseCase:
    """
    Caso de uso para gerar relatórios de risco detalhados.
    """
    
    def __init__(
        self,
        risk_repository: RiskRepository,
        portfolio_analytics_service: PortfolioAnalyticsService,
        exposure_service: ExposureService,
        circuit_breaker_service: CircuitBreakerService,
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            risk_repository: Repositório para acesso a dados de risco
            portfolio_analytics_service: Serviço de análise de portfólio
            exposure_service: Serviço de gerenciamento de exposição
            circuit_breaker_service: Serviço de circuit breaker
        """
        self._risk_repository = risk_repository
        self._portfolio_analytics = portfolio_analytics_service
        self._exposure_service = exposure_service
        self._circuit_breaker_service = circuit_breaker_service
    
    def execute(self, request: RiskReportRequest) -> RiskReport:
        """
        Executa a geração de um relatório de risco.
        
        Args:
            request: Solicitação de relatório com parâmetros
            
        Returns:
            Relatório de risco completo
        """
        # Definir período do relatório com base no timeframe solicitado
        now = datetime.utcnow()
        
        if request.timeframe != ReportTimeframe.CUSTOM:
            end_date = now
            if request.timeframe == ReportTimeframe.DAILY:
                start_date = now - timedelta(days=1)
            elif request.timeframe == ReportTimeframe.WEEKLY:
                start_date = now - timedelta(days=7)
            elif request.timeframe == ReportTimeframe.MONTHLY:
                start_date = now - timedelta(days=30)
            else:  # QUARTERLY
                start_date = now - timedelta(days=90)
        else:
            # Usar datas personalizadas se fornecidas
            if not request.start_date or not request.end_date:
                raise ValueError("Para timeframe personalizado, é necessário fornecer start_date e end_date")
            start_date = request.start_date
            end_date = request.end_date
        
        # Obter perfil de risco do portfólio
        profile = self._risk_repository.get_risk_profile_by_portfolio(request.portfolio_id)
        if not profile:
            raise ValueError(f"Perfil de risco não encontrado para o portfólio {request.portfolio_id}")
        
        # Obter estatísticas do portfólio
        portfolio_stats = self._portfolio_analytics.calculate_portfolio_stats(request.portfolio_id)
        
        # Obter métricas de risco históricas
        historical_metrics = self._risk_repository.get_metrics_by_date_range(
            portfolio_id=request.portfolio_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Preparar resumo de métricas
        metrics_summary = self._prepare_metrics_summary(historical_metrics, profile)
        
        # Obter violações de limites de risco
        risk_breaches = []
        if request.include_breaches:
            breach_events = self._risk_repository.get_threshold_breaches(
                portfolio_id=request.portfolio_id,
                start_date=start_date,
                end_date=end_date
            )
            
            for event in breach_events:
                risk_breaches.append({
                    "timestamp": event.timestamp.isoformat(),
                    "metric_type": event.metric.type.name,
                    "threshold": str(event.threshold_value),
                    "actual_value": str(event.actual_value),
                    "severity": "CRITICAL" if event.is_critical else "WARNING",
                    "message": event.get_message()
                })
        
        # Obter eventos de circuit breaker
        circuit_breaker_events = []
        if request.include_circuit_breakers:
            cb_status = self._circuit_breaker_service.get_all_circuit_breakers()
            
            for entity_id, status in cb_status.items():
                if entity_id == request.portfolio_id or status.get("portfolio_id") == request.portfolio_id:
                    circuit_breaker_events.append({
                        "entity_id": entity_id,
                        "name": status.get("name", "Unknown"),
                        "state": status.get("state", "Unknown"),
                        "last_updated": status.get("last_updated", "Unknown"),
                        "last_event": status.get("last_event", "")
                    })
        
        # Obter breakdown de exposição
        exposure_breakdown = self._exposure_service.get_exposure_metrics()
        
        # Calcular contribuição de risco por ativo
        asset_risk_contribution = self._portfolio_analytics.calculate_asset_risk_contribution(request.portfolio_id)
        
        # Obter matriz de correlação
        correlation_matrix = self._portfolio_analytics.get_correlation_matrix()
        
        # Gerar recomendações
        recommendations = []
        if request.include_recommendations:
            recommendations = self._generate_recommendations(
                portfolio_stats, metrics_summary, risk_breaches, exposure_breakdown
            )
        
        # Criar ID do relatório
        report_id = f"risk_report_{request.portfolio_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Montar relatório final
        report = RiskReport(
            portfolio_id=request.portfolio_id,
            generation_time=now,
            timeframe=request.timeframe,
            start_date=start_date,
            end_date=end_date,
            portfolio_stats=portfolio_stats,
            metrics_summary=metrics_summary,
            risk_breaches=risk_breaches,
            circuit_breaker_events=circuit_breaker_events,
            exposure_breakdown=exposure_breakdown,
            asset_risk_contribution=asset_risk_contribution,
            correlation_matrix=correlation_matrix,
            recommendations=recommendations,
            report_id=report_id
        )
        
        # Salvar relatório no repositório para histórico
        self._risk_repository.save_risk_report(report)
        
        return report
    
    def _prepare_metrics_summary(
        self, metrics: List[RiskMetric], profile: RiskProfile
    ) -> List[MetricSummary]:
        """
        Prepara o resumo das métricas de risco com análise de tendências.
        
        Args:
            metrics: Lista de métricas históricas
            profile: Perfil de risco com limites
            
        Returns:
            Lista de resumos de métricas
        """
        # Agrupar métricas por tipo
        metrics_by_type = {}
        for metric in metrics:
            if metric.type not in metrics_by_type:
                metrics_by_type[metric.type] = []
            metrics_by_type[metric.type].append(metric)
        
        summaries = []
        
        for metric_type, metric_list in metrics_by_type.items():
            if not metric_list:
                continue
            
            # Ordenar por timestamp
            sorted_metrics = sorted(metric_list, key=lambda m: m.timestamp)
            
            # Pegar valores para análise
            values = [m.value for m in sorted_metrics]
            current_value = values[-1] if values else 0
            avg_value = sum(values) / len(values) if values else 0
            min_value = min(values) if values else 0
            max_value = max(values) if values else 0
            
            # Determinar tendência
            if len(values) >= 2:
                # Calcular tendência com base nos últimos 30% dos pontos ou pelo menos 3 pontos
                trend_window = max(3, int(len(values) * 0.3))
                recent_values = values[-trend_window:]
                
                if recent_values[-1] > recent_values[0] * 1.05:  # 5% de aumento
                    trend = "up"
                elif recent_values[-1] < recent_values[0] * 0.95:  # 5% de diminuição
                    trend = "down"
                else:
                    trend = "stable"
            else:
                trend = "stable"
            
            # Determinar status com base nos limites do perfil
            status = "normal"
            threshold = None
            
            if metric_type in profile.thresholds:
                threshold_config = profile.thresholds[metric_type]
                threshold = threshold_config.critical_threshold
                
                warning_breached, critical_breached = profile.check_threshold_breach(metric_type, current_value)
                
                if critical_breached:
                    status = "critical"
                elif warning_breached:
                    status = "warning"
            
            # Criar resumo da métrica
            summary = MetricSummary(
                name=metric_type.name,
                current_value=current_value,
                average_value=avg_value,
                min_value=min_value,
                max_value=max_value,
                trend=trend,
                status=status,
                threshold=threshold
            )
            
            summaries.append(summary)
        
        return summaries
    
    def _generate_recommendations(
        self,
        stats: PortfolioStats,
        metrics_summary: List[MetricSummary],
        risk_breaches: List[Dict[str, str]],
        exposure_breakdown: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """
        Gera recomendações com base na análise das métricas e violações.
        
        Args:
            stats: Estatísticas do portfólio
            metrics_summary: Resumo de métricas
            risk_breaches: Lista de violações de limites
            exposure_breakdown: Breakdown de exposição
            
        Returns:
            Lista de recomendações
        """
        recommendations = []
        
        # Verificar alta concentração
        if stats.concentration_index > 0.25:
            recommendations.append(
                "Concentração de portfólio elevada. Considere diversificar as posições para reduzir "
                "o risco de concentração."
            )
        
        # Verificar correlação média alta
        if stats.correlation_avg > 0.7:
            recommendations.append(
                "Alta correlação entre ativos. O portfólio pode não estar adequadamente diversificado. "
                "Considere adicionar ativos com baixa correlação."
            )
        
        # Verificar Sharpe Ratio baixo
        if stats.sharpe_ratio < 0.5:
            recommendations.append(
                f"Sharpe Ratio baixo ({stats.sharpe_ratio:.2f}). Considere revisar estratégias para "
                "melhorar a relação risco/retorno."
            )
        
        # Verificar drawdown atual
        if stats.drawdown_current > 0.1:  # Mais de 10%
            recommendations.append(
                f"Drawdown atual de {stats.drawdown_current:.1f}%. Considere ajustar tamanhos de "
                "posição ou implementar estratégias de hedge."
            )
        
        # Verificar métricas em estado crítico
        critical_metrics = [m for m in metrics_summary if m.status == "critical"]
        if critical_metrics:
            for metric in critical_metrics:
                recommendations.append(
                    f"Métrica {metric.name} em estado CRÍTICO ({metric.current_value:.2f}). "
                    f"Ação imediata recomendada para reduzir exposição a risco."
                )
        
        # Verificar métricas com tendência ascendente negativa
        risk_metrics_increasing = [
            m for m in metrics_summary 
            if m.trend == "up" and m.name in ["VALUE_AT_RISK", "EXPECTED_SHORTFALL", "DRAWDOWN", "VOLATILITY"]
        ]
        if risk_metrics_increasing:
            metrics_names = ", ".join([m.name for m in risk_metrics_increasing])
            recommendations.append(
                f"Métricas de risco com tendência de aumento: {metrics_names}. "
                f"Monitore de perto e considere reduzir exposição se continuar aumentando."
            )
        
        # Analisar exposição por categoria
        categories = exposure_breakdown.get("categories", {})
        high_exposure_categories = []
        for category, data in categories.items():
            if data.get("status") in ("warning", "critical"):
                high_exposure_categories.append(category)
        
        if high_exposure_categories:
            categories_list = ", ".join(high_exposure_categories)
            recommendations.append(
                f"Alta exposição nas categorias: {categories_list}. "
                f"Considere rebalancear o portfólio para melhor distribuição de risco."
            )
        
        # Verificar se houve muitas violações recentes
        if len(risk_breaches) > 5:
            recommendations.append(
                f"Alto número de violações de limites de risco ({len(risk_breaches)}). "
                f"Considere revisar os parâmetros da estratégia ou ajustar os limites de risco."
            )
        
        # Recomendações baseadas em volatilidade
        if stats.volatility_daily > 3.0:  # Alta volatilidade diária
            recommendations.append(
                f"Volatilidade diária elevada ({stats.volatility_daily:.2f}%). "
                f"Considere reduzir tamanho das posições ou implementar estratégias de baixa volatilidade."
            )
        
        return recommendations