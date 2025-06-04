import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from src.risk_management.application.services.circuit_breaker_service import (
    CircuitBreakerConfig, CircuitBreakerService, CircuitBreakerState
)
from src.risk_management.application.services.exposure_service import (
    ExposureLimit, ExposureService
)
from src.risk_management.application.services.portfolio_analytics_service import (
    PortfolioAnalyticsService, PortfolioPosition
)
from src.risk_management.application.use_cases.generate_risk_report_use_case import (
    GenerateRiskReportUseCase, ReportTimeframe, RiskReportRequest
)
from src.risk_management.application.use_cases.validate_order_risk_use_case import (
    OrderRiskParams, ValidateOrderRiskUseCase
)
from src.risk_management.domain.entities.risk_metric import MetricType, RiskMetric, TimeFrame
from src.risk_management.domain.entities.risk_profile import RiskLevel, RiskProfile
from src.risk_management.domain.events.risk_threshold_breached_event import RiskThresholdBreachedEvent
from src.risk_management.infrastructure.alert_notifier import (
    AlertChannel, AlertNotifier, AlertPriority
)
from src.risk_management.infrastructure.risk_repository import RiskRepository
from src.risk_management.models.drawdown_control import (
    DrawdownControl, DrawdownControlStrategy
)
from src.risk_management.models.expected_shortfall import ESMethod, ExpectedShortfallModel
from src.risk_management.models.var_model import VaRMethod, VaRModel


class RiskManagementSettings(BaseSettings):
    """Configurações do módulo de gerenciamento de risco."""
    db_connection_string: str = Field(..., env="RISK_DB_CONNECTION_STRING")
    create_tables: bool = Field(False, env="RISK_CREATE_TABLES")
    log_level: str = Field("INFO", env="RISK_LOG_LEVEL")
    
    default_max_drawdown_pct: float = Field(20.0, env="RISK_DEFAULT_MAX_DRAWDOWN_PCT")
    default_var_confidence_level: float = Field(0.95, env="RISK_DEFAULT_VAR_CONFIDENCE_LEVEL")
    default_max_position_size_pct: float = Field(5.0, env="RISK_DEFAULT_MAX_POSITION_SIZE_PCT")
    
    alert_email_recipients: str = Field("", env="RISK_ALERT_EMAIL_RECIPIENTS")
    alert_slack_webhook: str = Field("", env="RISK_ALERT_SLACK_WEBHOOK")
    alert_telegram_bot_token: str = Field("", env="RISK_ALERT_TELEGRAM_BOT_TOKEN")
    alert_telegram_chat_id: str = Field("", env="RISK_ALERT_TELEGRAM_CHAT_ID")
    
    initial_capital: float = Field(10000.0, env="RISK_INITIAL_CAPITAL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class RiskManager:
    """
    Gerenciador principal do módulo de risco.
    Responsável por inicializar e coordenar todos os componentes.
    """
    
    def __init__(self, settings: Optional[RiskManagementSettings] = None):
        """
        Inicializa o gerenciador de risco.
        
        Args:
            settings: Configurações do módulo (carregadas do ambiente se None)
        """
        self.settings = settings or RiskManagementSettings()
        self.logger = self._setup_logger()
        
        # Inicializar modelos
        self.var_model = self._setup_var_model()
        self.es_model = self._setup_es_model()
        self.drawdown_control = self._setup_drawdown_control()
        
        # Inicializar infraestrutura
        self.risk_repository = self._setup_risk_repository()
        self.alert_notifier = self._setup_alert_notifier()
        
        # Inicializar serviços
        self.circuit_breaker_service = self._setup_circuit_breaker_service()
        self.exposure_service = self._setup_exposure_service()
        self.portfolio_analytics_service = self._setup_portfolio_analytics_service()
        
        # Inicializar casos de uso
        self.validate_order_risk_use_case = self._setup_validate_order_risk_use_case()
        self.generate_risk_report_use_case = self._setup_generate_risk_report_use_case()
        
        self.logger.info("Módulo de gerenciamento de risco inicializado com sucesso")
    
    def _setup_logger(self) -> logging.Logger:
        """Configura e retorna o logger."""
        logger = logging.getLogger("risk_management")
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Configurar handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Definir formato do log
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        
        # Adicionar handler ao logger
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_var_model(self) -> VaRModel:
        """Configura e retorna o modelo VaR."""
        return VaRModel(
            default_method=VaRMethod.HISTORICAL,
            default_confidence_level=self.settings.default_var_confidence_level,
            time_horizon=1,
            use_ewma=True,
            ewma_lambda=0.94,
            mcmc_simulations=10000,
            distribution="t-student",
            degrees_of_freedom=5,
            logger=self.logger
        )
    
    def _setup_es_model(self) -> ExpectedShortfallModel:
        """Configura e retorna o modelo Expected Shortfall."""
        return ExpectedShortfallModel(
            default_method=ESMethod.HISTORICAL,
            default_confidence_level=self.settings.default_var_confidence_level,
            time_horizon=1,
            mcmc_simulations=10000,
            distribution="t-student",
            degrees_of_freedom=5,
            logger=self.logger
        )
    
    def _setup_drawdown_control(self) -> DrawdownControl:
        """Configura e retorna o controlador de drawdown."""
        return DrawdownControl(
            max_drawdown_pct=self.settings.default_max_drawdown_pct,
            warning_threshold_pct=self.settings.default_max_drawdown_pct * 0.7,
            strategy=DrawdownControlStrategy.ADAPTIVE_THRESHOLD,
            lookback_window=60,
            volatility_multiplier=2.0,
            time_decay_factor=0.5,
            trailing_pct=5.0,
            logger=self.logger
        )
    
    def _setup_risk_repository(self) -> RiskRepository:
        """Configura e retorna o repositório de risco."""
        return RiskRepository(
            connection_string=self.settings.db_connection_string,
            create_tables=self.settings.create_tables,
            logger=self.logger
        )
    
    def _setup_alert_notifier(self) -> AlertNotifier:
        """Configura e retorna o notificador de alertas."""
        channels = {}
        
        # Configurar email se disponível
        if self.settings.alert_email_recipients:
            channels[AlertChannel.EMAIL] = {
                "recipients": self.settings.alert_email_recipients
            }
        
        # Configurar Slack se disponível
        if self.settings.alert_slack_webhook:
            channels[AlertChannel.SLACK] = {
                "webhook_url": self.settings.alert_slack_webhook
            }
        
        # Configurar Telegram se disponível
        if self.settings.alert_telegram_bot_token and self.settings.alert_telegram_chat_id:
            channels[AlertChannel.TELEGRAM] = {
                "bot_token": self.settings.alert_telegram_bot_token,
                "chat_id": self.settings.alert_telegram_chat_id
            }
        
        return AlertNotifier(
            channels=channels,
            min_priority=AlertPriority.LOW,
            rate_limit_minutes=5,
            include_metrics=True,
            logger=self.logger
        )
    
    def _setup_circuit_breaker_service(self) -> CircuitBreakerService:
        """Configura e retorna o serviço de circuit breaker."""
        return CircuitBreakerService(
            alert_notifier=self.alert_notifier,
            logger=self.logger
        )
    
    def _setup_exposure_service(self) -> ExposureService:
        """Configura e retorna o serviço de gerenciamento de exposição."""
        return ExposureService(
            total_capital=self.settings.initial_capital,
            logger=self.logger
        )
    
    def _setup_portfolio_analytics_service(self) -> PortfolioAnalyticsService:
        """Configura e retorna o serviço de análise de portfólio."""
        return PortfolioAnalyticsService(
            var_model=self.var_model,
            es_model=self.es_model,
            drawdown_control=self.drawdown_control,
            risk_free_rate=0.03,  # 3% anual, ajustável
            logger=self.logger
        )
    
    def _setup_validate_order_risk_use_case(self) -> ValidateOrderRiskUseCase:
        """Configura e retorna o caso de uso de validação de risco de ordem."""
        return ValidateOrderRiskUseCase(
            risk_repository=self.risk_repository,
            exposure_service=self.exposure_service,
            circuit_breaker_service=self.circuit_breaker_service
        )
    
    def _setup_generate_risk_report_use_case(self) -> GenerateRiskReportUseCase:
        """Configura e retorna o caso de uso de geração de relatório de risco."""
        return GenerateRiskReportUseCase(
            risk_repository=self.risk_repository,
            portfolio_analytics_service=self.portfolio_analytics_service,
            exposure_service=self.exposure_service,
            circuit_breaker_service=self.circuit_breaker_service
        )
    
    def create_default_risk_profiles(self) -> None:
        """Cria perfis de risco padrão se não existirem."""
        # Verificar se já existem perfis
        existing_profiles = self.risk_repository.get_applicable_profiles()
        if existing_profiles:
            self.logger.info(f"Perfis de risco existentes encontrados: {len(existing_profiles)}")
            return
        
        # Criar perfis padrão
        conservative = RiskProfile.create_conservative(
            name="Conservador Global",
            description="Perfil de risco conservador para uso geral"
        )
        
        moderate = RiskProfile.create_moderate(
            name="Moderado Global",
            description="Perfil de risco moderado para uso geral"
        )
        
        aggressive = RiskProfile.create_aggressive(
            name="Agressivo Global",
            description="Perfil de risco agressivo para uso geral"
        )
        
        # Salvar perfis
        for profile in [conservative, moderate, aggressive]:
            self.risk_repository.save_risk_profile(profile)
        
        self.logger.info("Perfis de risco padrão criados com sucesso")
    
    def register_default_circuit_breakers(self) -> None:
        """Registra circuit breakers padrão para uso global."""
        # Circuit breaker para drawdown
        drawdown_config = CircuitBreakerConfig(
            name="Drawdown Global",
            cooldown_period_minutes=60,
            reset_threshold=3,
            consecutive_success_to_close=5,
            failure_threshold=3,
            recovery_timeout_minutes=120,
            auto_reset=True,
            requires_manual_reset_if_critical=True,
            metrics_to_monitor={MetricType.DRAWDOWN}
        )
        
        # Circuit breaker para Value at Risk
        var_config = CircuitBreakerConfig(
            name="VaR Global",
            cooldown_period_minutes=30,
            reset_threshold=2,
            consecutive_success_to_close=3,
            failure_threshold=2,
            recovery_timeout_minutes=60,
            auto_reset=True,
            requires_manual_reset_if_critical=True,
            metrics_to_monitor={MetricType.VALUE_AT_RISK}
        )
        
        # Circuit breaker para volatilidade
        volatility_config = CircuitBreakerConfig(
            name="Volatilidade Global",
            cooldown_period_minutes=120,
            reset_threshold=3,
            consecutive_success_to_close=6,
            failure_threshold=3,
            recovery_timeout_minutes=240,
            auto_reset=False,
            requires_manual_reset_if_critical=True,
            metrics_to_monitor={MetricType.VOLATILITY}
        )
        
        # Registrar circuit breakers
        self.circuit_breaker_service.register_circuit_breaker("global", drawdown_config)
        self.circuit_breaker_service.register_circuit_breaker("global", var_config)
        self.circuit_breaker_service.register_circuit_breaker("global", volatility_config)
        
        self.logger.info("Circuit breakers padrão registrados com sucesso")
    
    def configure_default_exposure_limits(self) -> None:
        """Configura limites de exposição padrão para categorias comuns."""
        # Limite para criptomoedas de alta capitalização
        self.exposure_service.set_category_limit(
            "crypto_large_cap",
            ExposureLimit(
                max_percentage=30.0,
                max_absolute=None,
                warning_threshold_pct=0.8,
                include_unrealized_pnl=True,
                cooldown_minutes=30
            )
        )
        
        # Limite para criptomoedas de média capitalização
        self.exposure_service.set_category_limit(
            "crypto_mid_cap",
            ExposureLimit(
                max_percentage=20.0,
                max_absolute=None,
                warning_threshold_pct=0.8,
                include_unrealized_pnl=True,
                cooldown_minutes=30
            )
        )
        
        # Limite para criptomoedas de baixa capitalização
        self.exposure_service.set_category_limit(
            "crypto_small_cap",
            ExposureLimit(
                max_percentage=10.0,
                max_absolute=None,
                warning_threshold_pct=0.8,
                include_unrealized_pnl=True,
                cooldown_minutes=15
            )
        )
        
        # Limite para stablecoins
        self.exposure_service.set_category_limit(
            "stablecoin",
            ExposureLimit(
                max_percentage=50.0,
                max_absolute=None,
                warning_threshold_pct=0.9,
                include_unrealized_pnl=False,
                cooldown_minutes=60
            )
        )
        
        # Limite para tokens DeFi
        self.exposure_service.set_category_limit(
            "defi",
            ExposureLimit(
                max_percentage=15.0,
                max_absolute=None,
                warning_threshold_pct=0.8,
                include_unrealized_pnl=True,
                cooldown_minutes=20
            )
        )
        
        self.logger.info("Limites de exposição padrão configurados com sucesso")
    
    def validate_order(self, params: OrderRiskParams) -> Dict:
        """
        Valida uma ordem quanto aos limites de risco.
        
        Args:
            params: Parâmetros da ordem
            
        Returns:
            Resultado da validação
        """
        result = self.validate_order_risk_use_case.execute(params)
        
        return {
            "is_valid": result.is_valid,
            "validation_id": result.validation_id,
            "reasons": result.reasons,
            "warnings": result.warnings,
            "metrics": result.metrics
        }
    
    def generate_risk_report(self, request: RiskReportRequest) -> Dict:
        """
        Gera um relatório completo de risco.
        
        Args:
            request: Solicitação de relatório
            
        Returns:
            Relatório de risco formatado como dicionário
        """
        report = self.generate_risk_report_use_case.execute(request)
        
        # Converter para formato JSON-serializável
        return {
            "report_id": report.report_id,
            "portfolio_id": report.portfolio_id,
            "generation_time": report.generation_time.isoformat(),
            "timeframe": report.timeframe.value,
            "start_date": report.start_date.isoformat(),
            "end_date": report.end_date.isoformat(),
            "portfolio_stats": {
                "total_value": report.portfolio_stats.total_value,
                "unrealized_pnl": report.portfolio_stats.unrealized_pnl,
                "unrealized_pnl_pct": report.portfolio_stats.unrealized_pnl_pct,
                "num_positions": report.portfolio_stats.num_positions,
                "max_position_value": report.portfolio_stats.max_position_value,
                "max_position_weight": report.portfolio_stats.max_position_weight,
                "max_position_asset": report.portfolio_stats.max_position_asset,
                "concentration_index": report.portfolio_stats.concentration_index,
                "correlation_avg": report.portfolio_stats.correlation_avg,
                "volatility_daily": report.portfolio_stats.volatility_daily,
                "sharpe_ratio": report.portfolio_stats.sharpe_ratio,
                "sortino_ratio": report.portfolio_stats.sortino_ratio,
                "var_95": report.portfolio_stats.var_95,
                "var_99": report.portfolio_stats.var_99,
                "expected_shortfall_95": report.portfolio_stats.expected_shortfall_95,
                "max_drawdown": report.portfolio_stats.max_drawdown,
                "drawdown_current": report.portfolio_stats.drawdown_current,
                "drawdown_period": report.portfolio_stats.drawdown_period
            },
            "metrics_summary": [
                {
                    "name": m.name,
                    "current_value": m.current_value,
                    "average_value": m.average_value,
                    "min_value": m.min_value,
                    "max_value": m.max_value,
                    "trend": m.trend,
                    "status": m.status,
                    "threshold": m.threshold
                }
                for m in report.metrics_summary
            ],
            "risk_breaches": report.risk_breaches,
            "circuit_breaker_events": report.circuit_breaker_events,
            "exposure_breakdown": report.exposure_breakdown,
            "asset_risk_contribution": report.asset_risk_contribution,
            "correlation_matrix": report.correlation_matrix,
            "recommendations": report.recommendations
        }
    
    def process_metric(self, metric: RiskMetric) -> List[RiskThresholdBreachedEvent]:
        """
        Processa uma métrica de risco e verifica limites.
        
        Args:
            metric: Métrica de risco a ser processada
            
        Returns:
            Lista de eventos de violação de limite gerados
        """
        # Salvar métrica no repositório
        self.risk_repository.save_metric(metric)
        
        # Obter perfis de risco aplicáveis
        profiles = self.risk_repository.get_applicable_profiles(
            portfolio_id=metric.portfolio_id,
            strategy_id=metric.strategy_id,
            asset_id=metric.asset_id
        )
        
        # Verificar violações de limite
        breach_events = []
        
        for profile in profiles:
            # Verificar se o perfil tem limite para este tipo de métrica
            if metric.type not in profile.thresholds:
                continue
            
            # Verificar violação
            warning_breached, critical_breached = profile.check_threshold_breach(
                metric.type, metric.value
            )
            
            # Criar evento se houver violação
            if warning_breached or critical_breached:
                threshold_config = profile.thresholds[metric.type]
                threshold_value = threshold_config.critical_threshold if critical_breached else threshold_config.warning_threshold
                
                event = RiskThresholdBreachedEvent.create(
                    metric=metric,
                    profile=profile,
                    threshold_value=threshold_value,
                    is_critical=critical_breached,
                    risk_level=profile.risk_level,
                    portfolio_id=metric.portfolio_id,
                    strategy_id=metric.strategy_id,
                    asset_id=metric.asset_id
                )
                
                # Salvar evento
                self.risk_repository.save_threshold_breach(event)
                breach_events.append(event)
                
                # Enviar alerta
                self.alert_notifier.send_threshold_breach_alert(event)
                
                # Processar no circuit breaker
                self.circuit_breaker_service.process_threshold_event(event)
        
        return breach_events
    
    def update_portfolio_data(
        self,
        portfolio_id: str,
        positions: List[Dict],
        prices: Dict[str, float],
        returns: Dict[str, List[float]]
    ) -> List[RiskMetric]:
        """
        Atualiza dados do portfólio e calcula métricas de risco.
        
        Args:
            portfolio_id: ID do portfólio
            positions: Lista de posições atuais
            prices: Dicionário de preços atuais por ativo
            returns: Dicionário de retornos históricos por ativo
            
        Returns:
            Lista de métricas de risco geradas
        """
        # Converter posições para o formato esperado
        portfolio_positions = []
        total_value = 0
        
        for pos in positions:
            position_value = pos["quantity"] * prices.get(pos["asset_id"], 0)
            total_value += position_value
            
            portfolio_positions.append(PortfolioPosition(
                asset_id=pos["asset_id"],
                quantity=pos["quantity"],
                current_price=prices.get(pos["asset_id"], 0),
                avg_entry_price=pos.get("avg_entry_price", 0),
                unrealized_pnl=pos.get("unrealized_pnl", 0),
                position_value=position_value,
                weight=0,  # Será calculado depois
                metadata=pos.get("metadata", {})
            ))
        
        # Calcular pesos
        if total_value > 0:
            for pos in portfolio_positions:
                pos.weight = (pos.position_value / total_value) * 100
        
        # Atualizar capital total no serviço de exposição
        self.exposure_service.set_total_capital(total_value)
        
        # Atualizar dados no serviço de análise de portfólio
        self.portfolio_analytics_service.update_portfolio_data(
            positions=portfolio_positions,
            prices=prices,
            returns=returns
        )
        
        # Obter perfil de risco do portfólio
        profile = self.risk_repository.get_risk_profile_by_portfolio(portfolio_id)
        if not profile:
            self.logger.warning(f"Perfil de risco não encontrado para portfólio {portfolio_id}")
            # Usar perfil conservador como padrão
            profile = RiskProfile.create_conservative(
                name=f"Perfil Temporário para {portfolio_id}",
                description="Perfil conservador gerado automaticamente",
                portfolio_id=portfolio_id
            )
            self.risk_repository.save_risk_profile(profile)
        
        # Gerar métricas de risco
        metrics = self.portfolio_analytics_service.generate_risk_metrics(portfolio_id, profile)
        
        # Processar cada métrica para verificar limites
        all_breach_events = []
        for metric in metrics:
            breach_events = self.process_metric(metric)
            all_breach_events.extend(breach_events)
        
        # Atualizar exposição para cada posição
        exposure_metrics = []
        for pos in portfolio_positions:
            # Registrar categorias para o ativo se disponíveis
            if "categories" in pos.metadata:
                self.exposure_service.register_asset_category(pos.asset_id, pos.metadata["categories"])
            
            # Atualizar exposição
            metrics = self.exposure_service.update_exposure(pos.asset_id, pos.position_value)
            exposure_metrics.extend(metrics)
        
        # Verificar limites de exposição
        exposure_violations = self.exposure_service.check_exposure_limits(profile)
        
        # Enviar alertas para violações de exposição
        for entity_id, info, is_critical in exposure_violations:
            entity_type = info.metadata.get("type", "unknown")
            
            self.alert_notifier.send_exposure_alert(
                entity_id=entity_id,
                entity_type=entity_type,
                current_value=info.current_value,
                max_allowed=info.max_allowed,
                percentage_used=info.percentage_used,
                is_critical=is_critical
            )
        
        # Salvar métricas de exposição
        for metric in exposure_metrics:
            self.risk_repository.save_metric(metric)
            # Não processamos limites aqui porque já fizemos isso no check_exposure_limits
        
        # Verificar auto-reset de circuit breakers
        self.circuit_breaker_service.check_auto_reset()
        
        # Retornar todas as métricas geradas
        return metrics + exposure_metrics
    
    def check_position_against_risk_limits(
        self,
        asset_id: str,
        quantity: float,
        price: float,
        portfolio_id: Optional[str] = None
    ) -> Dict:
        """
        Verifica se uma posição está dentro dos limites de risco.
        
        Args:
            asset_id: ID do ativo
            quantity: Quantidade da posição
            price: Preço atual do ativo
            portfolio_id: ID do portfólio (opcional)
            
        Returns:
            Resultado da verificação
        """
        position_value = quantity * price
        
        # Verificar limites de exposição
        can_increase, reason = self.exposure_service.can_increase_position(asset_id, position_value)
        
        # Verificar circuit breakers
        circuit_breaker_active = False
        circuit_breaker_state = None
        
        if self.circuit_breaker_service.is_open(asset_id):
            circuit_breaker_active = True
            circuit_breaker_state = "open"
        elif self.circuit_breaker_service.is_open("global"):
            circuit_breaker_active = True
            circuit_breaker_state = "open"
        elif portfolio_id and self.circuit_breaker_service.is_open(portfolio_id):
            circuit_breaker_active = True
            circuit_breaker_state = "open"
        
        # Calcular percentual do capital
        total_capital = self.exposure_service.get_total_capital()
        position_pct = (position_value / total_capital) * 100 if total_capital > 0 else 0
        
        # Verificar perfis de risco aplicáveis
        profiles = self.risk_repository.get_applicable_profiles(
            portfolio_id=portfolio_id,
            asset_id=asset_id
        )
        
        profile_violations = []
        for profile in profiles:
            if position_pct > profile.max_position_size_pct:
                profile_violations.append(
                    f"Excede limite de tamanho de posição do perfil '{profile.name}': "
                    f"{position_pct:.2f}% > {profile.max_position_size_pct:.2f}%"
                )
        
        # Montar resultado
        result = {
            "is_valid": can_increase and not circuit_breaker_active and not profile_violations,
            "position_value": position_value,
            "position_percentage": position_pct,
            "exposure_limit_violated": not can_increase,
            "exposure_reason": reason if not can_increase else "",
            "circuit_breaker_active": circuit_breaker_active,
            "circuit_breaker_state": circuit_breaker_state,
            "profile_violations": profile_violations
        }
        
        return result
    
    def get_risk_summary(self, portfolio_id: str) -> Dict:
        """
        Obtém um resumo do estado atual de risco.
        
        Args:
            portfolio_id: ID do portfólio
            
        Returns:
            Resumo do estado de risco
        """
        # Obter métricas mais recentes
        var_metric = self.risk_repository.get_latest_metric(
            MetricType.VALUE_AT_RISK, portfolio_id=portfolio_id
        )
        
        es_metric = self.risk_repository.get_latest_metric(
            MetricType.EXPECTED_SHORTFALL, portfolio_id=portfolio_id
        )
        
        drawdown_metric = self.risk_repository.get_latest_metric(
            MetricType.DRAWDOWN, portfolio_id=portfolio_id
        )
        
        volatility_metric = self.risk_repository.get_latest_metric(
            MetricType.VOLATILITY, portfolio_id=portfolio_id
        )
        
        sharpe_metric = self.risk_repository.get_latest_metric(
            MetricType.SHARPE_RATIO, portfolio_id=portfolio_id
        )
        
        # Obter estado dos circuit breakers
        circuit_breakers = self.circuit_breaker_service.get_all_circuit_breakers()
        
        # Filtrar circuit breakers relevantes
        relevant_cbs = {}
        for entity_id, cb_info in circuit_breakers.items():
            if entity_id == "global" or entity_id == portfolio_id:
                relevant_cbs[entity_id] = cb_info
        
        # Obter violações recentes
        recent_breaches = self.risk_repository.get_threshold_breaches(
            start_date=datetime.utcnow() - timedelta(days=1),
            portfolio_id=portfolio_id,
            limit=10
        )
        
        breach_summaries = []
        for breach in recent_breaches:
            breach_summaries.append({
                "timestamp": breach.timestamp.isoformat(),
                "metric_type": breach.metric.type.name,
                "value": breach.actual_value,
                "threshold": breach.threshold_value,
                "is_critical": breach.is_critical,
                "message": breach.get_message()
            })
        
        # Obter métricas de exposição
        exposure_metrics = self.exposure_service.get_exposure_metrics()
        
        # Montar resumo
        summary = {
            "portfolio_id": portfolio_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {
                "var_95": var_metric.value if var_metric else None,
                "expected_shortfall_95": es_metric.value if es_metric else None,
                "current_drawdown": drawdown_metric.value if drawdown_metric else None,
                "volatility_daily": volatility_metric.value if volatility_metric else None,
                "sharpe_ratio": sharpe_metric.value if sharpe_metric else None
            },
            "circuit_breakers": relevant_cbs,
            "recent_breaches": breach_summaries,
            "exposure": {
                "total_capital": exposure_metrics.get("total_capital", 0),
                "total_exposure": exposure_metrics.get("summary", {}).get("total_exposure", 0),
                "exposure_percentage": exposure_metrics.get("summary", {}).get("exposure_percentage", 0),
                "asset_count": len(exposure_metrics.get("assets", {})),
                "category_count": len(exposure_metrics.get("categories", {}))
            },
            "status": "normal"
        }
        
        # Determinar status geral
        if any(cb.get("state") == "open" for cb in relevant_cbs.values()):
            summary["status"] = "critical"
        elif any(breach.is_critical for breach in recent_breaches):
            summary["status"] = "warning"
        
        return summary


def initialize_risk_manager() -> RiskManager:
    """
    Inicializa e retorna uma instância configurada do gerenciador de risco.
    
    Returns:
        Instância do gerenciador de risco
    """
    # Criar gerenciador
    risk_manager = RiskManager()
    
    # Criar configurações padrão
    risk_manager.create_default_risk_profiles()
    risk_manager.register_default_circuit_breakers()
    risk_manager.configure_default_exposure_limits()
    
    return risk_manager


# Singleton para uso em toda a aplicação
risk_manager = initialize_risk_manager()