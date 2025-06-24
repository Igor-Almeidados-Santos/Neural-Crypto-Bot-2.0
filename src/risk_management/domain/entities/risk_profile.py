from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from risk_management.domain.entities.risk_metric import MetricType


class RiskLevel(Enum):
    """Níveis de risco para perfis e alertas."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(Enum):
    """Categorias de risco gerenciadas pelo sistema."""
    MARKET = "market"
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    OPERATIONAL = "operational"
    CONCENTRATION = "concentration"
    VOLATILITY = "volatility"
    DRAWDOWN = "drawdown"
    EXECUTION = "execution"


@dataclass
class ThresholdConfig:
    """Configuração de limites para uma métrica específica."""
    metric_type: MetricType
    warning_threshold: float
    critical_threshold: float
    is_upper_bound: bool = True  # Se True, alerta quando acima do limite; se False, quando abaixo
    
    def is_warning_breached(self, value: float) -> bool:
        """Verifica se o valor ultrapassa o limite de alerta."""
        return value > self.warning_threshold if self.is_upper_bound else value < self.warning_threshold
    
    def is_critical_breached(self, value: float) -> bool:
        """Verifica se o valor ultrapassa o limite crítico."""
        return value > self.critical_threshold if self.is_upper_bound else value < self.critical_threshold


@dataclass
class RiskProfile:
    """
    Entidade que define o perfil de risco com limites e configurações.
    Pode ser aplicado a um portfólio, estratégia ou ativo específico.
    """
    id: UUID
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    risk_level: RiskLevel
    max_drawdown_pct: float
    max_position_size_pct: float
    max_leverage: float
    var_confidence_level: float = 0.95
    thresholds: Dict[MetricType, ThresholdConfig] = field(default_factory=dict)
    correlation_limits: Dict[str, float] = field(default_factory=dict)
    position_limits: Dict[str, float] = field(default_factory=dict)
    risk_categories: List[RiskCategory] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    asset_id: Optional[str] = None

    @classmethod
    def create_conservative(
        cls,
        name: str,
        description: str,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> "RiskProfile":
        """
        Factory method para criar um perfil de risco conservador.
        
        Args:
            name: Nome do perfil
            description: Descrição do perfil
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            Novo perfil de risco conservador
        """
        now = datetime.utcnow()
        profile = cls(
            id=uuid4(),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            risk_level=RiskLevel.LOW,
            max_drawdown_pct=5.0,
            max_position_size_pct=2.0,
            max_leverage=1.0,
            var_confidence_level=0.99,
            portfolio_id=portfolio_id,
            strategy_id=strategy_id,
            asset_id=asset_id,
            risk_categories=[
                RiskCategory.MARKET,
                RiskCategory.LIQUIDITY,
                RiskCategory.DRAWDOWN,
            ],
        )
        
        # Configurar limites padrão para perfil conservador
        profile.thresholds = {
            MetricType.VALUE_AT_RISK: ThresholdConfig(
                metric_type=MetricType.VALUE_AT_RISK,
                warning_threshold=3.0,
                critical_threshold=5.0,
                is_upper_bound=True,
            ),
            MetricType.EXPECTED_SHORTFALL: ThresholdConfig(
                metric_type=MetricType.EXPECTED_SHORTFALL,
                warning_threshold=4.0,
                critical_threshold=6.0,
                is_upper_bound=True,
            ),
            MetricType.DRAWDOWN: ThresholdConfig(
                metric_type=MetricType.DRAWDOWN,
                warning_threshold=3.0,
                critical_threshold=5.0,
                is_upper_bound=True,
            ),
            MetricType.VOLATILITY: ThresholdConfig(
                metric_type=MetricType.VOLATILITY,
                warning_threshold=15.0,
                critical_threshold=20.0,
                is_upper_bound=True,
            ),
            MetricType.SHARPE_RATIO: ThresholdConfig(
                metric_type=MetricType.SHARPE_RATIO,
                warning_threshold=1.0,
                critical_threshold=0.5,
                is_upper_bound=False,
            ),
        }
        
        return profile

    @classmethod
    def create_moderate(
        cls,
        name: str,
        description: str,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> "RiskProfile":
        """
        Factory method para criar um perfil de risco moderado.
        
        Args:
            name: Nome do perfil
            description: Descrição do perfil
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            Novo perfil de risco moderado
        """
        now = datetime.utcnow()
        profile = cls(
            id=uuid4(),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            risk_level=RiskLevel.MEDIUM,
            max_drawdown_pct=15.0,
            max_position_size_pct=5.0,
            max_leverage=2.0,
            var_confidence_level=0.95,
            portfolio_id=portfolio_id,
            strategy_id=strategy_id,
            asset_id=asset_id,
            risk_categories=[
                RiskCategory.MARKET,
                RiskCategory.LIQUIDITY,
                RiskCategory.DRAWDOWN,
                RiskCategory.VOLATILITY,
            ],
        )
        
        # Configurar limites padrão para perfil moderado
        profile.thresholds = {
            MetricType.VALUE_AT_RISK: ThresholdConfig(
                metric_type=MetricType.VALUE_AT_RISK,
                warning_threshold=8.0,
                critical_threshold=12.0,
                is_upper_bound=True,
            ),
            MetricType.EXPECTED_SHORTFALL: ThresholdConfig(
                metric_type=MetricType.EXPECTED_SHORTFALL,
                warning_threshold=10.0,
                critical_threshold=15.0,
                is_upper_bound=True,
            ),
            MetricType.DRAWDOWN: ThresholdConfig(
                metric_type=MetricType.DRAWDOWN,
                warning_threshold=10.0,
                critical_threshold=15.0,
                is_upper_bound=True,
            ),
            MetricType.VOLATILITY: ThresholdConfig(
                metric_type=MetricType.VOLATILITY,
                warning_threshold=25.0,
                critical_threshold=35.0,
                is_upper_bound=True,
            ),
            MetricType.SHARPE_RATIO: ThresholdConfig(
                metric_type=MetricType.SHARPE_RATIO,
                warning_threshold=0.7,
                critical_threshold=0.3,
                is_upper_bound=False,
            ),
        }
        
        return profile

    @classmethod
    def create_aggressive(
        cls,
        name: str,
        description: str,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None,
    ) -> "RiskProfile":
        """
        Factory method para criar um perfil de risco agressivo.
        
        Args:
            name: Nome do perfil
            description: Descrição do perfil
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            Novo perfil de risco agressivo
        """
        now = datetime.utcnow()
        profile = cls(
            id=uuid4(),
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
            risk_level=RiskLevel.HIGH,
            max_drawdown_pct=25.0,
            max_position_size_pct=10.0,
            max_leverage=3.0,
            var_confidence_level=0.90,
            portfolio_id=portfolio_id,
            strategy_id=strategy_id,
            asset_id=asset_id,
            risk_categories=[
                RiskCategory.MARKET,
                RiskCategory.LIQUIDITY,
                RiskCategory.VOLATILITY,
            ],
        )
        
        # Configurar limites padrão para perfil agressivo
        profile.thresholds = {
            MetricType.VALUE_AT_RISK: ThresholdConfig(
                metric_type=MetricType.VALUE_AT_RISK,
                warning_threshold=15.0,
                critical_threshold=20.0,
                is_upper_bound=True,
            ),
            MetricType.EXPECTED_SHORTFALL: ThresholdConfig(
                metric_type=MetricType.EXPECTED_SHORTFALL,
                warning_threshold=18.0,
                critical_threshold=25.0,
                is_upper_bound=True,
            ),
            MetricType.DRAWDOWN: ThresholdConfig(
                metric_type=MetricType.DRAWDOWN,
                warning_threshold=20.0,
                critical_threshold=25.0,
                is_upper_bound=True,
            ),
            MetricType.VOLATILITY: ThresholdConfig(
                metric_type=MetricType.VOLATILITY,
                warning_threshold=40.0,
                critical_threshold=50.0,
                is_upper_bound=True,
            ),
            MetricType.SHARPE_RATIO: ThresholdConfig(
                metric_type=MetricType.SHARPE_RATIO,
                warning_threshold=0.4,
                critical_threshold=0.1,
                is_upper_bound=False,
            ),
        }
        
        return profile

    def update_threshold(self, metric_type: MetricType, warning: float, critical: float, is_upper_bound: bool = True) -> None:
        """
        Atualiza ou adiciona uma configuração de limite para uma métrica específica.
        
        Args:
            metric_type: Tipo da métrica a ser configurada
            warning: Valor do limite de alerta
            critical: Valor do limite crítico
            is_upper_bound: Se True, alerta quando acima do limite; se False, quando abaixo
        """
        self.thresholds[metric_type] = ThresholdConfig(
            metric_type=metric_type,
            warning_threshold=warning,
            critical_threshold=critical,
            is_upper_bound=is_upper_bound,
        )
        self.updated_at = datetime.utcnow()

    def check_threshold_breach(self, metric_type: MetricType, value: float) -> Tuple[bool, bool]:
        """
        Verifica se um valor viola os limites de alerta ou crítico para uma métrica.
        
        Args:
            metric_type: Tipo da métrica a ser verificada
            value: Valor atual da métrica
            
        Returns:
            Tupla com (breach_warning, breach_critical)
        """
        if metric_type not in self.thresholds:
            return False, False
        
        threshold = self.thresholds[metric_type]
        return threshold.is_warning_breached(value), threshold.is_critical_breached(value)
    
    def is_applicable_to(self, portfolio_id: Optional[str] = None, 
                        strategy_id: Optional[str] = None, 
                        asset_id: Optional[str] = None) -> bool:
        """
        Verifica se este perfil de risco é aplicável a um portfólio, estratégia ou ativo específico.
        
        Args:
            portfolio_id: ID do portfólio a verificar
            strategy_id: ID da estratégia a verificar
            asset_id: ID do ativo a verificar
            
        Returns:
            True se o perfil é aplicável, False caso contrário
        """
        if not self.enabled:
            return False
            
        if self.portfolio_id and portfolio_id and self.portfolio_id == portfolio_id:
            return True
            
        if self.strategy_id and strategy_id and self.strategy_id == strategy_id:
            return True
            
        if self.asset_id and asset_id and self.asset_id == asset_id:
            return True
            
        # Se não tiver nenhum ID específico, é um perfil global
        if not self.portfolio_id and not self.strategy_id and not self.asset_id:
            return True
            
        return False