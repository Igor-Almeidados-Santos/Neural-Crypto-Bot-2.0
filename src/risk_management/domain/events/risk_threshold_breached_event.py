from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from uuid import UUID, uuid4

from src.risk_management.domain.entities.risk_metric import MetricType, RiskMetric
from src.risk_management.domain.entities.risk_profile import RiskLevel, RiskProfile


@dataclass
class RiskThresholdBreachedEvent:
    """
    Evento de domínio que representa a violação de um limite de risco.
    Este evento é imutável e pode ser usado para notificações e logging.
    """
    id: UUID
    timestamp: datetime
    metric: RiskMetric
    profile: RiskProfile
    threshold_value: float
    actual_value: float
    risk_level: RiskLevel
    is_critical: bool
    portfolio_id: Optional[str] = None
    strategy_id: Optional[str] = None
    asset_id: Optional[str] = None
    metadata: Dict[str, str] = None

    @classmethod
    def create(
        cls,
        metric: RiskMetric,
        profile: RiskProfile,
        threshold_value: float,
        is_critical: bool,
        risk_level: RiskLevel,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "RiskThresholdBreachedEvent":
        """
        Factory method para criar um novo evento de violação de limite.
        
        Args:
            metric: A métrica de risco que violou o limite
            profile: O perfil de risco que define o limite
            threshold_value: O valor do limite que foi violado
            is_critical: Se é uma violação crítica ou apenas um alerta
            risk_level: Nível de risco desta violação
            portfolio_id: ID do portfólio afetado (opcional)
            strategy_id: ID da estratégia afetada (opcional)
            asset_id: ID do ativo afetado (opcional)
            metadata: Dados adicionais sobre a violação (opcional)
            
        Returns:
            Novo evento de violação de limite
        """
        return cls(
            id=uuid4(),
            timestamp=datetime.utcnow(),
            metric=metric,
            profile=profile,
            threshold_value=threshold_value,
            actual_value=metric.value,
            risk_level=risk_level,
            is_critical=is_critical,
            portfolio_id=portfolio_id or metric.portfolio_id,
            strategy_id=strategy_id or metric.strategy_id,
            asset_id=asset_id or metric.asset_id,
            metadata=metadata or {},
        )
        
    def get_message(self) -> str:
        """
        Gera uma mensagem descritiva sobre o evento de violação.
        
        Returns:
            String com descrição formatada do evento
        """
        entity_type = "portfólio" if self.portfolio_id else "estratégia" if self.strategy_id else "ativo"
        entity_id = self.portfolio_id or self.strategy_id or self.asset_id or "desconhecido"
        severity = "CRÍTICO" if self.is_critical else "ALERTA"
        
        return (
            f"{severity}: Limite de {self.metric.type.name} excedido para {entity_type} {entity_id}. "
            f"Valor atual: {self.actual_value:.2f}, Limite: {self.threshold_value:.2f}. "
            f"Nível de risco: {self.risk_level.value.upper()}."
        )
    
    def requires_immediate_action(self) -> bool:
        """
        Determina se este evento requer ação imediata.
        
        Returns:
            True se o evento for crítico ou de alto risco
        """
        return self.is_critical or self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)