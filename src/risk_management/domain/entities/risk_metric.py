from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional
from uuid import UUID, uuid4


class MetricType(Enum):
    """Tipos de métricas de risco suportadas pelo sistema."""
    VALUE_AT_RISK = auto()
    EXPECTED_SHORTFALL = auto()
    DRAWDOWN = auto()
    VOLATILITY = auto()
    SHARPE_RATIO = auto()
    MAX_EXPOSURE = auto()
    CORRELATION = auto()
    LIQUIDITY = auto()
    CONCENTRATION = auto()
    CUSTOM = auto()


class TimeFrame(Enum):
    """Períodos temporais para métricas de risco."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass(frozen=True)
class RiskMetric:
    """
    Entidade de domínio que representa uma métrica de risco específica.
    Imutável para garantir consistência e rastreabilidade.
    """
    id: UUID
    type: MetricType
    value: float
    timestamp: datetime
    asset_id: Optional[str] = None
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    timeframe: Optional[TimeFrame] = None
    confidence_level: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        metric_type: MetricType,
        value: float,
        asset_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        portfolio_id: Optional[str] = None,
        timeframe: Optional[TimeFrame] = None,
        confidence_level: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "RiskMetric":
        """
        Factory method para criar uma nova métrica de risco com timestamp atual e ID gerado.
        
        Args:
            metric_type: Tipo da métrica de risco
            value: Valor da métrica
            asset_id: ID do ativo relacionado (opcional)
            strategy_id: ID da estratégia relacionada (opcional)
            portfolio_id: ID do portfólio relacionado (opcional)
            timeframe: Período de tempo da métrica (opcional)
            confidence_level: Nível de confiança para métricas estatísticas (opcional)
            metadata: Dados adicionais específicos da métrica (opcional)
            
        Returns:
            Nova instância de RiskMetric
        """
        return cls(
            id=uuid4(),
            type=metric_type,
            value=value,
            timestamp=datetime.utcnow(),
            asset_id=asset_id,
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            timeframe=timeframe,
            confidence_level=confidence_level,
            metadata=metadata or {},
        )

    def is_portfolio_level(self) -> bool:
        """Verifica se a métrica é no nível de portfólio."""
        return self.portfolio_id is not None

    def is_strategy_level(self) -> bool:
        """Verifica se a métrica é no nível de estratégia."""
        return self.strategy_id is not None

    def is_asset_level(self) -> bool:
        """Verifica se a métrica é no nível de ativo."""
        return self.asset_id is not None

    def exceeds_threshold(self, threshold: float) -> bool:
        """
        Verifica se o valor da métrica excede um limite definido.
        Para métricas como Sharpe Ratio, onde valores maiores são melhores,
        este método deve ser substituído nas subclasses especializadas.
        
        Args:
            threshold: Valor limite para comparação
            
        Returns:
            True se o valor exceder o limite, False caso contrário
        """
        return self.value > threshold