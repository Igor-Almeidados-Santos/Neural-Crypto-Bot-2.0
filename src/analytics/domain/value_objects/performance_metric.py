"""
Value object para métricas de desempenho de trading.

Este módulo define o ValueObject PerformanceMetric que encapsula
diferentes métricas de desempenho de estratégias de trading.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, Dict, Any, Union

from common.domain.value_objects.base_value_object import BaseValueObject


class MetricType(Enum):
    """Tipos de métricas de desempenho."""
    TOTAL_RETURN = auto()
    ANNUALIZED_RETURN = auto()
    MAX_DRAWDOWN = auto()
    SHARPE_RATIO = auto()
    SORTINO_RATIO = auto()
    CALMAR_RATIO = auto()
    VOLATILITY = auto()
    ALPHA = auto()
    BETA = auto()
    INFORMATION_RATIO = auto()
    WIN_RATE = auto()
    PROFIT_FACTOR = auto()
    EXPECTANCY = auto()
    AVERAGE_WIN = auto()
    AVERAGE_LOSS = auto()
    RISK_REWARD_RATIO = auto()
    EXPOSURE_TIME = auto()
    MAX_CONSECUTIVE_WINS = auto()
    MAX_CONSECUTIVE_LOSSES = auto()
    RECOVERY_FACTOR = auto()
    CUSTOM = auto()


@dataclass(frozen=True)
class PerformanceMetric(BaseValueObject):
    """
    Value object que representa uma métrica de desempenho.
    
    Attributes:
        metric_type: Tipo da métrica
        value: Valor numérico da métrica
        timestamp: Timestamp de quando a métrica foi calculada
        time_range: Intervalo de tempo ao qual a métrica se refere (ex: "1d", "1w", "1m", "1y")
        start_date: Data de início do período de análise
        end_date: Data de fim do período de análise
        strategy_id: ID da estratégia (opcional)
        asset: Ativo relacionado (opcional)
        custom_name: Nome personalizado para métricas do tipo CUSTOM
        metadata: Metadados adicionais sobre a métrica
    """
    metric_type: MetricType
    value: Decimal
    timestamp: datetime
    time_range: str
    start_date: datetime
    end_date: datetime
    strategy_id: Optional[str] = None
    asset: Optional[str] = None
    custom_name: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validações adicionais após inicialização."""
        super().__post_init__()
        # Validar valor da métrica
        object.__setattr__(self, 'value', Decimal(str(self.value)))
        
        # Validar tipo de métrica CUSTOM requer custom_name
        if self.metric_type == MetricType.CUSTOM and not self.custom_name:
            raise ValueError("Métricas do tipo CUSTOM requerem um custom_name")
            
        # Garantir que o intervalo de datas seja válido
        if self.start_date > self.end_date:
            raise ValueError("Data de início deve ser anterior à data de fim")
            
        # Inicializar metadata como dicionário vazio se for None
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    @property
    def name(self) -> str:
        """Retorna o nome da métrica."""
        if self.metric_type == MetricType.CUSTOM:
            return self.custom_name or "Custom Metric"
        return self.metric_type.name.lower().replace('_', ' ').title()
    
    @property
    def formatted_value(self) -> str:
        """Retorna o valor formatado de acordo com o tipo de métrica."""
        if self.metric_type in (MetricType.TOTAL_RETURN, MetricType.ANNUALIZED_RETURN, 
                               MetricType.MAX_DRAWDOWN, MetricType.VOLATILITY,
                               MetricType.ALPHA, MetricType.BETA, MetricType.WIN_RATE,
                               MetricType.EXPOSURE_TIME):
            # Formatar como percentual
            return f"{float(self.value) * 100:.2f}%"
        elif self.metric_type in (MetricType.SHARPE_RATIO, MetricType.SORTINO_RATIO, 
                                 MetricType.CALMAR_RATIO, MetricType.INFORMATION_RATIO,
                                 MetricType.PROFIT_FACTOR, MetricType.EXPECTANCY,
                                 MetricType.RISK_REWARD_RATIO, MetricType.RECOVERY_FACTOR):
            # Formatar com 2 casas decimais
            return f"{float(self.value):.2f}"
        else:
            # Valor sem formatação específica
            return str(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a métrica para um dicionário."""
        return {
            "name": self.name,
            "type": self.metric_type.name,
            "value": float(self.value),
            "formatted_value": self.formatted_value,
            "timestamp": self.timestamp.isoformat(),
            "time_range": self.time_range,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "strategy_id": self.strategy_id,
            "asset": self.asset,
            "custom_name": self.custom_name,
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class PerformanceMetricCollection(BaseValueObject):
    """
    Value object que representa uma coleção de métricas de desempenho.
    
    Attributes:
        strategy_id: ID da estratégia
        timestamp: Timestamp de quando as métricas foram calculadas
        time_range: Intervalo de tempo ao qual as métricas se referem
        start_date: Data de início do período de análise
        end_date: Data de fim do período de análise
        metrics: Dicionário de métricas de desempenho
        metadata: Metadados adicionais sobre a coleção
    """
    strategy_id: str
    timestamp: datetime
    time_range: str
    start_date: datetime
    end_date: datetime
    metrics: Dict[MetricType, PerformanceMetric]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validações adicionais após inicialização."""
        super().__post_init__()
        
        # Garantir que o intervalo de datas seja válido
        if self.start_date > self.end_date:
            raise ValueError("Data de início deve ser anterior à data de fim")
            
        # Inicializar metadata como dicionário vazio se for None
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
    
    def get_metric(self, metric_type: MetricType) -> Optional[PerformanceMetric]:
        """Obtém uma métrica específica da coleção."""
        return self.metrics.get(metric_type)
    
    def get_value(self, metric_type: MetricType) -> Optional[Decimal]:
        """Obtém o valor de uma métrica específica."""
        metric = self.get_metric(metric_type)
        return metric.value if metric else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a coleção para um dicionário."""
        return {
            "strategy_id": self.strategy_id,
            "timestamp": self.timestamp.isoformat(),
            "time_range": self.time_range,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metrics": {k.name: v.to_dict() for k, v in self.metrics.items()},
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class PerformanceAnalysis(BaseValueObject):
    """
    Value object que representa uma análise completa de desempenho com
    métricas e dados adicionais.
    
    Attributes:
        strategy_id: ID da estratégia
        strategy_name: Nome da estratégia
        timestamp: Timestamp de quando a análise foi realizada
        time_range: Intervalo de tempo ao qual a análise se refere
        start_date: Data de início do período de análise
        end_date: Data de fim do período de análise
        metrics: Coleção de métricas de desempenho
        initial_capital: Capital inicial no período
        final_capital: Capital final no período
        trades_count: Número de trades no período
        winning_trades_count: Número de trades vencedores
        losing_trades_count: Número de trades perdedores
        additional_data: Dados adicionais da análise
    """
    strategy_id: str
    strategy_name: str
    timestamp: datetime
    time_range: str
    start_date: datetime
    end_date: datetime
    metrics: PerformanceMetricCollection
    initial_capital: Decimal
    final_capital: Decimal
    trades_count: int
    winning_trades_count: int
    losing_trades_count: int
    additional_data: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        """Validações adicionais após inicialização."""
        super().__post_init__()
        
        # Garantir que o intervalo de datas seja válido
        if self.start_date > self.end_date:
            raise ValueError("Data de início deve ser anterior à data de fim")
            
        # Validar valores numéricos
        object.__setattr__(self, 'initial_capital', Decimal(str(self.initial_capital)))
        object.__setattr__(self, 'final_capital', Decimal(str(self.final_capital)))
        
        # Inicializar additional_data como dicionário vazio se for None
        if self.additional_data is None:
            object.__setattr__(self, 'additional_data', {})
    
    @property
    def profit_loss(self) -> Decimal:
        """Retorna o lucro/prejuízo absoluto."""
        return self.final_capital - self.initial_capital
    
    @property
    def profit_loss_percent(self) -> Decimal:
        """Retorna o lucro/prejuízo percentual."""
        if self.initial_capital == 0:
            return Decimal('0')
        return (self.final_capital - self.initial_capital) / self.initial_capital * 100
    
    @property
    def win_rate(self) -> Decimal:
        """Retorna a taxa de acerto (win rate)."""
        if self.trades_count == 0:
            return Decimal('0')
        return Decimal(self.winning_trades_count) / Decimal(self.trades_count)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte a análise para um dicionário."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_name": self.strategy_name,
            "timestamp": self.timestamp.isoformat(),
            "time_range": self.time_range,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metrics": self.metrics.to_dict(),
            "initial_capital": float(self.initial_capital),
            "final_capital": float(self.final_capital),
            "profit_loss": float(self.profit_loss),
            "profit_loss_percent": float(self.profit_loss_percent),
            "trades_count": self.trades_count,
            "winning_trades_count": self.winning_trades_count,
            "losing_trades_count": self.losing_trades_count,
            "win_rate": float(self.win_rate),
            "additional_data": self.additional_data
        }