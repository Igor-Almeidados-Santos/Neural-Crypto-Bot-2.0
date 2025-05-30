"""
Value objects para parâmetros de execução.

Este módulo define os value objects que contêm os parâmetros
para cada algoritmo de execução de ordens.
"""
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ExecutionParameters(BaseModel):
    """
    Classe base para parâmetros de execução.
    """
    pass


class TwapParameters(ExecutionParameters):
    """
    Parâmetros para o algoritmo TWAP (Time-Weighted Average Price).
    
    O TWAP divide uma ordem em partes iguais distribuídas ao longo
    de um período de tempo especificado.
    """
    
    duration_minutes: int = Field(..., gt=0)
    num_slices: int = Field(..., gt=0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation_rate: Optional[float] = None
    
    @validator('duration_minutes')
    def validate_duration(cls, v):
        """Valida que a duração está dentro de limites razoáveis."""
        if v < 1:
            raise ValueError("Duração deve ser de pelo menos 1 minuto")
        if v > 1440:  # 24 horas
            raise ValueError("Duração não deve exceder 24 horas (1440 minutos)")
        return v
    
    @validator('num_slices')
    def validate_slices(cls, v):
        """Valida que o número de fatias está dentro de limites razoáveis."""
        if v < 1:
            raise ValueError("Número de fatias deve ser pelo menos 1")
        if v > 1000:
            raise ValueError("Número de fatias não deve exceder 1000")
        return v
    
    @validator('max_participation_rate')
    def validate_participation_rate(cls, v):
        """Valida que a taxa de participação está entre 0 e 1."""
        if v is not None and (v <= 0 or v > 1):
            raise ValueError("Taxa de participação deve estar entre 0 e 1")
        return v


class IcebergParameters(ExecutionParameters):
    """
    Parâmetros para o algoritmo Iceberg.
    
    O Iceberg divide uma ordem em partes menores que são submetidas
    sequencialmente, ocultando o tamanho total da ordem.
    """
    
    display_size: float = Field(..., gt=0)
    size_variance: float = Field(0.0, ge=0, le=0.5)
    interval_seconds: float = Field(0.0, ge=0)
    interval_variance: float = Field(0.0, ge=0, le=0.5)
    price_adjustment_threshold: float = Field(0.0, ge=0)
    continue_on_failure: bool = Field(False)
    
    @validator('display_size')
    def validate_display_size(cls, v):
        """Valida que o tamanho de exibição é positivo."""
        if v <= 0:
            raise ValueError("Tamanho de exibição deve ser positivo")
        return v
    
    @validator('size_variance')
    def validate_size_variance(cls, v):
        """Valida que a variação de tamanho está entre 0 e 0.5 (50%)."""
        if v < 0 or v > 0.5:
            raise ValueError("Variação de tamanho deve estar entre 0 e 0.5 (50%)")
        return v
    
    @validator('interval_seconds')
    def validate_interval(cls, v):
        """Valida que o intervalo é não-negativo."""
        if v < 0:
            raise ValueError("Intervalo deve ser não-negativo")
        return v
    
    @validator('price_adjustment_threshold')
    def validate_price_threshold(cls, v):
        """Valida que o limite de ajuste de preço é não-negativo."""
        if v < 0:
            raise ValueError("Limite de ajuste de preço deve ser não-negativo")
        return v


class SmartRoutingParameters(ExecutionParameters):
    """
    Parâmetros para o algoritmo Smart Routing.
    
    O Smart Routing distribui uma ordem entre múltiplas exchanges
    para obter o melhor preço médio de execução.
    """
    
    exchanges: List[str] = Field(..., min_items=1)
    allocation_strategy: str = Field("balanced")
    execute_parallel: bool = Field(True)
    retry_failed: bool = Field(True)
    max_price_deviation: float = Field(0.02, ge=0)
    
    @validator('exchanges')
    def validate_exchanges(cls, v):
        """Valida que a lista de exchanges tem pelo menos uma exchange."""
        if not v:
            raise ValueError("Deve ser especificada pelo menos uma exchange")
        return v
    
    @validator('allocation_strategy')
    def validate_allocation_strategy(cls, v):
        """Valida que a estratégia de alocação é uma das suportadas."""
        valid_strategies = ["best_price", "proportional", "balanced"]
        if v not in valid_strategies:
            raise ValueError(f"Estratégia de alocação deve ser uma das: {valid_strategies}")
        return v
    
    @validator('max_price_deviation')
    def validate_price_deviation(cls, v):
        """Valida que o desvio máximo de preço é não-negativo."""
        if v < 0:
            raise ValueError("Desvio máximo de preço deve ser não-negativo")
        return v


class VwapParameters(ExecutionParameters):
    """
    Parâmetros para o algoritmo VWAP (Volume-Weighted Average Price).
    
    O VWAP divide uma ordem em partes distribuídas ao longo do tempo
    com base em perfis de volume históricos.
    """
    
    duration_minutes: int = Field(..., gt=0)
    num_slices: int = Field(..., gt=0)
    volume_profile: Optional[Dict[str, float]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    max_participation_rate: Optional[float] = None
    
    @validator('duration_minutes')
    def validate_duration(cls, v):
        """Valida que a duração está dentro de limites razoáveis."""
        if v < 1:
            raise ValueError("Duração deve ser de pelo menos 1 minuto")
        if v > 1440:  # 24 horas
            raise ValueError("Duração não deve exceder 24 horas (1440 minutos)")
        return v
    
    @validator('num_slices')
    def validate_slices(cls, v):
        """Valida que o número de fatias está dentro de limites razoáveis."""
        if v < 1:
            raise ValueError("Número de fatias deve ser pelo menos 1")
        if v > 1000:
            raise ValueError("Número de fatias não deve exceder 1000")
        return v
    
    @validator('volume_profile')
    def validate_volume_profile(cls, v):
        """Valida que o perfil de volume é válido."""
        if v is not None:
            # Verificar se as chaves são timestamps válidos e os valores somam 1
            total = sum(v.values())
            if not 0.99 <= total <= 1.01:  # permitir pequeno erro de arredondamento
                raise ValueError("A soma dos valores do perfil de volume deve ser aproximadamente 1")
        return v


class DirectParameters(ExecutionParameters):
    """
    Parâmetros para execução direta (sem algoritmo especial).
    
    A execução direta envia a ordem diretamente para a exchange,
    sem aplicar nenhum algoritmo especial.
    """
    
    retry_on_failure: bool = Field(False)
    max_retries: int = Field(3, ge=0)
    retry_delay_seconds: float = Field(1.0, ge=0)
    validate_market_price: bool = Field(False)
    max_price_deviation: float = Field(0.05, ge=0)
    
    @validator('max_retries')
    def validate_max_retries(cls, v):
        """Valida que o número máximo de tentativas é não-negativo."""
        if v < 0:
            raise ValueError("Número máximo de tentativas deve ser não-negativo")
        return v
    
    @validator('retry_delay_seconds')
    def validate_retry_delay(cls, v):
        """Valida que o atraso entre tentativas é não-negativo."""
        if v < 0:
            raise ValueError("Atraso entre tentativas deve ser não-negativo")
        return v
    
    @validator('max_price_deviation')
    def validate_price_deviation(cls, v):
        """Valida que o desvio máximo de preço é não-negativo."""
        if v < 0:
            raise ValueError("Desvio máximo de preço deve ser não-negativo")
        return v


def create_execution_parameters(
    algorithm: str, params_dict: Dict
) -> ExecutionParameters:
    """
    Cria um objeto de parâmetros de execução com base no algoritmo especificado.
    
    Args:
        algorithm: Nome do algoritmo de execução.
        params_dict: Dicionário com os parâmetros.
        
    Returns:
        ExecutionParameters: Objeto de parâmetros do tipo apropriado.
        
    Raises:
        ValueError: Se o algoritmo não for suportado.
    """
    if algorithm == "twap":
        return TwapParameters(**params_dict)
    elif algorithm == "iceberg":
        return IcebergParameters(**params_dict)
    elif algorithm == "smart_routing":
        return SmartRoutingParameters(**params_dict)
    elif algorithm == "vwap":
        return VwapParameters(**params_dict)
    elif algorithm == "direct":
        return DirectParameters(**params_dict)
    else:
        raise ValueError(f"Algoritmo não suportado: {algorithm}")