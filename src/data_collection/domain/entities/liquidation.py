"""
Entidade Liquidation para eventos de liquidação forçada.

Esta entidade representa eventos de liquidação de posições em contratos
perpétuos ou futuros, fundamentais para análise de risco de mercado
e detecção de movimentos extremos.
"""
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.common.domain.base_entity import BaseEntity
from src.common.domain.base_value_object import BaseValueObject


class LiquidationSide(Enum):
    """Lado da liquidação (long ou short)."""
    LONG = "long"
    SHORT = "short"
    
    @classmethod
    def from_string(cls, side: str) -> 'LiquidationSide':
        """Cria LiquidationSide a partir de string."""
        side_lower = side.lower().strip()
        
        # Mapeamentos comuns
        if side_lower in ['long', 'buy']:
            return cls.LONG
        elif side_lower in ['short', 'sell']:
            return cls.SHORT
        else:
            raise ValueError(f"Lado de liquidação inválido: {side}")
    
    def opposite(self) -> 'LiquidationSide':
        """Retorna o lado oposto."""
        return LiquidationSide.SHORT if self == LiquidationSide.LONG else LiquidationSide.LONG


@dataclass(frozen=True)
class LiquidationMetrics(BaseValueObject):
    """
    Value object com métricas calculadas da liquidação.
    
    Contém análises derivadas da liquidação para facilitar uso.
    """
    quantity: Decimal
    price: Decimal
    value: Decimal
    side: LiquidationSide
    
    def __post_init__(self):
        """Valida consistência das métricas."""
        # Verifica se value = quantity * price (com tolerância para arredondamento)
        expected_value = self.quantity * self.price
        tolerance = expected_value * Decimal('0.001')  # 0.1% tolerância
        
        if abs(self.value - expected_value) > tolerance:
            raise ValueError(
                f"Valor inconsistente: {self.value} != {self.quantity} * {self.price}"
            )
    
    @property
    def value_usd(self) -> Decimal:
        """Valor em USD (assumindo que já está em USD)."""
        return self.value
    
    @property
    def is_large_liquidation(self) -> bool:
        """Indica se é uma liquidação grande (>$100k)."""
        return self.value > Decimal('100000')
    
    @property
    def is_whale_liquidation(self) -> bool:
        """Indica se é uma liquidação de whale (>$1M)."""
        return self.value > Decimal('1000000')
    
    @property
    def liquidation_impact_category(self) -> str:
        """
        Categoriza o impacto potencial da liquidação.
        
        Returns:
            'low', 'medium', 'high', 'extreme'
        """
        value = self.value
        
        if value < Decimal('1000'):  # < $1k
            return 'low'
        elif value < Decimal('10000'):  # < $10k
            return 'medium'
        elif value < Decimal('100000'):  # < $100k
            return 'high'
        else:  # >= $100k
            return 'extreme'
    
    def calculate_market_impact_estimate(self, market_depth: Decimal) -> Decimal:
        """
        Estima impacto de mercado baseado na profundidade.
        
        Args:
            market_depth: Profundidade de mercado estimada
            
        Returns:
            Impacto estimado como percentual do preço
        """
        if market_depth <= 0:
            return Decimal('0')
        
        # Modelo simples: impacto = liquidation_value / market_depth
        impact_ratio = self.value / market_depth
        
        # Limita o impacto máximo a 10%
        return min(impact_ratio, Decimal('0.1'))


@dataclass(frozen=True)
class LiquidationRiskLevel(BaseValueObject):
    """
    Value object representando nível de risco da liquidação.
    
    Baseado em múltiplos fatores como valor, velocidade, concentração.
    """
    value_risk: str  # 'low', 'medium', 'high', 'extreme'
    timing_risk: str  # 'normal', 'clustered', 'cascade'
    overall_risk: str  # 'low', 'medium', 'high', 'critical'
    
    @classmethod
    def calculate_risk_level(
        cls,
        liquidation_value: Decimal,
        recent_liquidations_count: int = 0,
        time_since_last_liquidation_seconds: int = 3600
    ) -> 'LiquidationRiskLevel':
        """
        Calcula nível de risco baseado em parâmetros.
        
        Args:
            liquidation_value: Valor da liquidação
            recent_liquidations_count: Liquidações recentes
            time_since_last_liquidation_seconds: Tempo desde última liquidação
            
        Returns:
            LiquidationRiskLevel calculado
        """
        # Risco por valor
        if liquidation_value < Decimal('10000'):
            value_risk = 'low'
        elif liquidation_value < Decimal('100000'):
            value_risk = 'medium'
        elif liquidation_value < Decimal('1000000'):
            value_risk = 'high'
        else:
            value_risk = 'extreme'
        
        # Risco por timing
        if recent_liquidations_count > 10 and time_since_last_liquidation_seconds < 300:
            timing_risk = 'cascade'
        elif recent_liquidations_count > 3 and time_since_last_liquidation_seconds < 600:
            timing_risk = 'clustered'
        else:
            timing_risk = 'normal'
        
        # Risco geral
        risk_scores = {
            'low': 1, 'medium': 2, 'high': 3, 'extreme': 4,
            'normal': 1, 'clustered': 2, 'cascade': 3
        }
        
        total_score = risk_scores[value_risk] + risk_scores[timing_risk]
        
        if total_score <= 2:
            overall_risk = 'low'
        elif total_score <= 4:
            overall_risk = 'medium'
        elif total_score <= 6:
            overall_risk = 'high'
        else:
            overall_risk = 'critical'
        
        return cls(
            value_risk=value_risk,
            timing_risk=timing_risk,
            overall_risk=overall_risk
        )


class Liquidation(BaseEntity):
    """
    Entidade Liquidation.
    
    Representa um evento de liquidação forçada de uma posição em
    contratos perpétuos ou futuros.
    
    Attributes:
        exchange: Nome da exchange onde ocorreu a liquidação
        trading_pair: Par de negociação (ex: BTC/USDT)
        side: Lado da posição liquidada (long/short)
        quantity: Quantidade liquidada (em contratos ou tokens)
        price: Preço de liquidação
        value: Valor total liquidado (quantity * price)
        timestamp: Momento da liquidação
        raw_data: Dados brutos originais da exchange
    """
    
    def __init__(
        self,
        exchange: str,
        trading_pair: str,
        side: LiquidationSide,
        quantity: Decimal,
        price: Decimal,
        value: Decimal,
        timestamp: datetime,
        raw_data: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """
        Inicializa entidade Liquidation.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            side: Lado da liquidação (long/short)
            quantity: Quantidade liquidada
            price: Preço de liquidação
            value: Valor total liquidado
            timestamp: Timestamp da liquidação
            raw_data: Dados brutos originais
            id: ID único (será gerado se não fornecido)
            created_at: Data de criação
            updated_at: Data de atualização
        """
        # Validações
        self._validate_exchange(exchange)
        self._validate_trading_pair(trading_pair)
        self._validate_side(side)
        self._validate_quantity(quantity)
        self._validate_price(price)
        self._validate_value(value)
        self._validate_timestamp(timestamp)
        self._validate_consistency(quantity, price, value)
        
        # Atributos principais
        self.exchange = exchange.lower().strip()
        self.trading_pair = trading_pair.upper().strip()
        self.side = side
        self.quantity = quantity
        self.price = price
        self.value = value
        self.timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        self.raw_data = raw_data or {}
        
        # Value objects calculados
        self._metrics = LiquidationMetrics(quantity, price, value, side)
        
        # Inicializa BaseEntity
        super().__init__(
            id=id or self._generate_id(),
            created_at=created_at,
            updated_at=updated_at
        )
    
    @property
    def metrics(self) -> LiquidationMetrics:
        """Métricas calculadas da liquidação."""
        return self._metrics
    
    @property
    def value_usd(self) -> Decimal:
        """Valor em USD."""
        return self.metrics.value_usd
    
    @property
    def is_long_liquidation(self) -> bool:
        """Verifica se é liquidação de posição long."""
        return self.side == LiquidationSide.LONG
    
    @property
    def is_short_liquidation(self) -> bool:
        """Verifica se é liquidação de posição short."""
        return self.side == LiquidationSide.SHORT
    
    @property
    def is_large_liquidation(self) -> bool:
        """Verifica se é uma liquidação grande."""
        return self.metrics.is_large_liquidation
    
    @property
    def is_whale_liquidation(self) -> bool:
        """Verifica se é uma liquidação de whale."""
        return self.metrics.is_whale_liquidation
    
    @property
    def impact_category(self) -> str:
        """Categoria de impacto da liquidação."""
        return self.metrics.liquidation_impact_category
    
    def calculate_risk_level(
        self,
        recent_liquidations_count: int = 0,
        time_since_last_liquidation_seconds: int = 3600
    ) -> LiquidationRiskLevel:
        """
        Calcula nível de risco da liquidação.
        
        Args:
            recent_liquidations_count: Número de liquidações recentes
            time_since_last_liquidation_seconds: Tempo desde última liquidação
            
        Returns:
            LiquidationRiskLevel calculado
        """
        return LiquidationRiskLevel.calculate_risk_level(
            self.value,
            recent_liquidations_count,
            time_since_last_liquidation_seconds
        )
    
    def estimate_market_impact(self, market_depth: Decimal) -> Decimal:
        """
        Estima impacto no mercado.
        
        Args:
            market_depth: Profundidade de mercado estimada
            
        Returns:
            Impacto estimado como percentual
        """
        return self.metrics.calculate_market_impact_estimate(market_depth)
    
    def get_liquidation_direction_indicator(self) -> str:
        """
        Indica direção da pressão de liquidação.
        
        Returns:
            'bullish_pressure', 'bearish_pressure' ou 'neutral'
        """
        if self.side == LiquidationSide.LONG:
            # Long liquidations criam pressão de venda (bearish)
            return 'bearish_pressure'
        elif self.side == LiquidationSide.SHORT:
            # Short liquidations criam pressão de compra (bullish)
            return 'bullish_pressure'
        else:
            return 'neutral'
    
    def time_since_liquidation(self) -> timedelta:
        """
        Calcula tempo desde a liquidação.
        
        Returns:
            Timedelta desde a liquidação
        """
        return datetime.now(timezone.utc) - self.timestamp
    
    def is_recent(self, minutes: int = 60) -> bool:
        """
        Verifica se é uma liquidação recente.
        
        Args:
            minutes: Threshold em minutos
            
        Returns:
            True se liquidação ocorreu nos últimos X minutos
        """
        return self.time_since_liquidation().total_seconds() <= (minutes * 60)
    
    def compare_with(self, other: 'Liquidation') -> Dict[str, Any]:
        """
        Compara com outra liquidação.
        
        Args:
            other: Outra Liquidation para comparação
            
        Returns:
            Dicionário com análise comparativa
        """
        if not isinstance(other, Liquidation):
            raise TypeError("Comparação deve ser com outra Liquidation")
        
        value_ratio = self.value / other.value if other.value > 0 else Decimal('inf')
        time_diff = abs((self.timestamp - other.timestamp).total_seconds())
        
        return {
            'exchanges': [self.exchange, other.exchange],
            'trading_pairs': [self.trading_pair, other.trading_pair],
            'sides': [self.side.value, other.side.value],
            'values': [float(self.value), float(other.value)],
            'value_ratio': float(value_ratio),
            'prices': [float(self.price), float(other.price)],
            'quantities': [float(self.quantity), float(other.quantity)],
            'time_difference_seconds': time_diff,
            'same_side': self.side == other.side,
            'same_exchange': self.exchange == other.exchange,
            'same_pair': self.trading_pair == other.trading_pair,
            'larger_liquidation': 'self' if self.value > other.value else 'other',
            'impact_categories': [self.impact_category, other.impact_category]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'id': self.id,
            'exchange': self.exchange,
            'trading_pair': self.trading_pair,
            'side': self.side.value,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'value': float(self.value),
            'value_usd': float(self.value_usd),
            'timestamp': self.timestamp.isoformat(),
            'impact_category': self.impact_category,
            'is_large_liquidation': self.is_large_liquidation,
            'is_whale_liquidation': self.is_whale_liquidation,
            'liquidation_direction_indicator': self.get_liquidation_direction_indicator(),
            'time_since_liquidation_seconds': self.time_since_liquidation().total_seconds(),
            'is_recent': self.is_recent(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'raw_data': self.raw_data
        }
    
    def _generate_id(self) -> str:
        """Gera ID único baseado em exchange, pair, timestamp e valor."""
        # Cria hash baseado em dados únicos
        unique_string = (
            f"{self.exchange}:{self.trading_pair}:{self.timestamp.isoformat()}:"
            f"{self.side.value}:{self.value}"
        )
        hash_object = hashlib.sha256(unique_string.encode())
        hash_hex = hash_object.hexdigest()[:16]  # Primeiros 16 caracteres
        
        return f"liq_{hash_hex}"
    
    def _validate_exchange(self, exchange: str) -> None:
        """Valida nome da exchange."""
        if not exchange or not isinstance(exchange, str):
            raise ValueError("Exchange deve ser uma string não-vazia")
        
        if len(exchange.strip()) < 2:
            raise ValueError("Nome da exchange muito curto")
    
    def _validate_trading_pair(self, trading_pair: str) -> None:
        """Valida par de negociação."""
        if not trading_pair or not isinstance(trading_pair, str):
            raise ValueError("Trading pair deve ser uma string não-vazia")
        
        # Formato esperado: BASE/QUOTE ou BASE-USDT
        pair = trading_pair.upper().strip()
        
        if '/' not in pair and '-' not in pair:
            raise ValueError("Trading pair deve conter '/' ou '-' como separador")
        
        # Verifica se tem pelo menos 2 partes
        separator = '/' if '/' in pair else '-'
        parts = pair.split(separator)
        
        if len(parts) != 2:
            raise ValueError("Trading pair deve ter exatamente uma base e uma quote")
        
        base, quote = parts
        if len(base) < 2 or len(quote) < 2:
            raise ValueError("Base e quote devem ter pelo menos 2 caracteres")
    
    def _validate_side(self, side: LiquidationSide) -> None:
        """Valida lado da liquidação."""
        if not isinstance(side, LiquidationSide):
            raise TypeError("Side deve ser uma instância de LiquidationSide")
    
    def _validate_quantity(self, quantity: Decimal) -> None:
        """Valida quantidade."""
        if not isinstance(quantity, Decimal):
            raise TypeError("Quantity deve ser um Decimal")
        
        if quantity <= Decimal('0'):
            raise ValueError("Quantity deve ser positiva")
        
        # Verifica se não é um valor absurdamente grande
        if quantity > Decimal('1e12'):  # 1 trilhão
            raise ValueError("Quantity muito grande, possível erro nos dados")
    
    def _validate_price(self, price: Decimal) -> None:
        """Valida preço."""
        if not isinstance(price, Decimal):
            raise TypeError("Price deve ser um Decimal")
        
        if price <= Decimal('0'):
            raise ValueError("Price deve ser positivo")
        
        # Verifica se não é um valor absurdamente grande
        if price > Decimal('1e9'):  # 1 bilhão por token
            raise ValueError("Price muito alto, possível erro nos dados")
    
    def _validate_value(self, value: Decimal) -> None:
        """Valida valor."""
        if not isinstance(value, Decimal):
            raise TypeError("Value deve ser um Decimal")
        
        if value <= Decimal('0'):
            raise ValueError("Value deve ser positivo")
        
        # Verifica se não é um valor absurdamente grande
        if value > Decimal('1e15'):  # 1 quatrilhão de dólares
            raise ValueError("Value muito alto, possível erro nos dados")
    
    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Valida timestamp."""
        if not timestamp or not isinstance(timestamp, datetime):
            raise ValueError("Timestamp deve ser um datetime válido")
        
        # Não pode ser muito antigo ou muito no futuro
        now = datetime.now(timezone.utc)
        max_age = now - timedelta(days=365)  # Máximo 1 ano atrás
        max_future = now + timedelta(minutes=5)  # Máximo 5 minutos no futuro
        
        ts_utc = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        
        if ts_utc < max_age:
            raise ValueError(f"Timestamp muito antigo: {timestamp}")
        
        if ts_utc > max_future:
            raise ValueError(f"Timestamp muito no futuro: {timestamp}")
    
    def _validate_consistency(self, quantity: Decimal, price: Decimal, value: Decimal) -> None:
        """Valida consistência entre quantity, price e value."""
        expected_value = quantity * price
        tolerance = expected_value * Decimal('0.01')  # 1% tolerância
        
        if abs(value - expected_value) > tolerance:
            raise ValueError(
                f"Valor inconsistente: {value} != {quantity} * {price} "
                f"(diferença: {abs(value - expected_value)})"
            )
    
    def __eq__(self, other) -> bool:
        """Igualdade baseada em exchange, pair, timestamp e valor."""
        if not isinstance(other, Liquidation):
            return False
        
        return (
            self.exchange == other.exchange and
            self.trading_pair == other.trading_pair and
            self.timestamp == other.timestamp and
            self.value == other.value
        )
    
    def __hash__(self) -> int:
        """Hash baseado em exchange, pair, timestamp e valor."""
        return hash((self.exchange, self.trading_pair, self.timestamp, self.value))
    
    def __str__(self) -> str:
        """Representação em string."""
        return (
            f"Liquidation({self.exchange}:{self.trading_pair}, "
            f"{self.side.value}, ${self.value:,.0f}, "
            f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')})"
        )
    
    def __repr__(self) -> str:
        """Representação para debug."""
        return (
            f"Liquidation(id='{self.id}', exchange='{self.exchange}', "
            f"trading_pair='{self.trading_pair}', side={self.side}, "
            f"value={self.value}, timestamp={self.timestamp})"
        )


# Factory functions para criar instâncias comuns
def create_liquidation_from_api_data(
    exchange: str,
    api_data: Dict[str, Any],
    trading_pair: Optional[str] = None
) -> Liquidation:
    """
    Cria Liquidation a partir de dados de API.
    
    Args:
        exchange: Nome da exchange
        api_data: Dados da API
        trading_pair: Par de negociação (se não estiver nos dados)
        
    Returns:
        Instância de Liquidation
    """
    # Normaliza formato dos dados da API
    pair = trading_pair or api_data.get('symbol') or api_data.get('trading_pair')
    
    # Side
    side_raw = api_data.get('side') or api_data.get('direction')
    side = LiquidationSide.from_string(side_raw)
    
    # Valores numéricos
    quantity = Decimal(str(api_data.get('quantity', api_data.get('size', api_data.get('qty', 0)))))
    price = Decimal(str(api_data.get('price', api_data.get('liquidationPrice', 0))))
    value = api_data.get('value') or api_data.get('notional')
    
    if value is None:
        value = quantity * price
    else:
        value = Decimal(str(value))
    
    # Timestamp
    timestamp_raw = api_data.get('timestamp') or api_data.get('time') or api_data.get('liquidationTime')
    
    if isinstance(timestamp_raw, (int, float)):
        # Timestamp Unix (pode ser em segundos ou milissegundos)
        if timestamp_raw > 1e12:  # Milissegundos
            timestamp = datetime.fromtimestamp(timestamp_raw / 1000, tz=timezone.utc)
        else:  # Segundos
            timestamp = datetime.fromtimestamp(timestamp_raw, tz=timezone.utc)
    elif isinstance(timestamp_raw, str):
        # String ISO
        timestamp = datetime.fromisoformat(timestamp_raw.replace('Z', '+00:00'))
    else:
        timestamp = datetime.now(timezone.utc)
    
    return Liquidation(
        exchange=exchange,
        trading_pair=pair,
        side=side,
        quantity=quantity,
        price=price,
        value=value,
        timestamp=timestamp,
        raw_data=api_data
    )


def create_liquidation_batch_from_api_data(
    exchange: str,
    api_data_list: List[Dict[str, Any]],
    trading_pair: Optional[str] = None
) -> List[Liquidation]:
    """
    Cria múltiplas Liquidations a partir de lista de dados de API.
    
    Args:
        exchange: Nome da exchange
        api_data_list: Lista de dados da API
        trading_pair: Par de negociação (se não estiver nos dados)
        
    Returns:
        Lista de instâncias de Liquidation
    """
    liquidations = []
    
    for api_data in api_data_list:
        try:
            liquidation = create_liquidation_from_api_data(exchange, api_data, trading_pair)
            liquidations.append(liquidation)
        except Exception as e:
            # Log erro mas continua processando outras liquidações
            continue
    
    return liquidations


def create_synthetic_liquidation(
    exchange: str,
    trading_pair: str,
    side: LiquidationSide,
    value: Decimal,
    price: Decimal,
    timestamp: Optional[datetime] = None
) -> Liquidation:
    """
    Cria uma liquidação sintética para testes ou simulações.
    
    Args:
        exchange: Nome da exchange
        trading_pair: Par de negociação
        side: Lado da liquidação
        value: Valor total da liquidação
        price: Preço de liquidação
        timestamp: Timestamp (atual se não fornecido)
        
    Returns:
        Liquidation sintética
    """
    quantity = value / price
    
    return Liquidation(
        exchange=exchange,
        trading_pair=trading_pair,
        side=side,
        quantity=quantity,
        price=price,
        value=value,
        timestamp=timestamp or datetime.now(timezone.utc),
        raw_data={'synthetic': True}
    )


# Utility functions para análise de liquidações
def aggregate_liquidations_by_side(
    liquidations: List[Liquidation]
) -> Dict[str, Dict[str, Any]]:
    """
    Agrega liquidações por lado (long/short).
    
    Args:
        liquidations: Lista de liquidações
        
    Returns:
        Dicionário com agregações por lado
    """
    long_liquidations = [liq for liq in liquidations if liq.side == LiquidationSide.LONG]
    short_liquidations = [liq for liq in liquidations if liq.side == LiquidationSide.SHORT]
    
    def aggregate_side(side_liquidations: List[Liquidation]) -> Dict[str, Any]:
        if not side_liquidations:
            return {
                'count': 0,
                'total_value': 0,
                'avg_value': 0,
                'max_value': 0,
                'min_value': 0
            }
        
        values = [float(liq.value) for liq in side_liquidations]
        
        return {
            'count': len(side_liquidations),
            'total_value': sum(values),
            'avg_value': sum(values) / len(values),
            'max_value': max(values),
            'min_value': min(values),
            'large_liquidations': len([liq for liq in side_liquidations if liq.is_large_liquidation]),
            'whale_liquidations': len([liq for liq in side_liquidations if liq.is_whale_liquidation])
        }
    
    return {
        'long': aggregate_side(long_liquidations),
        'short': aggregate_side(short_liquidations),
        'total': {
            'count': len(liquidations),
            'long_count': len(long_liquidations),
            'short_count': len(short_liquidations),
            'long_short_ratio': len(long_liquidations) / max(1, len(short_liquidations))
        }
    }


def find_liquidation_clusters(
    liquidations: List[Liquidation],
    max_time_gap_minutes: int = 5,
    min_cluster_size: int = 3
) -> List[List[Liquidation]]:
    """
    Encontra clusters de liquidações (liquidações próximas no tempo).
    
    Args:
        liquidations: Lista de liquidações
        max_time_gap_minutes: Gap máximo entre liquidações no cluster
        min_cluster_size: Tamanho mínimo do cluster
        
    Returns:
        Lista de clusters (cada cluster é uma lista de liquidações)
    """
    if not liquidations:
        return []
    
    # Ordena por timestamp
    sorted_liquidations = sorted(liquidations, key=lambda x: x.timestamp)
    
    clusters = []
    current_cluster = [sorted_liquidations[0]]
    
    for i in range(1, len(sorted_liquidations)):
        current_liq = sorted_liquidations[i]
        last_liq = current_cluster[-1]
        
        time_gap = (current_liq.timestamp - last_liq.timestamp).total_seconds() / 60
        
        if time_gap <= max_time_gap_minutes:
            current_cluster.append(current_liq)
        else:
            # Finaliza cluster atual se atende tamanho mínimo
            if len(current_cluster) >= min_cluster_size:
                clusters.append(current_cluster)
            
            # Inicia novo cluster
            current_cluster = [current_liq]
    
    # Adiciona último cluster se atende tamanho mínimo
    if len(current_cluster) >= min_cluster_size:
        clusters.append(current_cluster)
    
    return clusters