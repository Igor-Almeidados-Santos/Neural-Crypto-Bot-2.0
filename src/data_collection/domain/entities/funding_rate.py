"""
Entidade Funding Rate para contratos perpétuos.

Esta entidade representa a taxa de financiamento (funding rate) cobrada
em contratos perpétuos, essencial para análise de custos de carry e
estratégias de arbitragem.
"""
import hashlib
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from common.domain.base_entity import BaseEntity
from common.domain.base_value_object import BaseValueObject


@dataclass(frozen=True)
class FundingInterval(BaseValueObject):
    """
    Value object representando intervalo de funding.
    
    Define a frequência com que o funding é cobrado.
    """
    hours: int
    
    def __post_init__(self):
        """Valida o intervalo."""
        if self.hours <= 0 or self.hours > 24:
            raise ValueError(f"Intervalo de funding inválido: {self.hours} horas")
        
        # Intervalos comuns: 1h, 4h, 8h, 24h
        valid_intervals = [1, 4, 8, 24]
        if self.hours not in valid_intervals:
            raise ValueError(f"Intervalo não suportado: {self.hours}h. Válidos: {valid_intervals}")
    
    @property
    def periods_per_day(self) -> int:
        """Número de períodos de funding por dia."""
        return 24 // self.hours
    
    @property
    def periods_per_year(self) -> int:
        """Número de períodos de funding por ano."""
        return self.periods_per_day * 365
    
    def to_annual_multiplier(self) -> Decimal:
        """Multiplicador para converter rate para base anual."""
        return Decimal(str(self.periods_per_year))


@dataclass(frozen=True)
class FundingRateMetrics(BaseValueObject):
    """
    Value object com métricas calculadas do funding rate.
    
    Contém análises derivadas do funding rate para facilitar uso.
    """
    rate: Decimal
    interval: FundingInterval
    
    @property
    def rate_bps(self) -> Decimal:
        """Taxa em basis points (1% = 100 bps)."""
        return self.rate * Decimal('10000')
    
    @property
    def daily_rate(self) -> Decimal:
        """Taxa diária equivalente."""
        return self.rate * Decimal(str(self.interval.periods_per_day))
    
    @property
    def weekly_rate(self) -> Decimal:
        """Taxa semanal equivalente."""
        return self.daily_rate * Decimal('7')
    
    @property
    def monthly_rate(self) -> Decimal:
        """Taxa mensal equivalente (30 dias)."""
        return self.daily_rate * Decimal('30')
    
    @property
    def annual_rate(self) -> Decimal:
        """Taxa anual equivalente."""
        return self.daily_rate * Decimal('365')
    
    def calculate_funding_cost(
        self,
        position_size: Decimal,
        periods: int = 1
    ) -> Decimal:
        """
        Calcula custo de funding para uma posição.
        
        Args:
            position_size: Tamanho da posição em valor nocional
            periods: Número de períodos de funding
            
        Returns:
            Custo total de funding
        """
        return position_size * self.rate * Decimal(str(periods))
    
    def is_contango(self) -> bool:
        """Indica se está em contango (funding positivo)."""
        return self.rate > Decimal('0')
    
    def is_backwardation(self) -> bool:
        """Indica se está em backwardation (funding negativo)."""
        return self.rate < Decimal('0')
    
    def get_market_sentiment(self) -> str:
        """
        Indica sentimento de mercado baseado no funding rate.
        
        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        rate_bps = abs(self.rate_bps)
        
        if self.rate > Decimal('0'):
            if rate_bps > Decimal('50'):  # > 0.5%
                return 'very_bullish'
            elif rate_bps > Decimal('10'):  # > 0.1%
                return 'bullish'
            else:
                return 'slightly_bullish'
        elif self.rate < Decimal('0'):
            if rate_bps > Decimal('50'):  # > 0.5%
                return 'very_bearish'
            elif rate_bps > Decimal('10'):  # > 0.1%
                return 'bearish'
            else:
                return 'slightly_bearish'
        else:
            return 'neutral'


class FundingRate(BaseEntity):
    """
    Entidade Funding Rate.
    
    Representa a taxa de financiamento para um contrato perpétuo
    em uma exchange específica.
    
    Attributes:
        exchange: Nome da exchange
        trading_pair: Par de negociação (ex: BTC/USDT)
        rate: Taxa de funding (formato decimal, ex: 0.0001 = 0.01%)
        timestamp: Momento da aplicação do funding
        next_funding_time: Próximo momento de cobrança de funding
        predicted_rate: Taxa prevista para o próximo período (opcional)
        interval_hours: Intervalo entre cobranças em horas (padrão: 8)
        raw_data: Dados brutos originais da exchange
    """
    
    def __init__(
        self,
        exchange: str,
        trading_pair: str,
        rate: Decimal,
        timestamp: datetime,
        next_funding_time: Optional[datetime] = None,
        predicted_rate: Optional[Decimal] = None,
        interval_hours: int = 8,
        raw_data: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None
    ):
        """
        Inicializa entidade FundingRate.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            rate: Taxa de funding
            timestamp: Timestamp do funding rate
            next_funding_time: Próximo momento de funding
            predicted_rate: Taxa prevista (opcional)
            interval_hours: Intervalo em horas
            raw_data: Dados brutos originais
            id: ID único (será gerado se não fornecido)
            created_at: Data de criação
            updated_at: Data de atualização
        """
        # Validações
        self._validate_exchange(exchange)
        self._validate_trading_pair(trading_pair)
        self._validate_rate(rate)
        self._validate_timestamp(timestamp)
        
        if predicted_rate is not None:
            self._validate_rate(predicted_rate)
        
        # Atributos principais
        self.exchange = exchange.lower().strip()
        self.trading_pair = trading_pair.upper().strip()
        self.rate = rate
        self.timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        self.next_funding_time = next_funding_time
        self.predicted_rate = predicted_rate
        self.interval_hours = interval_hours
        self.raw_data = raw_data or {}
        
        # Value objects calculados
        self._funding_interval = FundingInterval(interval_hours)
        self._metrics = FundingRateMetrics(rate, self._funding_interval)
        
        # Inicializa BaseEntity
        super().__init__(
            id=id or self._generate_id(),
            created_at=created_at,
            updated_at=updated_at
        )
    
    @property
    def funding_interval(self) -> FundingInterval:
        """Intervalo de funding."""
        return self._funding_interval
    
    @property
    def metrics(self) -> FundingRateMetrics:
        """Métricas calculadas do funding rate."""
        return self._metrics
    
    @property
    def rate_bps(self) -> Decimal:
        """Taxa em basis points."""
        return self.metrics.rate_bps
    
    @property
    def daily_rate(self) -> Decimal:
        """Taxa diária equivalente."""
        return self.metrics.daily_rate
    
    @property
    def annual_rate(self) -> Decimal:
        """Taxa anual equivalente."""
        return self.metrics.annual_rate
    
    def is_positive(self) -> bool:
        """Verifica se funding rate é positivo."""
        return self.rate > Decimal('0')
    
    def is_negative(self) -> bool:
        """Verifica se funding rate é negativo."""
        return self.rate < Decimal('0')
    
    def is_zero(self) -> bool:
        """Verifica se funding rate é zero."""
        return self.rate == Decimal('0')
    
    def is_extreme(self, threshold_bps: Decimal = Decimal('50')) -> bool:
        """
        Verifica se é um funding rate extremo.
        
        Args:
            threshold_bps: Threshold em basis points (padrão: 50 bps = 0.5%)
            
        Returns:
            True se o rate absoluto excede o threshold
        """
        return abs(self.rate_bps) > threshold_bps
    
    def get_market_sentiment(self) -> str:
        """Obtém sentimento de mercado baseado no funding rate."""
        return self.metrics.get_market_sentiment()
    
    def calculate_funding_cost(
        self,
        position_size: Decimal,
        periods: int = 1
    ) -> Decimal:
        """
        Calcula custo de funding para uma posição.
        
        Args:
            position_size: Tamanho da posição em valor nocional
            periods: Número de períodos de funding
            
        Returns:
            Custo total de funding
        """
        return self.metrics.calculate_funding_cost(position_size, periods)
    
    def time_to_next_funding(self) -> Optional[int]:
        """
        Calcula segundos até o próximo funding.
        
        Returns:
            Segundos até próximo funding ou None se não definido
        """
        if not self.next_funding_time:
            return None
        
        now = datetime.now(timezone.utc)
        if self.next_funding_time <= now:
            return 0
        
        return int((self.next_funding_time - now).total_seconds())
    
    def is_funding_soon(self, minutes: int = 15) -> bool:
        """
        Verifica se funding será cobrado em breve.
        
        Args:
            minutes: Threshold em minutos
            
        Returns:
            True se funding será cobrado nos próximos X minutos
        """
        seconds_to_funding = self.time_to_next_funding()
        if seconds_to_funding is None:
            return False
        
        return seconds_to_funding <= (minutes * 60)
    
    def compare_with(self, other: 'FundingRate') -> Dict[str, Any]:
        """
        Compara com outro funding rate.
        
        Args:
            other: Outro FundingRate para comparação
            
        Returns:
            Dicionário com análise comparativa
        """
        if not isinstance(other, FundingRate):
            raise TypeError("Comparação deve ser com outro FundingRate")
        
        if self.trading_pair != other.trading_pair:
            raise ValueError("Pares de negociação devem ser iguais para comparação")
        
        rate_diff = self.rate - other.rate
        rate_diff_bps = rate_diff * Decimal('10000')
        
        # Determina qual exchange é melhor para cada lado
        better_for_long = self.exchange if self.rate < other.rate else other.exchange
        better_for_short = self.exchange if self.rate > other.rate else other.exchange
        
        # Calcula oportunidade de arbitragem
        arbitrage_opportunity = abs(rate_diff_bps) > Decimal('5')  # > 5 bps
        
        return {
            'trading_pair': self.trading_pair,
            'exchanges': [self.exchange, other.exchange],
            'rates': [float(self.rate), float(other.rate)],
            'rate_difference': float(rate_diff),
            'rate_difference_bps': float(rate_diff_bps),
            'better_for_long': better_for_long,  # Exchange com menor rate
            'better_for_short': better_for_short,  # Exchange com maior rate
            'arbitrage_opportunity': arbitrage_opportunity,
            'self_sentiment': self.get_market_sentiment(),
            'other_sentiment': other.get_market_sentiment(),
            'timestamp_diff_seconds': abs((self.timestamp - other.timestamp).total_seconds())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'id': self.id,
            'exchange': self.exchange,
            'trading_pair': self.trading_pair,
            'rate': float(self.rate),
            'rate_bps': float(self.rate_bps),
            'timestamp': self.timestamp.isoformat(),
            'next_funding_time': self.next_funding_time.isoformat() if self.next_funding_time else None,
            'predicted_rate': float(self.predicted_rate) if self.predicted_rate else None,
            'interval_hours': self.interval_hours,
            'daily_rate': float(self.daily_rate),
            'annual_rate': float(self.annual_rate),
            'market_sentiment': self.get_market_sentiment(),
            'is_extreme': self.is_extreme(),
            'time_to_next_funding_seconds': self.time_to_next_funding(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'raw_data': self.raw_data
        }
    
    def _generate_id(self) -> str:
        """Gera ID único baseado em exchange, pair e timestamp."""
        # Cria hash baseado em dados únicos
        unique_string = f"{self.exchange}:{self.trading_pair}:{self.timestamp.isoformat()}"
        hash_object = hashlib.sha256(unique_string.encode())
        hash_hex = hash_object.hexdigest()[:16]  # Primeiros 16 caracteres
        
        return f"fr_{hash_hex}"
    
    def _validate_exchange(self, exchange: str) -> None:
        """Valida nome da exchange."""
        if not exchange or not isinstance(exchange, str):
            raise ValueError("Exchange deve ser uma string não-vazia")
        
        if len(exchange.strip()) < 2:
            raise ValueError("Nome da exchange muito curto")
        
        # Lista de exchanges conhecidas (pode ser expandida)
        known_exchanges = {
            'binance', 'bybit', 'okx', 'bitget', 'deribit', 'ftx',
            'bitmex', 'huobi', 'kucoin', 'gate', 'mexc', 'phemex'
        }
        
        if exchange.lower().strip() not in known_exchanges:
            # Log warning mas não falha - permite exchanges novas
            pass
    
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
    
    def _validate_rate(self, rate: Decimal) -> None:
        """Valida taxa de funding."""
        if not isinstance(rate, Decimal):
            raise TypeError("Rate deve ser um Decimal")
        
        # Funding rates tipicamente ficam entre -1% e +1%
        min_rate = Decimal('-0.01')  # -1%
        max_rate = Decimal('0.01')   # +1%
        
        if rate < min_rate or rate > max_rate:
            # Log warning mas não falha - permite rates extremos
            pass
    
    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Valida timestamp."""
        if not timestamp or not isinstance(timestamp, datetime):
            raise ValueError("Timestamp deve ser um datetime válido")
        
        # Não pode ser muito antigo ou muito no futuro
        now = datetime.now(timezone.utc)
        max_age = now - timedelta(days=365)  # Máximo 1 ano atrás
        max_future = now + timedelta(hours=1)  # Máximo 1 hora no futuro
        
        ts_utc = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        
        if ts_utc < max_age:
            raise ValueError(f"Timestamp muito antigo: {timestamp}")
        
        if ts_utc > max_future:
            raise ValueError(f"Timestamp muito no futuro: {timestamp}")
    
    def __eq__(self, other) -> bool:
        """Igualdade baseada em exchange, pair e timestamp."""
        if not isinstance(other, FundingRate):
            return False
        
        return (
            self.exchange == other.exchange and
            self.trading_pair == other.trading_pair and
            self.timestamp == other.timestamp
        )
    
    def __hash__(self) -> int:
        """Hash baseado em exchange, pair e timestamp."""
        return hash((self.exchange, self.trading_pair, self.timestamp))
    
    def __str__(self) -> str:
        """Representação em string."""
        return (
            f"FundingRate({self.exchange}:{self.trading_pair}, "
            f"rate={self.rate_bps:.1f}bps, "
            f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')})"
        )
    
    def __repr__(self) -> str:
        """Representação para debug."""
        return (
            f"FundingRate(id='{self.id}', exchange='{self.exchange}', "
            f"trading_pair='{self.trading_pair}', rate={self.rate}, "
            f"timestamp={self.timestamp})"
        )


# Factory functions para criar instâncias comuns
def create_funding_rate_from_api_data(
    exchange: str,
    api_data: Dict[str, Any],
    trading_pair: Optional[str] = None
) -> FundingRate:
    """
    Cria FundingRate a partir de dados de API.
    
    Args:
        exchange: Nome da exchange
        api_data: Dados da API
        trading_pair: Par de negociação (se não estiver nos dados)
        
    Returns:
        Instância de FundingRate
    """
    # Normaliza formato dos dados da API
    pair = trading_pair or api_data.get('symbol') or api_data.get('trading_pair')
    rate = Decimal(str(api_data.get('fundingRate', api_data.get('rate', 0))))
    
    # Tenta diferentes formatos de timestamp
    timestamp_raw = api_data.get('fundingTime') or api_data.get('timestamp') or api_data.get('time')
    
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
    
    # Next funding time
    next_funding_raw = api_data.get('nextFundingTime')
    next_funding_time = None
    
    if next_funding_raw:
        if isinstance(next_funding_raw, (int, float)):
            if next_funding_raw > 1e12:  # Milissegundos
                next_funding_time = datetime.fromtimestamp(next_funding_raw / 1000, tz=timezone.utc)
            else:  # Segundos
                next_funding_time = datetime.fromtimestamp(next_funding_raw, tz=timezone.utc)
        elif isinstance(next_funding_raw, str):
            next_funding_time = datetime.fromisoformat(next_funding_raw.replace('Z', '+00:00'))
    
    # Predicted rate
    predicted_rate = None
    if 'predictedFundingRate' in api_data:
        predicted_rate = Decimal(str(api_data['predictedFundingRate']))
    
    return FundingRate(
        exchange=exchange,
        trading_pair=pair,
        rate=rate,
        timestamp=timestamp,
        next_funding_time=next_funding_time,
        predicted_rate=predicted_rate,
        interval_hours=api_data.get('interval', 8),
        raw_data=api_data
    )


def create_zero_funding_rate(
    exchange: str,
    trading_pair: str,
    timestamp: Optional[datetime] = None
) -> FundingRate:
    """
    Cria um funding rate zero (para testes ou placeholder).
    
    Args:
        exchange: Nome da exchange
        trading_pair: Par de negociação
        timestamp: Timestamp (atual se não fornecido)
        
    Returns:
        FundingRate com rate zero
    """
    return FundingRate(
        exchange=exchange,
        trading_pair=trading_pair,
        rate=Decimal('0'),
        timestamp=timestamp or datetime.now(timezone.utc)
    )