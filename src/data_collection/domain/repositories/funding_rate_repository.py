"""
Repository para entidades FundingRate.

Este módulo implementa o padrão Repository para persistência de dados de funding rates,
essenciais para análise de contratos perpétuos e estratégias de arbitragem.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import statistics

from src.data_collection.domain.entities.funding_rate import FundingRate
from src.common.infrastructure.database.base_repository import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class FundingRateQuery:
    """Query builder para consultas de funding rates."""
    exchange: Optional[str] = None
    trading_pair: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_rate: Optional[Decimal] = None
    max_rate: Optional[Decimal] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: str = "timestamp"
    order_direction: str = "DESC"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'exchange': self.exchange,
            'trading_pair': self.trading_pair,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'min_rate': self.min_rate,
            'max_rate': self.max_rate,
            'limit': self.limit,
            'offset': self.offset,
            'order_by': self.order_by,
            'order_direction': self.order_direction
        }


@dataclass
class FundingRateStats:
    """Estatísticas de funding rates."""
    exchange: str
    trading_pair: str
    period_start: datetime
    period_end: datetime
    count: int
    avg_rate: Decimal
    min_rate: Decimal
    max_rate: Decimal
    median_rate: Decimal
    std_deviation: float
    positive_count: int
    negative_count: int
    zero_count: int
    annualized_rate: Decimal  # Taxa anualizada baseada na frequência
    total_funding_cost: Decimal  # Custo total de funding no período


@dataclass
class FundingRateComparison:
    """Comparação de funding rates entre exchanges."""
    trading_pair: str
    timestamp: datetime
    rates: Dict[str, Decimal]  # exchange -> rate
    spread: Decimal  # Diferença entre maior e menor rate
    arbitrage_opportunity: bool
    best_long_exchange: str  # Exchange com menor rate (melhor para long)
    best_short_exchange: str  # Exchange com maior rate (melhor para short)


class FundingRateRepositoryInterface(ABC):
    """Interface para repository de funding rates."""
    
    @abstractmethod
    async def save(self, funding_rate: FundingRate) -> FundingRate:
        """Salva um funding rate."""
        pass
    
    @abstractmethod
    async def save_batch(self, funding_rates: List[FundingRate]) -> List[FundingRate]:
        """Salva múltiplos funding rates em lote."""
        pass
    
    @abstractmethod
    async def find_by_id(self, funding_rate_id: str) -> Optional[FundingRate]:
        """Busca funding rate por ID."""
        pass
    
    @abstractmethod
    async def find_latest(
        self,
        exchange: str,
        trading_pair: str,
        limit: int = 1
    ) -> List[FundingRate]:
        """Busca funding rates mais recentes."""
        pass
    
    @abstractmethod
    async def find_by_query(self, query: FundingRateQuery) -> List[FundingRate]:
        """Busca funding rates por query."""
        pass
    
    @abstractmethod
    async def get_rate_at_time(
        self,
        exchange: str,
        trading_pair: str,
        timestamp: datetime
    ) -> Optional[FundingRate]:
        """Obtém funding rate válido em um momento específico."""
        pass
    
    @abstractmethod
    async def get_statistics(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[FundingRateStats]:
        """Calcula estatísticas de funding rates para um período."""
        pass
    
    @abstractmethod
    async def get_cross_exchange_comparison(
        self,
        trading_pair: str,
        timestamp: datetime,
        exchanges: Optional[List[str]] = None
    ) -> Optional[FundingRateComparison]:
        """Compara funding rates entre exchanges."""
        pass
    
    @abstractmethod
    async def get_rate_history(
        self,
        exchange: str,
        trading_pair: str,
        days: int = 30
    ) -> List[Tuple[datetime, Decimal]]:
        """Obtém histórico de funding rates."""
        pass
    
    @abstractmethod
    async def delete_old_rates(
        self,
        exchange: str,
        trading_pair: str,
        older_than: datetime
    ) -> int:
        """Remove funding rates antigos."""
        pass


class PostgreSQLFundingRateRepository(FundingRateRepositoryInterface, BaseRepository):
    """
    Implementação PostgreSQL do repository de funding rates.
    
    Otimizada para consultas temporais e análises estatísticas.
    """
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        self.table_name = "funding_rates"
        self._create_table_sql = """
        CREATE TABLE IF NOT EXISTS funding_rates (
            id VARCHAR(255) PRIMARY KEY,
            exchange VARCHAR(50) NOT NULL,
            trading_pair VARCHAR(20) NOT NULL,
            rate NUMERIC(15,8) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            next_funding_time TIMESTAMPTZ,
            predicted_rate NUMERIC(15,8),
            interval_hours INTEGER DEFAULT 8,
            raw_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(exchange, trading_pair, timestamp)
        );
        
        -- Índices para otimização
        CREATE INDEX IF NOT EXISTS idx_funding_rates_exchange_pair_time 
        ON funding_rates (exchange, trading_pair, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_funding_rates_timestamp 
        ON funding_rates (timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_funding_rates_rate 
        ON funding_rates (exchange, trading_pair, rate);
        
        CREATE INDEX IF NOT EXISTS idx_funding_rates_next_funding 
        ON funding_rates (next_funding_time);
        
        -- Índice para comparações cross-exchange
        CREATE INDEX IF NOT EXISTS idx_funding_rates_pair_time 
        ON funding_rates (trading_pair, timestamp DESC);
        
        -- TimescaleDB hypertable (se disponível)
        SELECT create_hypertable('funding_rates', 'timestamp', if_not_exists => TRUE);
        """
    
    async def initialize(self) -> None:
        """Inicializa o repository."""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(self._create_table_sql)
        logger.info("FundingRate repository inicializado")
    
    async def save(self, funding_rate: FundingRate) -> FundingRate:
        """Salva um funding rate."""
        sql = """
        INSERT INTO funding_rates (
            id, exchange, trading_pair, rate, timestamp,
            next_funding_time, predicted_rate, interval_hours, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (exchange, trading_pair, timestamp)
        DO UPDATE SET
            rate = EXCLUDED.rate,
            next_funding_time = EXCLUDED.next_funding_time,
            predicted_rate = EXCLUDED.predicted_rate,
            interval_hours = EXCLUDED.interval_hours,
            raw_data = EXCLUDED.raw_data,
            updated_at = NOW()
        RETURNING id, created_at, updated_at
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                sql,
                funding_rate.id,
                funding_rate.exchange,
                funding_rate.trading_pair,
                funding_rate.rate,
                funding_rate.timestamp,
                funding_rate.next_funding_time,
                funding_rate.predicted_rate,
                funding_rate.interval_hours,
                funding_rate.raw_data
            )
            
            if row:
                funding_rate.created_at = row['created_at']
                funding_rate.updated_at = row['updated_at']
        
        return funding_rate
    
    async def save_batch(self, funding_rates: List[FundingRate]) -> List[FundingRate]:
        """Salva múltiplos funding rates em lote."""
        if not funding_rates:
            return []
        
        sql = """
        INSERT INTO funding_rates (
            id, exchange, trading_pair, rate, timestamp,
            next_funding_time, predicted_rate, interval_hours, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (exchange, trading_pair, timestamp)
        DO UPDATE SET
            rate = EXCLUDED.rate,
            next_funding_time = EXCLUDED.next_funding_time,
            predicted_rate = EXCLUDED.predicted_rate,
            interval_hours = EXCLUDED.interval_hours,
            raw_data = EXCLUDED.raw_data,
            updated_at = NOW()
        """
        
        batch_data = [
            (
                fr.id,
                fr.exchange,
                fr.trading_pair,
                fr.rate,
                fr.timestamp,
                fr.next_funding_time,
                fr.predicted_rate,
                fr.interval_hours,
                fr.raw_data
            )
            for fr in funding_rates
        ]
        
        async with self.db_manager.get_connection() as conn:
            await conn.executemany(sql, batch_data)
        
        logger.debug(f"Salvos {len(funding_rates)} funding rates em lote")
        return funding_rates
    
    async def find_by_id(self, funding_rate_id: str) -> Optional[FundingRate]:
        """Busca funding rate por ID."""
        sql = "SELECT * FROM funding_rates WHERE id = $1"
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, funding_rate_id)
            
            if row:
                return self._row_to_funding_rate(row)
        
        return None
    
    async def find_latest(
        self,
        exchange: str,
        trading_pair: str,
        limit: int = 1
    ) -> List[FundingRate]:
        """Busca funding rates mais recentes."""
        sql = """
        SELECT * FROM funding_rates 
        WHERE exchange = $1 AND trading_pair = $2
        ORDER BY timestamp DESC
        LIMIT $3
        """
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange, trading_pair, limit)
            return [self._row_to_funding_rate(row) for row in rows]
    
    async def find_by_query(self, query: FundingRateQuery) -> List[FundingRate]:
        """Busca funding rates por query."""
        where_conditions = []
        params = []
        param_count = 0
        
        # Constrói condições WHERE
        if query.exchange:
            param_count += 1
            where_conditions.append(f"exchange = ${param_count}")
            params.append(query.exchange)
        
        if query.trading_pair:
            param_count += 1
            where_conditions.append(f"trading_pair = ${param_count}")
            params.append(query.trading_pair)
        
        if query.start_time:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(query.start_time)
        
        if query.end_time:
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(query.end_time)
        
        if query.min_rate:
            param_count += 1
            where_conditions.append(f"rate >= ${param_count}")
            params.append(query.min_rate)
        
        if query.max_rate:
            param_count += 1
            where_conditions.append(f"rate <= ${param_count}")
            params.append(query.max_rate)
        
        # Constrói SQL
        sql = "SELECT * FROM funding_rates"
        
        if where_conditions:
            sql += " WHERE " + " AND ".join(where_conditions)
        
        sql += f" ORDER BY {query.order_by} {query.order_direction}"
        
        if query.limit:
            param_count += 1
            sql += f" LIMIT ${param_count}"
            params.append(query.limit)
        
        if query.offset:
            param_count += 1
            sql += f" OFFSET ${param_count}"
            params.append(query.offset)
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_funding_rate(row) for row in rows]
    
    async def get_rate_at_time(
        self,
        exchange: str,
        trading_pair: str,
        timestamp: datetime
    ) -> Optional[FundingRate]:
        """Obtém funding rate válido em um momento específico."""
        sql = """
        SELECT * FROM funding_rates
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp <= $3
        AND (next_funding_time IS NULL OR next_funding_time > $3)
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, exchange, trading_pair, timestamp)
            
            if row:
                return self._row_to_funding_rate(row)
        
        return None
    
    async def get_statistics(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[FundingRateStats]:
        """Calcula estatísticas de funding rates para um período."""
        sql = """
        SELECT 
            COUNT(*) as count,
            AVG(rate) as avg_rate,
            MIN(rate) as min_rate,
            MAX(rate) as max_rate,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rate) as median_rate,
            STDDEV(rate) as std_deviation,
            COUNT(*) FILTER (WHERE rate > 0) as positive_count,
            COUNT(*) FILTER (WHERE rate < 0) as negative_count,
            COUNT(*) FILTER (WHERE rate = 0) as zero_count,
            array_agg(rate ORDER BY timestamp) as rates_array
        FROM funding_rates
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp >= $3 AND timestamp <= $4
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, exchange, trading_pair, start_time, end_time)
            
            if row and row['count'] > 0:
                # Calcula taxa anualizada baseada na frequência média
                rates_array = row['rates_array']
                avg_interval_hours = 8  # Padrão para a maioria das exchanges
                periods_per_year = (365 * 24) / avg_interval_hours
                annualized_rate = Decimal(str(row['avg_rate'])) * Decimal(str(periods_per_year))
                
                # Calcula custo total de funding
                total_funding_cost = sum(Decimal(str(rate)) for rate in rates_array)
                
                return FundingRateStats(
                    exchange=exchange,
                    trading_pair=trading_pair,
                    period_start=start_time,
                    period_end=end_time,
                    count=row['count'],
                    avg_rate=Decimal(str(row['avg_rate'])),
                    min_rate=Decimal(str(row['min_rate'])),
                    max_rate=Decimal(str(row['max_rate'])),
                    median_rate=Decimal(str(row['median_rate'])),
                    std_deviation=float(row['std_deviation'] or 0),
                    positive_count=row['positive_count'],
                    negative_count=row['negative_count'],
                    zero_count=row['zero_count'],
                    annualized_rate=annualized_rate,
                    total_funding_cost=total_funding_cost
                )
        
        return None
    
    async def get_cross_exchange_comparison(
        self,
        trading_pair: str,
        timestamp: datetime,
        exchanges: Optional[List[str]] = None
    ) -> Optional[FundingRateComparison]:
        """Compara funding rates entre exchanges."""
        # Constrói query baseado nas exchanges especificadas
        exchange_condition = ""
        params = [trading_pair, timestamp]
        
        if exchanges:
            placeholders = ",".join(f"${i+3}" for i in range(len(exchanges)))
            exchange_condition = f"AND exchange IN ({placeholders})"
            params.extend(exchanges)
        
        sql = f"""
        WITH latest_rates AS (
            SELECT DISTINCT ON (exchange) 
                exchange, rate
            FROM funding_rates
            WHERE trading_pair = $1
            AND timestamp <= $2
            {exchange_condition}
            ORDER BY exchange, timestamp DESC
        )
        SELECT 
            array_agg(exchange) as exchanges,
            array_agg(rate) as rates
        FROM latest_rates
        WHERE rate IS NOT NULL
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, *params)
            
            if row and row['exchanges']:
                exchanges_list = row['exchanges']
                rates_list = [Decimal(str(rate)) for rate in row['rates']]
                
                if len(exchanges_list) < 2:
                    return None
                
                # Cria dicionário de rates
                rates_dict = dict(zip(exchanges_list, rates_list))
                
                # Calcula estatísticas
                min_rate = min(rates_list)
                max_rate = max(rates_list)
                spread = max_rate - min_rate
                
                best_long_exchange = min(rates_dict.keys(), key=lambda x: rates_dict[x])
                best_short_exchange = max(rates_dict.keys(), key=lambda x: rates_dict[x])
                
                # Define threshold para oportunidade de arbitragem (0.01% = 0.0001)
                arbitrage_opportunity = spread > Decimal('0.0001')
                
                return FundingRateComparison(
                    trading_pair=trading_pair,
                    timestamp=timestamp,
                    rates=rates_dict,
                    spread=spread,
                    arbitrage_opportunity=arbitrage_opportunity,
                    best_long_exchange=best_long_exchange,
                    best_short_exchange=best_short_exchange
                )
        
        return None
    
    async def get_rate_history(
        self,
        exchange: str,
        trading_pair: str,
        days: int = 30
    ) -> List[Tuple[datetime, Decimal]]:
        """Obtém histórico de funding rates."""
        start_time = datetime.utcnow() - timedelta(days=days)
        
        sql = """
        SELECT timestamp, rate
        FROM funding_rates
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp >= $3
        ORDER BY timestamp ASC
        """
        
        history = []
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange, trading_pair, start_time)
            
            for row in rows:
                history.append((row['timestamp'], Decimal(str(row['rate']))))
        
        return history
    
    async def delete_old_rates(
        self,
        exchange: str,
        trading_pair: str,
        older_than: datetime
    ) -> int:
        """Remove funding rates antigos."""
        sql = """
        DELETE FROM funding_rates 
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp < $3
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.execute(sql, exchange, trading_pair, older_than)
            
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Removidos {deleted_count} funding rates antigos")
            return deleted_count
    
    def _row_to_funding_rate(self, row) -> FundingRate:
        """Converte row do banco para entidade FundingRate."""
        return FundingRate(
            id=row['id'],
            exchange=row['exchange'],
            trading_pair=row['trading_pair'],
            rate=Decimal(str(row['rate'])),
            timestamp=row['timestamp'],
            next_funding_time=row['next_funding_time'],
            predicted_rate=Decimal(str(row['predicted_rate'])) if row['predicted_rate'] else None,
            interval_hours=row['interval_hours'],
            raw_data=row['raw_data'],
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at')
        )
    
    async def get_repository_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas gerais do repository."""
        sql = """
        SELECT 
            COUNT(*) as total_rates,
            COUNT(DISTINCT exchange) as total_exchanges,
            COUNT(DISTINCT trading_pair) as total_pairs,
            MIN(timestamp) as oldest_rate,
            MAX(timestamp) as newest_rate,
            AVG(rate) as avg_rate,
            MIN(rate) as min_rate,
            MAX(rate) as max_rate
        FROM funding_rates
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql)
            
            return {
                'total_rates': row['total_rates'],
                'total_exchanges': row['total_exchanges'],
                'total_pairs': row['total_pairs'],
                'oldest_rate': row['oldest_rate'],
                'newest_rate': row['newest_rate'],
                'avg_rate': float(row['avg_rate']) if row['avg_rate'] else 0,
                'min_rate': float(row['min_rate']) if row['min_rate'] else 0,
                'max_rate': float(row['max_rate']) if row['max_rate'] else 0
            }


class FundingRateService:
    """
    Serviço de alto nível para operações com funding rates.
    
    Combina repository com lógica de negócio e análises especializadas.
    """
    
    def __init__(self, repository: FundingRateRepositoryInterface):
        self.repository = repository
    
    async def find_arbitrage_opportunities(
        self,
        trading_pairs: List[str],
        min_spread_bps: float = 10.0,  # 0.1%
        exchanges: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Encontra oportunidades de arbitragem de funding rates."""
        opportunities = []
        current_time = datetime.utcnow()
        
        for pair in trading_pairs:
            comparison = await self.repository.get_cross_exchange_comparison(
                pair, current_time, exchanges
            )
            
            if comparison and comparison.arbitrage_opportunity:
                spread_bps = float(comparison.spread) * 10000  # Converte para basis points
                
                if spread_bps >= min_spread_bps:
                    opportunities.append({
                        'trading_pair': pair,
                        'spread_bps': spread_bps,
                        'spread_decimal': float(comparison.spread),
                        'best_long_exchange': comparison.best_long_exchange,
                        'best_short_exchange': comparison.best_short_exchange,
                        'long_rate': float(comparison.rates[comparison.best_long_exchange]),
                        'short_rate': float(comparison.rates[comparison.best_short_exchange]),
                        'all_rates': {k: float(v) for k, v in comparison.rates.items()},
                        'timestamp': comparison.timestamp.isoformat()
                    })
        
        # Ordena por spread (maior primeiro)
        opportunities.sort(key=lambda x: x['spread_bps'], reverse=True)
        return opportunities
    
    async def calculate_funding_cost_analysis(
        self,
        exchange: str,
        trading_pair: str,
        position_size_usd: Decimal,
        days: int = 30
    ) -> Dict[str, Any]:
        """Calcula análise detalhada de custo de funding."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        # Obtém estatísticas do período
        stats = await self.repository.get_statistics(
            exchange, trading_pair, start_time, end_time
        )
        
        if not stats:
            return {'error': 'Dados insuficientes'}
        
        # Calcula custos baseado no tamanho da posição
        daily_avg_rate = stats.avg_rate / Decimal('3')  # 3 funding por dia (a cada 8h)
        daily_funding_cost = position_size_usd * daily_avg_rate
        
        monthly_cost = daily_funding_cost * 30
        yearly_cost = daily_funding_cost * 365
        
        # Calcula custos nos cenários extremos
        worst_case_daily = position_size_usd * (stats.max_rate / Decimal('3'))
        best_case_daily = position_size_usd * (stats.min_rate / Decimal('3'))
        
        return {
            'exchange': exchange,
            'trading_pair': trading_pair,
            'position_size_usd': float(position_size_usd),
            'period_days': days,
            'statistics': {
                'avg_rate': float(stats.avg_rate),
                'min_rate': float(stats.min_rate),
                'max_rate': float(stats.max_rate),
                'median_rate': float(stats.median_rate),
                'std_deviation': stats.std_deviation,
                'annualized_rate': float(stats.annualized_rate)
            },
            'funding_costs': {
                'daily_avg_usd': float(daily_funding_cost),
                'monthly_projected_usd': float(monthly_cost),
                'yearly_projected_usd': float(yearly_cost),
                'worst_case_daily_usd': float(worst_case_daily),
                'best_case_daily_usd': float(best_case_daily)
            },
            'funding_frequency': {
                'positive_percent': (stats.positive_count / stats.count) * 100,
                'negative_percent': (stats.negative_count / stats.count) * 100,
                'zero_percent': (stats.zero_count / stats.count) * 100
            }
        }
    
    async def predict_next_funding_rates(
        self,
        exchange: str,
        trading_pair: str,
        periods_ahead: int = 3
    ) -> List[Dict[str, Any]]:
        """Prevê próximos funding rates baseado em tendências históricas."""
        # Obtém histórico recente (últimos 30 dias)
        history = await self.repository.get_rate_history(
            exchange, trading_pair, 30
        )
        
        if len(history) < 10:
            return []
        
        # Extrai apenas as taxas para análise
        rates = [float(rate) for _, rate in history]
        
        # Cálculos estatísticos simples
        recent_avg = statistics.mean(rates[-10:])  # Média dos últimos 10
        overall_avg = statistics.mean(rates)
        trend = recent_avg - overall_avg
        
        # Obtém último funding rate
        latest_rate = rates[-1]
        
        predictions = []
        next_time = datetime.utcnow()
        
        for i in range(periods_ahead):
            # Incrementa tempo (assumindo 8h entre funding)
            next_time += timedelta(hours=8)
            
            # Predição simples baseada na tendência
            predicted_rate = latest_rate + (trend * (i + 1) * 0.1)  # Atenua a tendência
            
            # Limita a predição a valores razoáveis (-1% a +1%)
            predicted_rate = max(-0.01, min(0.01, predicted_rate))
            
            predictions.append({
                'timestamp': next_time.isoformat(),
                'predicted_rate': predicted_rate,
                'confidence': max(0.3, 0.9 - (i * 0.2)),  # Confiança decrescente
                'trend_direction': 'positive' if trend > 0 else 'negative' if trend < 0 else 'neutral'
            })
        
        return predictions
    
    async def analyze_funding_rate_patterns(
        self,
        exchange: str,
        trading_pair: str,
        days: int = 90
    ) -> Dict[str, Any]:
        """Analisa padrões em funding rates."""
        history = await self.repository.get_rate_history(
            exchange, trading_pair, days
        )
        
        if len(history) < 30:
            return {'error': 'Dados insuficientes para análise de padrões'}
        
        # Agrupa por hora do dia
        hourly_patterns = {}
        for timestamp, rate in history:
            hour = timestamp.hour
            if hour not in hourly_patterns:
                hourly_patterns[hour] = []
            hourly_patterns[hour].append(float(rate))
        
        # Calcula médias por hora
        hourly_averages = {
            hour: statistics.mean(rates) 
            for hour, rates in hourly_patterns.items()
        }
        
        # Agrupa por dia da semana
        weekday_patterns = {}
        for timestamp, rate in history:
            weekday = timestamp.weekday()  # 0=Segunda, 6=Domingo
            if weekday not in weekday_patterns:
                weekday_patterns[weekday] = []
            weekday_patterns[weekday].append(float(rate))
        
        weekday_averages = {
            day: statistics.mean(rates) 
            for day, rates in weekday_patterns.items()
        }
        
        weekday_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        
        # Identifica extremos
        rates = [float(rate) for _, rate in history]
        avg_rate = statistics.mean(rates)
        
        extreme_periods = []
        for timestamp, rate in history:
            if abs(float(rate)) > abs(avg_rate) * 3:  # 3x a média
                extreme_periods.append({
                    'timestamp': timestamp.isoformat(),
                    'rate': float(rate),
                    'type': 'high_positive' if rate > 0 else 'high_negative'
                })
        
        return {
            'exchange': exchange,
            'trading_pair': trading_pair,
            'analysis_period_days': days,
            'total_observations': len(history),
            'hourly_patterns': {
                f"{hour:02d}:00": avg_rate 
                for hour, avg_rate in sorted(hourly_averages.items())
            },
            'weekday_patterns': {
                weekday_names[day]: avg_rate 
                for day, avg_rate in sorted(weekday_averages.items())
            },
            'extreme_periods': extreme_periods[-10:],  # Últimos 10 extremos
            'best_funding_hours': sorted(
                hourly_averages.items(), 
                key=lambda x: x[1]
            )[:3],  # 3 melhores horas (menores rates)
            'worst_funding_hours': sorted(
                hourly_averages.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]  # 3 piores horas (maiores rates)
        }