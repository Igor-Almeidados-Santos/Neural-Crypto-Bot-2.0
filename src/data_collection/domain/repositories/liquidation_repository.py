"""
Repository para entidades Liquidation.

Este módulo implementa o padrão Repository para persistência de dados de liquidações,
fundamentais para análise de risco de mercado e detecção de movimentos extremos.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import statistics

from data_collection.domain.entities.liquidation import Liquidation, LiquidationSide
from common.infrastructure.database.base_repository import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class LiquidationQuery:
    """Query builder para consultas de liquidações."""
    exchange: Optional[str] = None
    trading_pair: Optional[str] = None
    side: Optional[LiquidationSide] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_value: Optional[Decimal] = None
    max_value: Optional[Decimal] = None
    min_price: Optional[Decimal] = None
    max_price: Optional[Decimal] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: str = "timestamp"
    order_direction: str = "DESC"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'exchange': self.exchange,
            'trading_pair': self.trading_pair,
            'side': self.side.value if self.side else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'min_price': self.min_price,
            'max_price': self.max_price,
            'limit': self.limit,
            'offset': self.offset,
            'order_by': self.order_by,
            'order_direction': self.order_direction
        }


@dataclass
class LiquidationStats:
    """Estatísticas de liquidações."""
    exchange: str
    trading_pair: str
    period_start: datetime
    period_end: datetime
    total_liquidations: int
    total_value: Decimal
    avg_liquidation_value: Decimal
    max_liquidation_value: Decimal
    long_liquidations: int
    short_liquidations: int
    long_value: Decimal
    short_value: Decimal
    liquidation_rate_per_hour: float
    dominant_side: LiquidationSide
    price_range: Tuple[Decimal, Decimal]  # (min_price, max_price)


@dataclass
class LiquidationHeatmap:
    """Heatmap de liquidações por preço."""
    exchange: str
    trading_pair: str
    timestamp: datetime
    price_levels: List[Decimal]  # Níveis de preço
    long_liquidations: List[int]  # Quantidade de longs liquidados por nível
    short_liquidations: List[int]  # Quantidade de shorts liquidados por nível
    liquidation_values: List[Decimal]  # Valor total liquidado por nível


@dataclass
class LiquidationCluster:
    """Cluster de liquidações (concentração em curto período)."""
    exchange: str
    trading_pair: str
    start_time: datetime
    end_time: datetime
    duration_seconds: int
    liquidation_count: int
    total_value: Decimal
    dominant_side: LiquidationSide
    trigger_price: Decimal  # Preço que possivelmente trigger o cluster
    intensity: float  # Intensidade do cluster (liquidações por segundo)


class LiquidationRepositoryInterface(ABC):
    """Interface para repository de liquidações."""
    
    @abstractmethod
    async def save(self, liquidation: Liquidation) -> Liquidation:
        """Salva uma liquidação."""
        pass
    
    @abstractmethod
    async def save_batch(self, liquidations: List[Liquidation]) -> List[Liquidation]:
        """Salva múltiplas liquidações em lote."""
        pass
    
    @abstractmethod
    async def find_by_id(self, liquidation_id: str) -> Optional[Liquidation]:
        """Busca liquidação por ID."""
        pass
    
    @abstractmethod
    async def find_by_query(self, query: LiquidationQuery) -> List[Liquidation]:
        """Busca liquidações por query."""
        pass
    
    @abstractmethod
    async def get_statistics(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[LiquidationStats]:
        """Calcula estatísticas de liquidações para um período."""
        pass
    
    @abstractmethod
    async def get_liquidation_heatmap(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        price_buckets: int = 50
    ) -> Optional[LiquidationHeatmap]:
        """Gera heatmap de liquidações por faixa de preço."""
        pass
    
    @abstractmethod
    async def detect_liquidation_clusters(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        min_cluster_size: int = 5,
        max_cluster_duration_minutes: int = 5
    ) -> List[LiquidationCluster]:
        """Detecta clusters de liquidações."""
        pass
    
    @abstractmethod
    async def get_large_liquidations(
        self,
        exchange: str,
        trading_pair: str,
        min_value: Decimal,
        hours: int = 24
    ) -> List[Liquidation]:
        """Obtém liquidações de grande valor."""
        pass
    
    @abstractmethod
    async def delete_old_liquidations(
        self,
        exchange: str,
        trading_pair: str,
        older_than: datetime
    ) -> int:
        """Remove liquidações antigas."""
        pass


class PostgreSQLLiquidationRepository(LiquidationRepositoryInterface, BaseRepository):
    """
    Implementação PostgreSQL do repository de liquidações.
    
    Otimizada para análises de risco e detecção de movimentos extremos.
    """
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        self.table_name = "liquidations"
        self._create_table_sql = """
        CREATE TABLE IF NOT EXISTS liquidations (
            id VARCHAR(255) PRIMARY KEY,
            exchange VARCHAR(50) NOT NULL,
            trading_pair VARCHAR(20) NOT NULL,
            side VARCHAR(10) NOT NULL,
            quantity NUMERIC(30,8) NOT NULL,
            price NUMERIC(20,8) NOT NULL,
            value NUMERIC(30,8) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            raw_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(exchange, trading_pair, id)
        );
        
        -- Índices para otimização
        CREATE INDEX IF NOT EXISTS idx_liquidations_exchange_pair_time 
        ON liquidations (exchange, trading_pair, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_liquidations_timestamp 
        ON liquidations (timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_liquidations_value 
        ON liquidations (exchange, trading_pair, value DESC);
        
        CREATE INDEX IF NOT EXISTS idx_liquidations_side_time 
        ON liquidations (exchange, trading_pair, side, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_liquidations_price_time 
        ON liquidations (exchange, trading_pair, price, timestamp);
        
        -- Índice para detecção de clusters
        CREATE INDEX IF NOT EXISTS idx_liquidations_cluster_detection 
        ON liquidations (exchange, trading_pair, timestamp, value);
        
        -- TimescaleDB hypertable (se disponível)
        SELECT create_hypertable('liquidations', 'timestamp', if_not_exists => TRUE);
        """
    
    async def initialize(self) -> None:
        """Inicializa o repository."""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(self._create_table_sql)
        logger.info("Liquidation repository inicializado")
    
    async def save(self, liquidation: Liquidation) -> Liquidation:
        """Salva uma liquidação."""
        sql = """
        INSERT INTO liquidations (
            id, exchange, trading_pair, side, quantity, price, value, timestamp, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (exchange, trading_pair, id)
        DO UPDATE SET
            side = EXCLUDED.side,
            quantity = EXCLUDED.quantity,
            price = EXCLUDED.price,
            value = EXCLUDED.value,
            timestamp = EXCLUDED.timestamp,
            raw_data = EXCLUDED.raw_data
        RETURNING id, created_at
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                sql,
                liquidation.id,
                liquidation.exchange,
                liquidation.trading_pair,
                liquidation.side.value,
                liquidation.quantity,
                liquidation.price,
                liquidation.value,
                liquidation.timestamp,
                liquidation.raw_data
            )
            
            if row:
                liquidation.created_at = row['created_at']
        
        return liquidation
    
    async def save_batch(self, liquidations: List[Liquidation]) -> List[Liquidation]:
        """Salva múltiplas liquidações em lote."""
        if not liquidations:
            return []
        
        sql = """
        INSERT INTO liquidations (
            id, exchange, trading_pair, side, quantity, price, value, timestamp, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (exchange, trading_pair, id)
        DO UPDATE SET
            side = EXCLUDED.side,
            quantity = EXCLUDED.quantity,
            price = EXCLUDED.price,
            value = EXCLUDED.value,
            timestamp = EXCLUDED.timestamp,
            raw_data = EXCLUDED.raw_data
        """
        
        batch_data = [
            (
                liq.id,
                liq.exchange,
                liq.trading_pair,
                liq.side.value,
                liq.quantity,
                liq.price,
                liq.value,
                liq.timestamp,
                liq.raw_data
            )
            for liq in liquidations
        ]
        
        async with self.db_manager.get_connection() as conn:
            await conn.executemany(sql, batch_data)
        
        logger.debug(f"Salvas {len(liquidations)} liquidações em lote")
        return liquidations
    
    async def find_by_id(self, liquidation_id: str) -> Optional[Liquidation]:
        """Busca liquidação por ID."""
        sql = "SELECT * FROM liquidations WHERE id = $1"
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, liquidation_id)
            
            if row:
                return self._row_to_liquidation(row)
        
        return None
    
    async def find_by_query(self, query: LiquidationQuery) -> List[Liquidation]:
        """Busca liquidações por query."""
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
        
        if query.side:
            param_count += 1
            where_conditions.append(f"side = ${param_count}")
            params.append(query.side.value)
        
        if query.start_time:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(query.start_time)
        
        if query.end_time:
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(query.end_time)
        
        if query.min_value:
            param_count += 1
            where_conditions.append(f"value >= ${param_count}")
            params.append(query.min_value)
        
        if query.max_value:
            param_count += 1
            where_conditions.append(f"value <= ${param_count}")
            params.append(query.max_value)
        
        if query.min_price:
            param_count += 1
            where_conditions.append(f"price >= ${param_count}")
            params.append(query.min_price)
        
        if query.max_price:
            param_count += 1
            where_conditions.append(f"price <= ${param_count}")
            params.append(query.max_price)
        
        # Constrói SQL
        sql = "SELECT * FROM liquidations"
        
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
            return [self._row_to_liquidation(row) for row in rows]
    
    async def get_statistics(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[LiquidationStats]:
        """Calcula estatísticas de liquidações para um período."""
        sql = """
        SELECT 
            COUNT(*) as total_liquidations,
            SUM(value) as total_value,
            AVG(value) as avg_liquidation_value,
            MAX(value) as max_liquidation_value,
            COUNT(*) FILTER (WHERE side = 'long') as long_liquidations,
            COUNT(*) FILTER (WHERE side = 'short') as short_liquidations,
            COALESCE(SUM(value) FILTER (WHERE side = 'long'), 0) as long_value,
            COALESCE(SUM(value) FILTER (WHERE side = 'short'), 0) as short_value,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM liquidations
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp >= $3 AND timestamp <= $4
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, exchange, trading_pair, start_time, end_time)
            
            if row and row['total_liquidations'] > 0:
                # Calcula taxa de liquidação por hora
                duration_hours = (end_time - start_time).total_seconds() / 3600
                liquidation_rate_per_hour = row['total_liquidations'] / max(1, duration_hours)
                
                # Determina lado dominante
                long_value = Decimal(str(row['long_value']))
                short_value = Decimal(str(row['short_value']))
                dominant_side = LiquidationSide.LONG if long_value > short_value else LiquidationSide.SHORT
                
                return LiquidationStats(
                    exchange=exchange,
                    trading_pair=trading_pair,
                    period_start=start_time,
                    period_end=end_time,
                    total_liquidations=row['total_liquidations'],
                    total_value=Decimal(str(row['total_value'])),
                    avg_liquidation_value=Decimal(str(row['avg_liquidation_value'])),
                    max_liquidation_value=Decimal(str(row['max_liquidation_value'])),
                    long_liquidations=row['long_liquidations'],
                    short_liquidations=row['short_liquidations'],
                    long_value=long_value,
                    short_value=short_value,
                    liquidation_rate_per_hour=liquidation_rate_per_hour,
                    dominant_side=dominant_side,
                    price_range=(Decimal(str(row['min_price'])), Decimal(str(row['max_price'])))
                )
        
        return None
    
    async def get_liquidation_heatmap(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        price_buckets: int = 50
    ) -> Optional[LiquidationHeatmap]:
        """Gera heatmap de liquidações por faixa de preço."""
        # Primeiro, obtém range de preços
        price_range_sql = """
        SELECT MIN(price) as min_price, MAX(price) as max_price
        FROM liquidations
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp >= $3 AND timestamp <= $4
        """
        
        async with self.db_manager.get_connection() as conn:
            price_row = await conn.fetchrow(
                price_range_sql, exchange, trading_pair, start_time, end_time
            )
            
            if not price_row or not price_row['min_price']:
                return None
            
            min_price = Decimal(str(price_row['min_price']))
            max_price = Decimal(str(price_row['max_price']))
            
            if min_price == max_price:
                return None
            
            # Calcula o tamanho do bucket
            price_step = (max_price - min_price) / price_buckets
            
            # Query para agrupar liquidações por faixa de preço
            heatmap_sql = """
            SELECT 
                floor((price - $5) / $6) as bucket_index,
                COUNT(*) FILTER (WHERE side = 'long') as long_count,
                COUNT(*) FILTER (WHERE side = 'short') as short_count,
                SUM(value) as total_value
            FROM liquidations
            WHERE exchange = $1 AND trading_pair = $2
            AND timestamp >= $3 AND timestamp <= $4
            GROUP BY bucket_index
            ORDER BY bucket_index
            """
            
            heatmap_rows = await conn.fetch(
                heatmap_sql, exchange, trading_pair, start_time, end_time,
                min_price, price_step
            )
            
            # Constrói arrays do heatmap
            price_levels = []
            long_liquidations = []
            short_liquidations = []
            liquidation_values = []
            
            for i in range(price_buckets):
                bucket_price = min_price + (price_step * i)
                price_levels.append(bucket_price)
                
                # Encontra dados para este bucket
                bucket_data = None
                for row in heatmap_rows:
                    if int(row['bucket_index']) == i:
                        bucket_data = row
                        break
                
                if bucket_data:
                    long_liquidations.append(bucket_data['long_count'] or 0)
                    short_liquidations.append(bucket_data['short_count'] or 0)
                    liquidation_values.append(Decimal(str(bucket_data['total_value'] or 0)))
                else:
                    long_liquidations.append(0)
                    short_liquidations.append(0)
                    liquidation_values.append(Decimal('0'))
            
            return LiquidationHeatmap(
                exchange=exchange,
                trading_pair=trading_pair,
                timestamp=datetime.utcnow(),
                price_levels=price_levels,
                long_liquidations=long_liquidations,
                short_liquidations=short_liquidations,
                liquidation_values=liquidation_values
            )
    
    async def detect_liquidation_clusters(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        min_cluster_size: int = 5,
        max_cluster_duration_minutes: int = 5
    ) -> List[LiquidationCluster]:
        """Detecta clusters de liquidações."""
        # Query para encontrar grupos de liquidações próximas no tempo
        sql = """
        WITH liquidation_windows AS (
            SELECT 
                *,
                LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp
            FROM liquidations
            WHERE exchange = $1 AND trading_pair = $2
            AND timestamp >= $3 AND timestamp <= $4
            ORDER BY timestamp
        ),
        cluster_starts AS (
            SELECT 
                *,
                CASE 
                    WHEN prev_timestamp IS NULL 
                         OR (timestamp - prev_timestamp) > interval '$5 minutes'
                    THEN 1 
                    ELSE 0 
                END as is_cluster_start
            FROM liquidation_windows
        ),
        cluster_groups AS (
            SELECT 
                *,
                SUM(is_cluster_start) OVER (ORDER BY timestamp) as cluster_id
            FROM cluster_starts
        )
        SELECT 
            cluster_id,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            COUNT(*) as liquidation_count,
            SUM(value) as total_value,
            AVG(price) as avg_price,
            CASE 
                WHEN SUM(CASE WHEN side = 'long' THEN value ELSE 0 END) > 
                     SUM(CASE WHEN side = 'short' THEN value ELSE 0 END) 
                THEN 'long'
                ELSE 'short'
            END as dominant_side
        FROM cluster_groups
        GROUP BY cluster_id
        HAVING COUNT(*) >= $6
        ORDER BY start_time
        """
        
        clusters = []
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(
                sql, exchange, trading_pair, start_time, end_time,
                max_cluster_duration_minutes, min_cluster_size
            )
            
            for row in rows:
                duration_seconds = int((row['end_time'] - row['start_time']).total_seconds())
                intensity = row['liquidation_count'] / max(1, duration_seconds)
                
                cluster = LiquidationCluster(
                    exchange=exchange,
                    trading_pair=trading_pair,
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    duration_seconds=duration_seconds,
                    liquidation_count=row['liquidation_count'],
                    total_value=Decimal(str(row['total_value'])),
                    dominant_side=LiquidationSide(row['dominant_side']),
                    trigger_price=Decimal(str(row['avg_price'])),
                    intensity=intensity
                )
                clusters.append(cluster)
        
        return clusters
    
    async def get_large_liquidations(
        self,
        exchange: str,
        trading_pair: str,
        min_value: Decimal,
        hours: int = 24
    ) -> List[Liquidation]:
        """Obtém liquidações de grande valor."""
        start_time = datetime.utcnow() - timedelta(hours=hours)
        
        sql = """
        SELECT * FROM liquidations
        WHERE exchange = $1 AND trading_pair = $2
        AND value >= $3 AND timestamp >= $4
        ORDER BY value DESC
        LIMIT 100
        """
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange, trading_pair, min_value, start_time)
            return [self._row_to_liquidation(row) for row in rows]
    
    async def delete_old_liquidations(
        self,
        exchange: str,
        trading_pair: str,
        older_than: datetime
    ) -> int:
        """Remove liquidações antigas."""
        sql = """
        DELETE FROM liquidations 
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp < $3
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.execute(sql, exchange, trading_pair, older_than)
            
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Removidas {deleted_count} liquidações antigas")
            return deleted_count
    
    def _row_to_liquidation(self, row) -> Liquidation:
        """Converte row do banco para entidade Liquidation."""
        return Liquidation(
            id=row['id'],
            exchange=row['exchange'],
            trading_pair=row['trading_pair'],
            side=LiquidationSide(row['side']),
            quantity=Decimal(str(row['quantity'])),
            price=Decimal(str(row['price'])),
            value=Decimal(str(row['value'])),
            timestamp=row['timestamp'],
            raw_data=row['raw_data'],
            created_at=row.get('created_at')
        )
    
    async def get_repository_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas gerais do repository."""
        sql = """
        SELECT 
            COUNT(*) as total_liquidations,
            COUNT(DISTINCT exchange) as total_exchanges,
            COUNT(DISTINCT trading_pair) as total_pairs,
            MIN(timestamp) as oldest_liquidation,
            MAX(timestamp) as newest_liquidation,
            SUM(value) as total_value,
            AVG(value) as avg_value,
            MAX(value) as max_value,
            COUNT(*) FILTER (WHERE side = 'long') as total_long,
            COUNT(*) FILTER (WHERE side = 'short') as total_short
        FROM liquidations
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql)
            
            return {
                'total_liquidations': row['total_liquidations'],
                'total_exchanges': row['total_exchanges'],
                'total_pairs': row['total_pairs'],
                'oldest_liquidation': row['oldest_liquidation'],
                'newest_liquidation': row['newest_liquidation'],
                'total_value': float(row['total_value']) if row['total_value'] else 0,
                'avg_value': float(row['avg_value']) if row['avg_value'] else 0,
                'max_value': float(row['max_value']) if row['max_value'] else 0,
                'total_long': row['total_long'],
                'total_short': row['total_short'],
                'long_short_ratio': row['total_long'] / max(1, row['total_short'])
            }


class LiquidationService:
    """
    Serviço de alto nível para operações com liquidações.
    
    Combina repository com lógica de negócio e análises de risco avançadas.
    """
    
    def __init__(self, repository: LiquidationRepositoryInterface):
        self.repository = repository
    
    async def analyze_liquidation_cascade_risk(
        self,
        exchange: str,
        trading_pair: str,
        current_price: Decimal,
        price_move_percent: float = 5.0,
        hours_lookback: int = 24
    ) -> Dict[str, Any]:
        """Analisa risco de cascata de liquidações."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_lookback)
        
        # Calcula faixas de preço de risco
        price_move = current_price * Decimal(str(price_move_percent / 100))
        lower_risk_price = current_price - price_move
        upper_risk_price = current_price + price_move
        
        # Busca liquidações recentes nessas faixas
        query = LiquidationQuery(
            exchange=exchange,
            trading_pair=trading_pair,
            start_time=start_time,
            end_time=end_time,
            min_price=lower_risk_price,
            max_price=upper_risk_price
        )
        
        recent_liquidations = await self.repository.find_by_query(query)
        
        if not recent_liquidations:
            return {
                'risk_level': 'low',
                'message': 'Dados insuficientes para análise'
            }
        
        # Analisa padrões
        long_liquidations = [liq for liq in recent_liquidations if liq.side == LiquidationSide.LONG]
        short_liquidations = [liq for liq in recent_liquidations if liq.side == LiquidationSide.SHORT]
        
        total_long_value = sum(liq.value for liq in long_liquidations)
        total_short_value = sum(liq.value for liq in short_liquidations)
        
        # Determina concentração de risco
        price_concentrations = {}
        for liq in recent_liquidations:
            price_bucket = int(liq.price / (current_price * Decimal('0.01')))  # Buckets de 1%
            if price_bucket not in price_concentrations:
                price_concentrations[price_bucket] = {'count': 0, 'value': Decimal('0')}
            price_concentrations[price_bucket]['count'] += 1
            price_concentrations[price_bucket]['value'] += liq.value
        
        # Identifica níveis de alta concentração
        high_risk_levels = []
        for bucket, data in price_concentrations.items():
            if data['count'] >= 5 or data['value'] > current_price * 1000:  # Concentração significativa
                bucket_price = current_price * (Decimal('1') + Decimal(str(bucket)) * Decimal('0.01'))
                high_risk_levels.append({
                    'price': float(bucket_price),
                    'liquidation_count': data['count'],
                    'total_value': float(data['value']),
                    'distance_percent': abs(float((bucket_price - current_price) / current_price * 100))
                })
        
        # Determina nível de risco
        total_liquidations = len(recent_liquidations)
        total_value = sum(liq.value for liq in recent_liquidations)
        
        if total_liquidations > 50 and len(high_risk_levels) > 3:
            risk_level = 'high'
        elif total_liquidations > 20 and len(high_risk_levels) > 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'current_price': float(current_price),
            'analysis_period_hours': hours_lookback,
            'total_liquidations': total_liquidations,
            'total_value': float(total_value),
            'long_liquidations': len(long_liquidations),
            'short_liquidations': len(short_liquidations),
            'long_value': float(total_long_value),
            'short_value': float(total_short_value),
            'high_risk_levels': sorted(high_risk_levels, key=lambda x: x['distance_percent']),
            'dominant_side': 'long' if total_long_value > total_short_value else 'short',
            'price_range_analyzed': {
                'lower': float(lower_risk_price),
                'upper': float(upper_risk_price)
            }
        }
    
    async def detect_unusual_liquidation_activity(
        self,
        exchange: str,
        trading_pair: str,
        hours: int = 1,
        threshold_multiplier: float = 3.0
    ) -> Dict[str, Any]:
        """Detecta atividade anômala de liquidações."""
        end_time = datetime.utcnow()
        recent_start = end_time - timedelta(hours=hours)
        baseline_start = end_time - timedelta(hours=hours*24)  # 24x o período para baseline
        
        # Obtém estatísticas do período recente
        recent_stats = await self.repository.get_statistics(
            exchange, trading_pair, recent_start, end_time
        )
        
        # Obtém estatísticas do período baseline
        baseline_stats = await self.repository.get_statistics(
            exchange, trading_pair, baseline_start, recent_start
        )
        
        if not recent_stats or not baseline_stats:
            return {'error': 'Dados insuficientes'}
        
        # Compara métricas
        recent_rate = recent_stats.liquidation_rate_per_hour
        baseline_rate = baseline_stats.liquidation_rate_per_hour
        
        recent_value_rate = float(recent_stats.total_value) / hours
        baseline_value_rate = float(baseline_stats.total_value) / (hours * 24)
        
        # Detecta anomalias
        anomalies = []
        
        if recent_rate > baseline_rate * threshold_multiplier:
            anomalies.append({
                'type': 'high_frequency',
                'metric': 'liquidation_rate',
                'recent_value': recent_rate,
                'baseline_value': baseline_rate,
                'multiplier': recent_rate / max(0.1, baseline_rate)
            })
        
        if recent_value_rate > baseline_value_rate * threshold_multiplier:
            anomalies.append({
                'type': 'high_value',
                'metric': 'value_rate',
                'recent_value': recent_value_rate,
                'baseline_value': baseline_value_rate,
                'multiplier': recent_value_rate / max(0.1, baseline_value_rate)
            })
        
        # Detecta desequilíbrio de lado
        recent_long_ratio = recent_stats.long_liquidations / max(1, recent_stats.total_liquidations)
        baseline_long_ratio = baseline_stats.long_liquidations / max(1, baseline_stats.total_liquidations)
        
        if abs(recent_long_ratio - baseline_long_ratio) > 0.3:  # 30% diferença
            anomalies.append({
                'type': 'side_imbalance',
                'metric': 'long_ratio',
                'recent_value': recent_long_ratio,
                'baseline_value': baseline_long_ratio,
                'deviation': abs(recent_long_ratio - baseline_long_ratio)
            })
        
        return {
            'exchange': exchange,
            'trading_pair': trading_pair,
            'analysis_period_hours': hours,
            'anomalies_detected': len(anomalies),
            'anomalies': anomalies,
            'recent_stats': {
                'liquidation_count': recent_stats.total_liquidations,
                'total_value': float(recent_stats.total_value),
                'rate_per_hour': recent_stats.liquidation_rate_per_hour,
                'dominant_side': recent_stats.dominant_side.value
            },
            'baseline_stats': {
                'liquidation_count': baseline_stats.total_liquidations,
                'total_value': float(baseline_stats.total_value),
                'rate_per_hour': baseline_stats.liquidation_rate_per_hour,
                'dominant_side': baseline_stats.dominant_side.value
            },
            'is_unusual': len(anomalies) > 0
        }
    
    async def generate_liquidation_alerts(
        self,
        exchanges: List[str],
        trading_pairs: List[str],
        large_liquidation_threshold: Decimal = Decimal('100000'),  # $100k
        cluster_threshold: int = 10
    ) -> List[Dict[str, Any]]:
        """Gera alertas baseados em atividade de liquidações."""
        alerts = []
        current_time = datetime.utcnow()
        lookback_time = current_time - timedelta(minutes=30)
        
        for exchange in exchanges:
            for pair in trading_pairs:
                # Alerta para liquidações grandes
                large_liquidations = await self.repository.get_large_liquidations(
                    exchange, pair, large_liquidation_threshold, 1
                )
                
                for liq in large_liquidations:
                    if liq.timestamp >= lookback_time:
                        alerts.append({
                            'type': 'large_liquidation',
                            'exchange': exchange,
                            'trading_pair': pair,
                            'timestamp': liq.timestamp.isoformat(),
                            'side': liq.side.value,
                            'value': float(liq.value),
                            'price': float(liq.price),
                            'severity': 'high' if liq.value > large_liquidation_threshold * 5 else 'medium'
                        })
                
                # Alerta para clusters
                clusters = await self.repository.detect_liquidation_clusters(
                    exchange, pair, lookback_time, current_time, cluster_threshold, 5
                )
                
                for cluster in clusters:
                    alerts.append({
                        'type': 'liquidation_cluster',
                        'exchange': exchange,
                        'trading_pair': pair,
                        'start_time': cluster.start_time.isoformat(),
                        'end_time': cluster.end_time.isoformat(),
                        'liquidation_count': cluster.liquidation_count,
                        'total_value': float(cluster.total_value),
                        'dominant_side': cluster.dominant_side.value,
                        'intensity': cluster.intensity,
                        'severity': 'high' if cluster.liquidation_count > cluster_threshold * 2 else 'medium'
                    })
        
        # Ordena alertas por severidade e tempo
        severity_order = {'high': 3, 'medium': 2, 'low': 1}
        alerts.sort(key=lambda x: (severity_order.get(x['severity'], 0), x.get('timestamp', '')), reverse=True)
        
        return alerts[:50]  # Limita a 50 alertas mais importantes