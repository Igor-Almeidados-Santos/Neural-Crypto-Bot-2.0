"""
Repository para entidades Candle.

Este módulo implementa o padrão Repository para persistência de dados de candles,
fornecendo uma interface consistente para operações CRUD e consultas complexas.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.common.infrastructure.database.base_repository import BaseRepository
from src.common.domain.base_entity import BaseEntity

logger = logging.getLogger(__name__)


@dataclass
class CandleQuery:
    """Query builder para consultas de candles."""
    exchange: Optional[str] = None
    trading_pair: Optional[str] = None
    timeframe: Optional[TimeFrame] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: str = "timestamp"
    order_direction: str = "ASC"
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'exchange': self.exchange,
            'trading_pair': self.trading_pair,
            'timeframe': self.timeframe.value if self.timeframe else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'limit': self.limit,
            'offset': self.offset,
            'order_by': self.order_by,
            'order_direction': self.order_direction
        }


@dataclass
class CandleAggregation:
    """Resultado de agregação de candles."""
    exchange: str
    trading_pair: str
    timeframe: TimeFrame
    period_start: datetime
    period_end: datetime
    candle_count: int
    first_price: Decimal
    last_price: Decimal
    highest_price: Decimal
    lowest_price: Decimal
    total_volume: Decimal
    total_trades: int
    price_change: Decimal
    price_change_percent: Decimal
    volume_weighted_average_price: Decimal


class CandleRepositoryInterface(ABC):
    """Interface para repository de candles."""
    
    @abstractmethod
    async def save(self, candle: Candle) -> Candle:
        """Salva um candle."""
        pass
    
    @abstractmethod
    async def save_batch(self, candles: List[Candle]) -> List[Candle]:
        """Salva múltiplos candles em lote."""
        pass
    
    @abstractmethod
    async def find_by_id(self, candle_id: str) -> Optional[Candle]:
        """Busca candle por ID."""
        pass
    
    @abstractmethod
    async def find_by_key(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        timestamp: datetime
    ) -> Optional[Candle]:
        """Busca candle por chave natural."""
        pass
    
    @abstractmethod
    async def find_by_query(self, query: CandleQuery) -> List[Candle]:
        """Busca candles por query."""
        pass
    
    @abstractmethod
    async def find_latest(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        limit: int = 1
    ) -> List[Candle]:
        """Busca candles mais recentes."""
        pass
    
    @abstractmethod
    async def find_in_range(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Busca candles em um período."""
        pass
    
    @abstractmethod
    async def get_aggregation(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[CandleAggregation]:
        """Obtém agregação de candles para um período."""
        pass
    
    @abstractmethod
    async def delete_old_candles(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        older_than: datetime
    ) -> int:
        """Remove candles antigos."""
        pass
    
    @abstractmethod
    async def get_available_pairs(self, exchange: str) -> List[str]:
        """Obtém pares disponíveis para uma exchange."""
        pass
    
    @abstractmethod
    async def get_timeframe_coverage(
        self,
        exchange: str,
        trading_pair: str
    ) -> Dict[TimeFrame, Tuple[datetime, datetime]]:
        """Obtém cobertura de timeframes para um par."""
        pass


class PostgreSQLCandleRepository(CandleRepositoryInterface, BaseRepository):
    """
    Implementação PostgreSQL do repository de candles.
    
    Utiliza TimescaleDB para otimização de séries temporais.
    """
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        self.table_name = "candles"
        self._create_table_sql = """
        CREATE TABLE IF NOT EXISTS candles (
            id VARCHAR(255) PRIMARY KEY,
            exchange VARCHAR(50) NOT NULL,
            trading_pair VARCHAR(20) NOT NULL,
            timeframe VARCHAR(10) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open NUMERIC(20,8) NOT NULL,
            high NUMERIC(20,8) NOT NULL,
            low NUMERIC(20,8) NOT NULL,
            close NUMERIC(20,8) NOT NULL,
            volume NUMERIC(30,8) NOT NULL,
            trades INTEGER DEFAULT 0,
            raw_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(exchange, trading_pair, timeframe, timestamp)
        );
        
        -- Índices para otimização
        CREATE INDEX IF NOT EXISTS idx_candles_exchange_pair_timeframe_time 
        ON candles (exchange, trading_pair, timeframe, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_candles_timestamp 
        ON candles (timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_candles_exchange_pair 
        ON candles (exchange, trading_pair);
        
        -- TimescaleDB hypertable (se disponível)
        SELECT create_hypertable('candles', 'timestamp', if_not_exists => TRUE);
        """
    
    async def initialize(self) -> None:
        """Inicializa o repository."""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(self._create_table_sql)
        logger.info("Candle repository inicializado")
    
    async def save(self, candle: Candle) -> Candle:
        """Salva um candle."""
        sql = """
        INSERT INTO candles (
            id, exchange, trading_pair, timeframe, timestamp,
            open, high, low, close, volume, trades, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (exchange, trading_pair, timeframe, timestamp)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            trades = EXCLUDED.trades,
            raw_data = EXCLUDED.raw_data,
            updated_at = NOW()
        RETURNING id, created_at, updated_at
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                sql,
                candle.id,
                candle.exchange,
                candle.trading_pair,
                candle.timeframe.value,
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.trades,
                candle.raw_data
            )
            
            if row:
                candle.created_at = row['created_at']
                candle.updated_at = row['updated_at']
        
        return candle
    
    async def save_batch(self, candles: List[Candle]) -> List[Candle]:
        """Salva múltiplos candles em lote."""
        if not candles:
            return []
        
        sql = """
        INSERT INTO candles (
            id, exchange, trading_pair, timeframe, timestamp,
            open, high, low, close, volume, trades, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (exchange, trading_pair, timeframe, timestamp)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            trades = EXCLUDED.trades,
            raw_data = EXCLUDED.raw_data,
            updated_at = NOW()
        """
        
        batch_data = [
            (
                candle.id,
                candle.exchange,
                candle.trading_pair,
                candle.timeframe.value,
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
                candle.trades,
                candle.raw_data
            )
            for candle in candles
        ]
        
        async with self.db_manager.get_connection() as conn:
            await conn.executemany(sql, batch_data)
        
        logger.debug(f"Salvos {len(candles)} candles em lote")
        return candles
    
    async def find_by_id(self, candle_id: str) -> Optional[Candle]:
        """Busca candle por ID."""
        sql = """
        SELECT * FROM candles WHERE id = $1
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, candle_id)
            
            if row:
                return self._row_to_candle(row)
        
        return None
    
    async def find_by_key(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        timestamp: datetime
    ) -> Optional[Candle]:
        """Busca candle por chave natural."""
        sql = """
        SELECT * FROM candles 
        WHERE exchange = $1 AND trading_pair = $2 
        AND timeframe = $3 AND timestamp = $4
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                sql, exchange, trading_pair, timeframe.value, timestamp
            )
            
            if row:
                return self._row_to_candle(row)
        
        return None
    
    async def find_by_query(self, query: CandleQuery) -> List[Candle]:
        """Busca candles por query."""
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
        
        if query.timeframe:
            param_count += 1
            where_conditions.append(f"timeframe = ${param_count}")
            params.append(query.timeframe.value)
        
        if query.start_time:
            param_count += 1
            where_conditions.append(f"timestamp >= ${param_count}")
            params.append(query.start_time)
        
        if query.end_time:
            param_count += 1
            where_conditions.append(f"timestamp <= ${param_count}")
            params.append(query.end_time)
        
        # Constrói SQL
        sql = "SELECT * FROM candles"
        
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
            return [self._row_to_candle(row) for row in rows]
    
    async def find_latest(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        limit: int = 1
    ) -> List[Candle]:
        """Busca candles mais recentes."""
        sql = """
        SELECT * FROM candles 
        WHERE exchange = $1 AND trading_pair = $2 AND timeframe = $3
        ORDER BY timestamp DESC
        LIMIT $4
        """
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange, trading_pair, timeframe.value, limit)
            return [self._row_to_candle(row) for row in rows]
    
    async def find_in_range(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> List[Candle]:
        """Busca candles em um período."""
        sql = """
        SELECT * FROM candles 
        WHERE exchange = $1 AND trading_pair = $2 AND timeframe = $3
        AND timestamp >= $4 AND timestamp <= $5
        ORDER BY timestamp ASC
        """
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(
                sql, exchange, trading_pair, timeframe.value, start_time, end_time
            )
            return [self._row_to_candle(row) for row in rows]
    
    async def get_aggregation(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> Optional[CandleAggregation]:
        """Obtém agregação de candles para um período."""
        sql = """
        SELECT 
            COUNT(*) as candle_count,
            FIRST(open, timestamp) as first_price,
            LAST(close, timestamp) as last_price,
            MAX(high) as highest_price,
            MIN(low) as lowest_price,
            SUM(volume) as total_volume,
            SUM(trades) as total_trades,
            SUM(close * volume) / NULLIF(SUM(volume), 0) as vwap
        FROM candles
        WHERE exchange = $1 AND trading_pair = $2 AND timeframe = $3
        AND timestamp >= $4 AND timestamp <= $5
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                sql, exchange, trading_pair, timeframe.value, start_time, end_time
            )
            
            if row and row['candle_count'] > 0:
                first_price = Decimal(str(row['first_price']))
                last_price = Decimal(str(row['last_price']))
                price_change = last_price - first_price
                price_change_percent = (price_change / first_price * 100) if first_price != 0 else Decimal('0')
                
                return CandleAggregation(
                    exchange=exchange,
                    trading_pair=trading_pair,
                    timeframe=timeframe,
                    period_start=start_time,
                    period_end=end_time,
                    candle_count=row['candle_count'],
                    first_price=first_price,
                    last_price=last_price,
                    highest_price=Decimal(str(row['highest_price'])),
                    lowest_price=Decimal(str(row['lowest_price'])),
                    total_volume=Decimal(str(row['total_volume'])),
                    total_trades=row['total_trades'] or 0,
                    price_change=price_change,
                    price_change_percent=price_change_percent,
                    volume_weighted_average_price=Decimal(str(row['vwap'] or 0))
                )
        
        return None
    
    async def delete_old_candles(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        older_than: datetime
    ) -> int:
        """Remove candles antigos."""
        sql = """
        DELETE FROM candles 
        WHERE exchange = $1 AND trading_pair = $2 AND timeframe = $3
        AND timestamp < $4
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.execute(
                sql, exchange, trading_pair, timeframe.value, older_than
            )
            
            # Extrai número de linhas afetadas
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Removidos {deleted_count} candles antigos")
            return deleted_count
    
    async def get_available_pairs(self, exchange: str) -> List[str]:
        """Obtém pares disponíveis para uma exchange."""
        sql = """
        SELECT DISTINCT trading_pair 
        FROM candles 
        WHERE exchange = $1
        ORDER BY trading_pair
        """
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange)
            return [row['trading_pair'] for row in rows]
    
    async def get_timeframe_coverage(
        self,
        exchange: str,
        trading_pair: str
    ) -> Dict[TimeFrame, Tuple[datetime, datetime]]:
        """Obtém cobertura de timeframes para um par."""
        sql = """
        SELECT 
            timeframe,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time
        FROM candles
        WHERE exchange = $1 AND trading_pair = $2
        GROUP BY timeframe
        """
        
        coverage = {}
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange, trading_pair)
            
            for row in rows:
                try:
                    timeframe = TimeFrame(row['timeframe'])
                    coverage[timeframe] = (row['start_time'], row['end_time'])
                except ValueError:
                    # Ignora timeframes desconhecidos
                    continue
        
        return coverage
    
    def _row_to_candle(self, row) -> Candle:
        """Converte row do banco para entidade Candle."""
        return Candle(
            id=row['id'],
            exchange=row['exchange'],
            trading_pair=row['trading_pair'],
            timestamp=row['timestamp'],
            timeframe=TimeFrame(row['timeframe']),
            open=Decimal(str(row['open'])),
            high=Decimal(str(row['high'])),
            low=Decimal(str(row['low'])),
            close=Decimal(str(row['close'])),
            volume=Decimal(str(row['volume'])),
            trades=row['trades'] or 0,
            raw_data=row['raw_data'],
            created_at=row.get('created_at'),
            updated_at=row.get('updated_at')
        )
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas do repository."""
        sql = """
        SELECT 
            COUNT(*) as total_candles,
            COUNT(DISTINCT exchange) as total_exchanges,
            COUNT(DISTINCT trading_pair) as total_pairs,
            COUNT(DISTINCT timeframe) as total_timeframes,
            MIN(timestamp) as oldest_candle,
            MAX(timestamp) as newest_candle,
            AVG(volume) as avg_volume
        FROM candles
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql)
            
            return {
                'total_candles': row['total_candles'],
                'total_exchanges': row['total_exchanges'],
                'total_pairs': row['total_pairs'],
                'total_timeframes': row['total_timeframes'],
                'oldest_candle': row['oldest_candle'],
                'newest_candle': row['newest_candle'],
                'avg_volume': float(row['avg_volume']) if row['avg_volume'] else 0
            }
    
    async def cleanup_duplicates(self) -> int:
        """Remove candles duplicados baseado na chave natural."""
        sql = """
        DELETE FROM candles
        WHERE id NOT IN (
            SELECT DISTINCT ON (exchange, trading_pair, timeframe, timestamp) id
            FROM candles
            ORDER BY exchange, trading_pair, timeframe, timestamp, created_at DESC
        )
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.execute(sql)
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Removidos {deleted_count} candles duplicados")
            return deleted_count


class CandleService:
    """
    Serviço de alto nível para operações com candles.
    
    Combina repository com lógica de negócio adicional.
    """
    
    def __init__(self, repository: CandleRepositoryInterface):
        self.repository = repository
    
    async def save_candles_with_validation(self, candles: List[Candle]) -> List[Candle]:
        """Salva candles com validação."""
        valid_candles = []
        
        for candle in candles:
            if self._validate_candle(candle):
                valid_candles.append(candle)
            else:
                logger.warning(f"Candle inválido ignorado: {candle.id}")
        
        if valid_candles:
            return await self.repository.save_batch(valid_candles)
        
        return []
    
    async def get_missing_candles(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> List[datetime]:
        """Identifica candles faltantes em um período."""
        existing_candles = await self.repository.find_in_range(
            exchange, trading_pair, timeframe, start_time, end_time
        )
        
        existing_timestamps = {candle.timestamp for candle in existing_candles}
        
        # Gera timestamps esperados baseado no timeframe
        expected_timestamps = self._generate_expected_timestamps(
            timeframe, start_time, end_time
        )
        
        missing_timestamps = [
            ts for ts in expected_timestamps 
            if ts not in existing_timestamps
        ]
        
        return sorted(missing_timestamps)
    
    async def fill_gaps(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime,
        fill_method: str = "forward_fill"
    ) -> List[Candle]:
        """Preenche lacunas nos dados de candles."""
        candles = await self.repository.find_in_range(
            exchange, trading_pair, timeframe, start_time, end_time
        )
        
        if not candles:
            return []
        
        # Ordena candles por timestamp
        candles.sort(key=lambda x: x.timestamp)
        
        filled_candles = []
        expected_timestamps = self._generate_expected_timestamps(
            timeframe, start_time, end_time
        )
        
        candle_dict = {candle.timestamp: candle for candle in candles}
        last_candle = None
        
        for timestamp in expected_timestamps:
            if timestamp in candle_dict:
                candle = candle_dict[timestamp]
                filled_candles.append(candle)
                last_candle = candle
            elif last_candle and fill_method == "forward_fill":
                # Cria candle sintético usando forward fill
                synthetic_candle = self._create_synthetic_candle(
                    last_candle, timestamp
                )
                filled_candles.append(synthetic_candle)
        
        return filled_candles
    
    def _validate_candle(self, candle: Candle) -> bool:
        """Valida dados do candle."""
        if not candle.exchange or not candle.trading_pair:
            return False
        
        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
            return False
        
        if candle.high < max(candle.open, candle.close):
            return False
        
        if candle.low > min(candle.open, candle.close):
            return False
        
        if candle.volume < 0:
            return False
        
        return True
    
    def _generate_expected_timestamps(
        self,
        timeframe: TimeFrame,
        start_time: datetime,
        end_time: datetime
    ) -> List[datetime]:
        """Gera timestamps esperados para um timeframe."""
        timestamps = []
        current_time = start_time
        
        # Define incremento baseado no timeframe
        if timeframe == TimeFrame.MINUTE_1:
            delta = timedelta(minutes=1)
        elif timeframe == TimeFrame.MINUTE_3:
            delta = timedelta(minutes=3)
        elif timeframe == TimeFrame.MINUTE_5:
            delta = timedelta(minutes=5)
        elif timeframe == TimeFrame.MINUTE_15:
            delta = timedelta(minutes=15)
        elif timeframe == TimeFrame.MINUTE_30:
            delta = timedelta(minutes=30)
        elif timeframe == TimeFrame.HOUR_1:
            delta = timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_4:
            delta = timedelta(hours=4)
        elif timeframe == TimeFrame.DAY_1:
            delta = timedelta(days=1)
        else:
            delta = timedelta(minutes=1)  # Fallback
        
        while current_time <= end_time:
            timestamps.append(current_time)
            current_time += delta
        
        return timestamps
    
    def _create_synthetic_candle(self, base_candle: Candle, timestamp: datetime) -> Candle:
        """Cria candle sintético para preenchimento de lacuna."""
        return Candle(
            exchange=base_candle.exchange,
            trading_pair=base_candle.trading_pair,
            timestamp=timestamp,
            timeframe=base_candle.timeframe,
            open=base_candle.close,  # Usa close anterior como open
            high=base_candle.close,
            low=base_candle.close,
            close=base_candle.close,
            volume=Decimal('0'),  # Volume zero para candle sintético
            trades=0
        )