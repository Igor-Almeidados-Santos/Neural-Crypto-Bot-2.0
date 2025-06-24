"""
Repository para entidades OrderBook.

Este módulo implementa o padrão Repository para persistência de dados de orderbooks,
com otimizações específicas para dados de alta frequência e consultas em tempo real.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json

from data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from common.infrastructure.database.base_repository import BaseRepository

logger = logging.getLogger(__name__)


@dataclass
class OrderBookQuery:
    """Query builder para consultas de orderbooks."""
    exchange: Optional[str] = None
    trading_pair: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_depth: Optional[int] = None
    max_spread_percent: Optional[float] = None
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
            'min_depth': self.min_depth,
            'max_spread_percent': self.max_spread_percent,
            'limit': self.limit,
            'offset': self.offset,
            'order_by': self.order_by,
            'order_direction': self.order_direction
        }


@dataclass
class OrderBookSnapshot:
    """Snapshot simplificado de orderbook para análises."""
    exchange: str
    trading_pair: str
    timestamp: datetime
    best_bid: Decimal
    best_ask: Decimal
    spread: Decimal
    spread_percent: Decimal
    bid_volume_10: Decimal  # Volume total dos 10 melhores bids
    ask_volume_10: Decimal  # Volume total dos 10 melhores asks
    total_bid_levels: int
    total_ask_levels: int
    mid_price: Decimal


@dataclass
class LiquidityMetrics:
    """Métricas de liquidez do orderbook."""
    exchange: str
    trading_pair: str
    timestamp: datetime
    spread_bps: float  # Spread em basis points
    market_depth_1pct: Decimal  # Profundidade de mercado em 1%
    market_depth_5pct: Decimal  # Profundidade de mercado em 5%
    bid_ask_imbalance: float  # Desequilíbrio bid/ask
    price_impact_buy_1k: Decimal  # Impacto de preço para compra de $1k
    price_impact_sell_1k: Decimal  # Impacto de preço para venda de $1k
    effective_spread: Decimal
    quoted_spread: Decimal


class OrderBookRepositoryInterface(ABC):
    """Interface para repository de orderbooks."""
    
    @abstractmethod
    async def save(self, orderbook: OrderBook) -> OrderBook:
        """Salva um orderbook."""
        pass
    
    @abstractmethod
    async def save_batch(self, orderbooks: List[OrderBook]) -> List[OrderBook]:
        """Salva múltiplos orderbooks em lote."""
        pass
    
    @abstractmethod
    async def find_by_id(self, orderbook_id: str) -> Optional[OrderBook]:
        """Busca orderbook por ID."""
        pass
    
    @abstractmethod
    async def find_latest(
        self,
        exchange: str,
        trading_pair: str,
        limit: int = 1
    ) -> List[OrderBook]:
        """Busca orderbooks mais recentes."""
        pass
    
    @abstractmethod
    async def find_by_query(self, query: OrderBookQuery) -> List[OrderBook]:
        """Busca orderbooks por query."""
        pass
    
    @abstractmethod
    async def get_snapshots(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> List[OrderBookSnapshot]:
        """Obtém snapshots periódicos de orderbooks."""
        pass
    
    @abstractmethod
    async def get_liquidity_metrics(
        self,
        exchange: str,
        trading_pair: str,
        timestamp: datetime
    ) -> Optional[LiquidityMetrics]:
        """Calcula métricas de liquidez para um orderbook."""
        pass
    
    @abstractmethod
    async def delete_old_orderbooks(
        self,
        exchange: str,
        trading_pair: str,
        older_than: datetime
    ) -> int:
        """Remove orderbooks antigos."""
        pass
    
    @abstractmethod
    async def get_spread_history(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 1
    ) -> List[Tuple[datetime, Decimal, Decimal]]:
        """Obtém histórico de spreads."""
        pass


class PostgreSQLOrderBookRepository(OrderBookRepositoryInterface, BaseRepository):
    """
    Implementação PostgreSQL do repository de orderbooks.
    
    Otimizada para dados de alta frequência com estrutura híbrida:
    - Dados principais em colunas relacionais
    - Dados detalhados (bids/asks) em JSONB
    """
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        self.table_name = "orderbooks"
        self._create_table_sql = """
        CREATE TABLE IF NOT EXISTS orderbooks (
            id VARCHAR(255) PRIMARY KEY,
            exchange VARCHAR(50) NOT NULL,
            trading_pair VARCHAR(20) NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            best_bid NUMERIC(20,8),
            best_ask NUMERIC(20,8),
            spread NUMERIC(20,8),
            spread_percent NUMERIC(10,4),
            mid_price NUMERIC(20,8),
            total_bid_levels INTEGER DEFAULT 0,
            total_ask_levels INTEGER DEFAULT 0,
            bid_volume_10 NUMERIC(30,8) DEFAULT 0,
            ask_volume_10 NUMERIC(30,8) DEFAULT 0,
            bids JSONB,
            asks JSONB,
            raw_data JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(exchange, trading_pair, timestamp)
        );
        
        -- Índices para otimização
        CREATE INDEX IF NOT EXISTS idx_orderbooks_exchange_pair_time 
        ON orderbooks (exchange, trading_pair, timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_orderbooks_timestamp 
        ON orderbooks (timestamp DESC);
        
        CREATE INDEX IF NOT EXISTS idx_orderbooks_spread 
        ON orderbooks (exchange, trading_pair, spread);
        
        -- Índices GIN para JSONB
        CREATE INDEX IF NOT EXISTS idx_orderbooks_bids_gin 
        ON orderbooks USING GIN (bids);
        
        CREATE INDEX IF NOT EXISTS idx_orderbooks_asks_gin 
        ON orderbooks USING GIN (asks);
        
        -- TimescaleDB hypertable (se disponível)
        SELECT create_hypertable('orderbooks', 'timestamp', if_not_exists => TRUE);
        """
    
    async def initialize(self) -> None:
        """Inicializa o repository."""
        async with self.db_manager.get_connection() as conn:
            await conn.execute(self._create_table_sql)
        logger.info("OrderBook repository inicializado")
    
    async def save(self, orderbook: OrderBook) -> OrderBook:
        """Salva um orderbook."""
        # Pré-calcula métricas para otimização de consultas
        metrics = self._calculate_orderbook_metrics(orderbook)
        
        sql = """
        INSERT INTO orderbooks (
            id, exchange, trading_pair, timestamp,
            best_bid, best_ask, spread, spread_percent, mid_price,
            total_bid_levels, total_ask_levels, bid_volume_10, ask_volume_10,
            bids, asks, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (exchange, trading_pair, timestamp)
        DO UPDATE SET
            best_bid = EXCLUDED.best_bid,
            best_ask = EXCLUDED.best_ask,
            spread = EXCLUDED.spread,
            spread_percent = EXCLUDED.spread_percent,
            mid_price = EXCLUDED.mid_price,
            total_bid_levels = EXCLUDED.total_bid_levels,
            total_ask_levels = EXCLUDED.total_ask_levels,
            bid_volume_10 = EXCLUDED.bid_volume_10,
            ask_volume_10 = EXCLUDED.ask_volume_10,
            bids = EXCLUDED.bids,
            asks = EXCLUDED.asks,
            raw_data = EXCLUDED.raw_data
        RETURNING id, created_at
        """
        
        # Serializa bids e asks para JSONB
        bids_json = [
            {"price": float(level.price), "amount": float(level.amount), "count": level.count}
            for level in orderbook.bids[:50]  # Limita a 50 níveis para otimização
        ]
        
        asks_json = [
            {"price": float(level.price), "amount": float(level.amount), "count": level.count}
            for level in orderbook.asks[:50]  # Limita a 50 níveis para otimização
        ]
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(
                sql,
                orderbook.id,
                orderbook.exchange,
                orderbook.trading_pair,
                orderbook.timestamp,
                metrics['best_bid'],
                metrics['best_ask'],
                metrics['spread'],
                metrics['spread_percent'],
                metrics['mid_price'],
                len(orderbook.bids),
                len(orderbook.asks),
                metrics['bid_volume_10'],
                metrics['ask_volume_10'],
                json.dumps(bids_json),
                json.dumps(asks_json),
                orderbook.raw_data
            )
            
            if row:
                orderbook.created_at = row['created_at']
        
        return orderbook
    
    async def save_batch(self, orderbooks: List[OrderBook]) -> List[OrderBook]:
        """Salva múltiplos orderbooks em lote."""
        if not orderbooks:
            return []
        
        sql = """
        INSERT INTO orderbooks (
            id, exchange, trading_pair, timestamp,
            best_bid, best_ask, spread, spread_percent, mid_price,
            total_bid_levels, total_ask_levels, bid_volume_10, ask_volume_10,
            bids, asks, raw_data
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        ON CONFLICT (exchange, trading_pair, timestamp)
        DO UPDATE SET
            best_bid = EXCLUDED.best_bid,
            best_ask = EXCLUDED.best_ask,
            spread = EXCLUDED.spread,
            spread_percent = EXCLUDED.spread_percent,
            mid_price = EXCLUDED.mid_price,
            total_bid_levels = EXCLUDED.total_bid_levels,
            total_ask_levels = EXCLUDED.total_ask_levels,
            bid_volume_10 = EXCLUDED.bid_volume_10,
            ask_volume_10 = EXCLUDED.ask_volume_10,
            bids = EXCLUDED.bids,
            asks = EXCLUDED.asks,
            raw_data = EXCLUDED.raw_data
        """
        
        batch_data = []
        for orderbook in orderbooks:
            metrics = self._calculate_orderbook_metrics(orderbook)
            
            bids_json = [
                {"price": float(level.price), "amount": float(level.amount), "count": level.count}
                for level in orderbook.bids[:50]
            ]
            
            asks_json = [
                {"price": float(level.price), "amount": float(level.amount), "count": level.count}
                for level in orderbook.asks[:50]
            ]
            
            batch_data.append((
                orderbook.id,
                orderbook.exchange,
                orderbook.trading_pair,
                orderbook.timestamp,
                metrics['best_bid'],
                metrics['best_ask'],
                metrics['spread'],
                metrics['spread_percent'],
                metrics['mid_price'],
                len(orderbook.bids),
                len(orderbook.asks),
                metrics['bid_volume_10'],
                metrics['ask_volume_10'],
                json.dumps(bids_json),
                json.dumps(asks_json),
                orderbook.raw_data
            ))
        
        async with self.db_manager.get_connection() as conn:
            await conn.executemany(sql, batch_data)
        
        logger.debug(f"Salvos {len(orderbooks)} orderbooks em lote")
        return orderbooks
    
    async def find_by_id(self, orderbook_id: str) -> Optional[OrderBook]:
        """Busca orderbook por ID."""
        sql = """
        SELECT * FROM orderbooks WHERE id = $1
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, orderbook_id)
            
            if row:
                return self._row_to_orderbook(row)
        
        return None
    
    async def find_latest(
        self,
        exchange: str,
        trading_pair: str,
        limit: int = 1
    ) -> List[OrderBook]:
        """Busca orderbooks mais recentes."""
        sql = """
        SELECT * FROM orderbooks 
        WHERE exchange = $1 AND trading_pair = $2
        ORDER BY timestamp DESC
        LIMIT $3
        """
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(sql, exchange, trading_pair, limit)
            return [self._row_to_orderbook(row) for row in rows]
    
    async def find_by_query(self, query: OrderBookQuery) -> List[OrderBook]:
        """Busca orderbooks por query."""
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
        
        if query.min_depth:
            param_count += 1
            where_conditions.append(f"(total_bid_levels + total_ask_levels) >= ${param_count}")
            params.append(query.min_depth)
        
        if query.max_spread_percent:
            param_count += 1
            where_conditions.append(f"spread_percent <= ${param_count}")
            params.append(query.max_spread_percent)
        
        # Constrói SQL
        sql = "SELECT * FROM orderbooks"
        
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
            return [self._row_to_orderbook(row) for row in rows]
    
    async def get_snapshots(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval_seconds: int = 60
    ) -> List[OrderBookSnapshot]:
        """Obtém snapshots periódicos de orderbooks."""
        sql = """
        WITH time_buckets AS (
            SELECT generate_series($4, $5, interval '1 second' * $6) as bucket_time
        ),
        orderbook_buckets AS (
            SELECT 
                time_bucket($6 * interval '1 second', timestamp) as bucket,
                exchange, trading_pair,
                FIRST(timestamp, timestamp) as timestamp,
                FIRST(best_bid, timestamp) as best_bid,
                FIRST(best_ask, timestamp) as best_ask,
                FIRST(spread, timestamp) as spread,
                FIRST(spread_percent, timestamp) as spread_percent,
                FIRST(mid_price, timestamp) as mid_price,
                FIRST(bid_volume_10, timestamp) as bid_volume_10,
                FIRST(ask_volume_10, timestamp) as ask_volume_10,
                FIRST(total_bid_levels, timestamp) as total_bid_levels,
                FIRST(total_ask_levels, timestamp) as total_ask_levels
            FROM orderbooks
            WHERE exchange = $1 AND trading_pair = $2
            AND timestamp >= $4 AND timestamp <= $5
            GROUP BY bucket, exchange, trading_pair
        )
        SELECT * FROM orderbook_buckets
        ORDER BY bucket
        """
        
        snapshots = []
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(
                sql, exchange, trading_pair, interval_seconds, start_time, end_time
            )
            
            for row in rows:
                if row['best_bid'] and row['best_ask']:
                    snapshot = OrderBookSnapshot(
                        exchange=row['exchange'],
                        trading_pair=row['trading_pair'],
                        timestamp=row['timestamp'],
                        best_bid=Decimal(str(row['best_bid'])),
                        best_ask=Decimal(str(row['best_ask'])),
                        spread=Decimal(str(row['spread'])),
                        spread_percent=Decimal(str(row['spread_percent'])),
                        bid_volume_10=Decimal(str(row['bid_volume_10'])),
                        ask_volume_10=Decimal(str(row['ask_volume_10'])),
                        total_bid_levels=row['total_bid_levels'],
                        total_ask_levels=row['total_ask_levels'],
                        mid_price=Decimal(str(row['mid_price']))
                    )
                    snapshots.append(snapshot)
        
        return snapshots
    
    async def get_liquidity_metrics(
        self,
        exchange: str,
        trading_pair: str,
        timestamp: datetime
    ) -> Optional[LiquidityMetrics]:
        """Calcula métricas de liquidez para um orderbook."""
        # Busca orderbook mais próximo do timestamp
        sql = """
        SELECT * FROM orderbooks
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp <= $3
        ORDER BY timestamp DESC
        LIMIT 1
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql, exchange, trading_pair, timestamp)
            
            if not row:
                return None
            
            orderbook = self._row_to_orderbook(row)
            return self._calculate_liquidity_metrics(orderbook)
    
    async def delete_old_orderbooks(
        self,
        exchange: str,
        trading_pair: str,
        older_than: datetime
    ) -> int:
        """Remove orderbooks antigos."""
        sql = """
        DELETE FROM orderbooks 
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp < $3
        """
        
        async with self.db_manager.get_connection() as conn:
            result = await conn.execute(sql, exchange, trading_pair, older_than)
            
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Removidos {deleted_count} orderbooks antigos")
            return deleted_count
    
    async def get_spread_history(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 1
    ) -> List[Tuple[datetime, Decimal, Decimal]]:
        """Obtém histórico de spreads."""
        sql = """
        SELECT 
            time_bucket($5 * interval '1 minute', timestamp) as bucket,
            AVG(spread) as avg_spread,
            AVG(spread_percent) as avg_spread_percent
        FROM orderbooks
        WHERE exchange = $1 AND trading_pair = $2
        AND timestamp >= $3 AND timestamp <= $4
        GROUP BY bucket
        ORDER BY bucket
        """
        
        spreads = []
        
        async with self.db_manager.get_connection() as conn:
            rows = await conn.fetch(
                sql, exchange, trading_pair, start_time, end_time, interval_minutes
            )
            
            for row in rows:
                spreads.append((
                    row['bucket'],
                    Decimal(str(row['avg_spread'])),
                    Decimal(str(row['avg_spread_percent']))
                ))
        
        return spreads
    
    def _row_to_orderbook(self, row) -> OrderBook:
        """Converte row do banco para entidade OrderBook."""
        # Deserializa bids e asks
        bids = []
        if row['bids']:
            bids_data = json.loads(row['bids']) if isinstance(row['bids'], str) else row['bids']
            for bid_data in bids_data:
                bids.append(OrderBookLevel(
                    price=Decimal(str(bid_data['price'])),
                    amount=Decimal(str(bid_data['amount'])),
                    count=bid_data.get('count')
                ))
        
        asks = []
        if row['asks']:
            asks_data = json.loads(row['asks']) if isinstance(row['asks'], str) else row['asks']
            for ask_data in asks_data:
                asks.append(OrderBookLevel(
                    price=Decimal(str(ask_data['price'])),
                    amount=Decimal(str(ask_data['amount'])),
                    count=ask_data.get('count')
                ))
        
        return OrderBook(
            id=row['id'],
            exchange=row['exchange'],
            trading_pair=row['trading_pair'],
            timestamp=row['timestamp'],
            bids=bids,
            asks=asks,
            raw_data=row['raw_data'],
            created_at=row.get('created_at')
        )
    
    def _calculate_orderbook_metrics(self, orderbook: OrderBook) -> Dict[str, Any]:
        """Calcula métricas pré-computadas do orderbook."""
        metrics = {
            'best_bid': None,
            'best_ask': None,
            'spread': None,
            'spread_percent': None,
            'mid_price': None,
            'bid_volume_10': Decimal('0'),
            'ask_volume_10': Decimal('0')
        }
        
        if orderbook.bids:
            metrics['best_bid'] = orderbook.bids[0].price
            # Calcula volume dos 10 melhores bids
            top_bids = orderbook.bids[:10]
            metrics['bid_volume_10'] = sum(level.amount for level in top_bids)
        
        if orderbook.asks:
            metrics['best_ask'] = orderbook.asks[0].price
            # Calcula volume dos 10 melhores asks
            top_asks = orderbook.asks[:10]
            metrics['ask_volume_10'] = sum(level.amount for level in top_asks)
        
        if metrics['best_bid'] and metrics['best_ask']:
            metrics['spread'] = metrics['best_ask'] - metrics['best_bid']
            metrics['mid_price'] = (metrics['best_bid'] + metrics['best_ask']) / 2
            
            if metrics['mid_price'] > 0:
                metrics['spread_percent'] = (metrics['spread'] / metrics['mid_price']) * 100
        
        return metrics
    
    def _calculate_liquidity_metrics(self, orderbook: OrderBook) -> LiquidityMetrics:
        """Calcula métricas avançadas de liquidez."""
        best_bid = orderbook.bids[0].price if orderbook.bids else Decimal('0')
        best_ask = orderbook.asks[0].price if orderbook.asks else Decimal('0')
        
        if best_bid == 0 or best_ask == 0:
            # Retorna métricas vazias se não houver dados suficientes
            return LiquidityMetrics(
                exchange=orderbook.exchange,
                trading_pair=orderbook.trading_pair,
                timestamp=orderbook.timestamp,
                spread_bps=0,
                market_depth_1pct=Decimal('0'),
                market_depth_5pct=Decimal('0'),
                bid_ask_imbalance=0,
                price_impact_buy_1k=Decimal('0'),
                price_impact_sell_1k=Decimal('0'),
                effective_spread=Decimal('0'),
                quoted_spread=Decimal('0')
            )
        
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        spread_bps = float((spread / mid_price) * 10000)
        
        # Calcula profundidade de mercado
        market_depth_1pct = self._calculate_market_depth(orderbook, 0.01)
        market_depth_5pct = self._calculate_market_depth(orderbook, 0.05)
        
        # Calcula desequilíbrio bid/ask
        bid_volume = sum(level.amount for level in orderbook.bids[:10])
        ask_volume = sum(level.amount for level in orderbook.asks[:10])
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            bid_ask_imbalance = float((bid_volume - ask_volume) / total_volume)
        else:
            bid_ask_imbalance = 0
        
        # Calcula impacto de preço para $1000
        price_impact_buy_1k = self._calculate_price_impact(orderbook.asks, Decimal('1000'), mid_price)
        price_impact_sell_1k = self._calculate_price_impact(orderbook.bids, Decimal('1000'), mid_price)
        
        return LiquidityMetrics(
            exchange=orderbook.exchange,
            trading_pair=orderbook.trading_pair,
            timestamp=orderbook.timestamp,
            spread_bps=spread_bps,
            market_depth_1pct=market_depth_1pct,
            market_depth_5pct=market_depth_5pct,
            bid_ask_imbalance=bid_ask_imbalance,
            price_impact_buy_1k=price_impact_buy_1k,
            price_impact_sell_1k=price_impact_sell_1k,
            effective_spread=spread,
            quoted_spread=spread
        )
    
    def _calculate_market_depth(self, orderbook: OrderBook, percent: float) -> Decimal:
        """Calcula profundidade de mercado para uma porcentagem de preço."""
        if not orderbook.bids or not orderbook.asks:
            return Decimal('0')
        
        mid_price = orderbook.mid_price
        if mid_price == 0:
            return Decimal('0')
        
        price_threshold = mid_price * Decimal(str(percent))
        
        # Calcula volume de bids dentro do threshold
        bid_volume = Decimal('0')
        for level in orderbook.bids:
            if mid_price - level.price <= price_threshold:
                bid_volume += level.amount
            else:
                break
        
        # Calcula volume de asks dentro do threshold
        ask_volume = Decimal('0')
        for level in orderbook.asks:
            if level.price - mid_price <= price_threshold:
                ask_volume += level.amount
            else:
                break
        
        return bid_volume + ask_volume
    
    def _calculate_price_impact(
        self,
        levels: List[OrderBookLevel],
        dollar_amount: Decimal,
        mid_price: Decimal
    ) -> Decimal:
        """Calcula impacto de preço para um valor em dólares."""
        if not levels or mid_price == 0:
            return Decimal('0')
        
        remaining_amount = dollar_amount
        total_shares = Decimal('0')
        weighted_price = Decimal('0')
        
        for level in levels:
            level_value = level.price * level.amount
            
            if level_value >= remaining_amount:
                # Usa apenas parte deste nível
                shares_needed = remaining_amount / level.price
                total_shares += shares_needed
                weighted_price += level.price * shares_needed
                break
            else:
                # Usa todo este nível
                total_shares += level.amount
                weighted_price += level.price * level.amount
                remaining_amount -= level_value
        
        if total_shares == 0:
            return Decimal('0')
        
        avg_execution_price = weighted_price / total_shares
        price_impact = abs(avg_execution_price - mid_price) / mid_price
        
        return price_impact
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas do repository."""
        sql = """
        SELECT 
            COUNT(*) as total_orderbooks,
            COUNT(DISTINCT exchange) as total_exchanges,
            COUNT(DISTINCT trading_pair) as total_pairs,
            MIN(timestamp) as oldest_orderbook,
            MAX(timestamp) as newest_orderbook,
            AVG(spread_percent) as avg_spread_percent,
            AVG(total_bid_levels + total_ask_levels) as avg_depth
        FROM orderbooks
        """
        
        async with self.db_manager.get_connection() as conn:
            row = await conn.fetchrow(sql)
            
            return {
                'total_orderbooks': row['total_orderbooks'],
                'total_exchanges': row['total_exchanges'],
                'total_pairs': row['total_pairs'],
                'oldest_orderbook': row['oldest_orderbook'],
                'newest_orderbook': row['newest_orderbook'],
                'avg_spread_percent': float(row['avg_spread_percent'] or 0),
                'avg_depth': float(row['avg_depth'] or 0)
            }


class OrderBookService:
    """
    Serviço de alto nível para operações com orderbooks.
    
    Combina repository com lógica de negócio e análises avançadas.
    """
    
    def __init__(self, repository: OrderBookRepositoryInterface):
        self.repository = repository
    
    async def analyze_liquidity_trends(
        self,
        exchange: str,
        trading_pair: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Analisa tendências de liquidez."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Obtém snapshots de cada hora
        snapshots = await self.repository.get_snapshots(
            exchange, trading_pair, start_time, end_time, 3600  # 1 hora
        )
        
        if not snapshots:
            return {'error': 'Dados insuficientes'}
        
        # Calcula estatísticas
        spreads = [float(s.spread_percent) for s in snapshots]
        depths = [float(s.bid_volume_10 + s.ask_volume_10) for s in snapshots]
        
        return {
            'period_hours': hours,
            'total_snapshots': len(snapshots),
            'spread_stats': {
                'avg': sum(spreads) / len(spreads),
                'min': min(spreads),
                'max': max(spreads),
                'current': spreads[-1] if spreads else 0
            },
            'depth_stats': {
                'avg': sum(depths) / len(depths),
                'min': min(depths),
                'max': max(depths),
                'current': depths[-1] if depths else 0
            },
            'timestamps': [s.timestamp.isoformat() for s in snapshots]
        }
    
    async def detect_liquidity_events(
        self,
        exchange: str,
        trading_pair: str,
        hours: int = 1,
        spread_threshold: float = 2.0,  # 2x spread normal
        depth_threshold: float = 0.5   # 50% redução na profundidade
    ) -> List[Dict[str, Any]]:
        """Detecta eventos de liquidez anômalos."""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        snapshots = await self.repository.get_snapshots(
            exchange, trading_pair, start_time, end_time, 60  # 1 minuto
        )
        
        if len(snapshots) < 10:
            return []
        
        # Calcula métricas de baseline
        baseline_spread = sum(float(s.spread_percent) for s in snapshots[:10]) / 10
        baseline_depth = sum(float(s.bid_volume_10 + s.ask_volume_10) for s in snapshots[:10]) / 10
        
        events = []
        
        for snapshot in snapshots[10:]:
            current_spread = float(snapshot.spread_percent)
            current_depth = float(snapshot.bid_volume_10 + snapshot.ask_volume_10)
            
            # Detecta spread anômalo
            if current_spread > baseline_spread * spread_threshold:
                events.append({
                    'type': 'high_spread',
                    'timestamp': snapshot.timestamp.isoformat(),
                    'spread_percent': current_spread,
                    'baseline_spread': baseline_spread,
                    'multiplier': current_spread / baseline_spread
                })
            
            # Detecta baixa liquidez
            if current_depth < baseline_depth * depth_threshold:
                events.append({
                    'type': 'low_liquidity',
                    'timestamp': snapshot.timestamp.isoformat(),
                    'depth': current_depth,
                    'baseline_depth': baseline_depth,
                    'reduction_percent': (1 - current_depth / baseline_depth) * 100
                })
        
        return events