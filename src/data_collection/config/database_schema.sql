"""
Database Schema for Data Collection Module

Creates all necessary tables, indexes, and constraints for storing
cryptocurrency market data with optimal performance.
"""

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS metadata;

-- Set search path
SET search_path TO market_data, metadata, public;

-- ==============================================================================
-- CANDLES TABLE (OHLCV DATA)
-- ==============================================================================

CREATE TABLE IF NOT EXISTS candles (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8),
    quote_volume DECIMAL(20,8),
    trade_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT candles_unique UNIQUE (exchange, trading_pair, timeframe, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('candles', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_candles_exchange_pair ON candles (exchange, trading_pair);
CREATE INDEX IF NOT EXISTS idx_candles_timeframe ON candles (timeframe);
CREATE INDEX IF NOT EXISTS idx_candles_timestamp ON candles (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_candles_volume ON candles (volume DESC) WHERE volume IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_candles_metadata ON candles USING GIN (metadata);

-- ==============================================================================
-- ORDER BOOKS TABLE
-- ==============================================================================

CREATE TABLE IF NOT EXISTS orderbooks (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    best_bid DECIMAL(20,8),
    best_ask DECIMAL(20,8),
    bid_count INTEGER DEFAULT 0,
    ask_count INTEGER DEFAULT 0,
    total_bid_volume DECIMAL(20,8),
    total_ask_volume DECIMAL(20,8),
    spread DECIMAL(20,8),
    spread_percentage DECIMAL(10,6),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT orderbooks_unique UNIQUE (exchange, trading_pair, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('orderbooks', 'timestamp', chunk_time_interval => INTERVAL '6 hours');

-- Order book entries (bids and asks)
CREATE TABLE IF NOT EXISTS orderbook_entries (
    id BIGSERIAL PRIMARY KEY,
    orderbook_id BIGINT NOT NULL REFERENCES orderbooks(id) ON DELETE CASCADE,
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    order_count INTEGER,
    side VARCHAR(3) NOT NULL CHECK (side IN ('bid', 'ask')),
    
    CONSTRAINT orderbook_entries_unique UNIQUE (orderbook_id, price, side)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_orderbooks_exchange_pair ON orderbooks (exchange, trading_pair);
CREATE INDEX IF NOT EXISTS idx_orderbooks_timestamp ON orderbooks (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_orderbook_entries_orderbook_id ON orderbook_entries (orderbook_id);
CREATE INDEX IF NOT EXISTS idx_orderbook_entries_price_side ON orderbook_entries (price, side);

-- ==============================================================================
-- TRADES TABLE
-- ==============================================================================

CREATE TABLE IF NOT EXISTS trades (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    trade_id VARCHAR(100),
    price DECIMAL(20,8) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL,
    side VARCHAR(4) NOT NULL CHECK (side IN ('buy', 'sell')),
    taker_order_id VARCHAR(100),
    maker_order_id VARCHAR(100),
    fee DECIMAL(20,8),
    fee_currency VARCHAR(10),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT trades_unique UNIQUE (exchange, trading_pair, trade_id)
);

-- Convert to hypertable
SELECT create_hypertable('trades', 'timestamp', chunk_time_interval => INTERVAL '1 hour');

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_trades_exchange_pair ON trades (exchange, trading_pair);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_price ON trades (price);
CREATE INDEX IF NOT EXISTS idx_trades_quantity ON trades (quantity DESC);
CREATE INDEX IF NOT EXISTS idx_trades_side ON trades (side);

-- ==============================================================================
-- FUNDING RATES TABLE
-- ==============================================================================

CREATE TABLE IF NOT EXISTS funding_rates (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    funding_rate DECIMAL(12,8) NOT NULL,
    next_funding_time TIMESTAMPTZ,
    predicted_rate DECIMAL(12,8),
    mark_price DECIMAL(20,8),
    index_price DECIMAL(20,8),
    interest_rate DECIMAL(12,8),
    premium DECIMAL(12,8),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT funding_rates_unique UNIQUE (exchange, trading_pair, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('funding_rates', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_funding_rates_exchange_pair ON funding_rates (exchange, trading_pair);
CREATE INDEX IF NOT EXISTS idx_funding_rates_timestamp ON funding_rates (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_funding_rates_rate ON funding_rates (funding_rate);
CREATE INDEX IF NOT EXISTS idx_funding_rates_next_funding ON funding_rates (next_funding_time);

-- ==============================================================================
-- LIQUIDATIONS TABLE
-- ==============================================================================

CREATE TABLE IF NOT EXISTS liquidations (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    liquidation_type VARCHAR(20) NOT NULL,
    side VARCHAR(5) NOT NULL CHECK (side IN ('long', 'short')),
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    value DECIMAL(20,8),
    leverage DECIMAL(10,2),
    margin DECIMAL(20,8),
    trader_id VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT liquidations_unique UNIQUE (exchange, trading_pair, timestamp, trader_id)
);

-- Convert to hypertable
SELECT create_hypertable('liquidations', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_liquidations_exchange_pair ON liquidations (exchange, trading_pair);
CREATE INDEX IF NOT EXISTS idx_liquidations_timestamp ON liquidations (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_liquidations_value ON liquidations (value DESC) WHERE value IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_liquidations_side ON liquidations (side);
CREATE INDEX IF NOT EXISTS idx_liquidations_type ON liquidations (liquidation_type);

-- ==============================================================================
-- METADATA TABLES
-- ==============================================================================

-- Exchange information
CREATE TABLE IF NOT EXISTS metadata.exchanges (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    api_url VARCHAR(255),
    websocket_url VARCHAR(255),
    countries TEXT[],
    fees JSONB,
    features JSONB,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trading pair information
CREATE TABLE IF NOT EXISTS metadata.trading_pairs (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    base_currency VARCHAR(10) NOT NULL,
    quote_currency VARCHAR(10) NOT NULL,
    active BOOLEAN DEFAULT true,
    min_quantity DECIMAL(20,8),
    max_quantity DECIMAL(20,8),
    quantity_precision INTEGER,
    price_precision INTEGER,
    min_price DECIMAL(20,8),
    max_price DECIMAL(20,8),
    tick_size DECIMAL(20,8),
    contract_type VARCHAR(20),
    settlement_currency VARCHAR(10),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    CONSTRAINT trading_pairs_unique UNIQUE (exchange, symbol)
);

-- Data quality metrics
CREATE TABLE IF NOT EXISTS metadata.data_quality_metrics (
    id BIGSERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    data_type VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    records_count INTEGER DEFAULT 0,
    missing_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    quality_score DECIMAL(5,2),
    latency_ms INTEGER,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('metadata.data_quality_metrics', 'timestamp', chunk_time_interval => INTERVAL '1 day');

-- ==============================================================================
-- COMPRESSION POLICIES
-- ==============================================================================

-- Compress old data to save space
SELECT add_compression_policy('candles', INTERVAL '7 days');
SELECT add_compression_policy('orderbooks', INTERVAL '1 day');
SELECT add_compression_policy('trades', INTERVAL '3 days');
SELECT add_compression_policy('funding_rates', INTERVAL '7 days');
SELECT add_compression_policy('liquidations', INTERVAL '7 days');

-- ==============================================================================
-- DATA RETENTION POLICIES
-- ==============================================================================

-- Automatically drop old data
SELECT add_retention_policy('orderbooks', INTERVAL '30 days');
SELECT add_retention_policy('trades', INTERVAL '90 days');
SELECT add_retention_policy('metadata.data_quality_metrics', INTERVAL '180 days');

-- Keep candles, funding rates, and liquidations longer
SELECT add_retention_policy('candles', INTERVAL '2 years');
SELECT add_retention_policy('funding_rates', INTERVAL '1 year');
SELECT add_retention_policy('liquidations', INTERVAL '1 year');

-- ==============================================================================
-- CONTINUOUS AGGREGATES (MATERIALIZED VIEWS)
-- ==============================================================================

-- Daily candle aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS candles_daily
WITH (timescaledb.continuous) AS
SELECT
    exchange,
    trading_pair,
    time_bucket('1 day', timestamp) AS day,
    FIRST(open_price, timestamp) AS open_price,
    MAX(high_price) AS high_price,
    MIN(low_price) AS low_price,
    LAST(close_price, timestamp) AS close_price,
    SUM(volume) AS volume,
    SUM(quote_volume) AS quote_volume,
    SUM(trade_count) AS trade_count
FROM candles
GROUP BY exchange, trading_pair, day;

-- Hourly trade aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS trades_hourly
WITH (timescaledb.continuous) AS
SELECT
    exchange,
    trading_pair,
    time_bucket('1 hour', timestamp) AS hour,
    COUNT(*) AS trade_count,
    SUM(quantity) AS total_volume,
    AVG(price) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price,
    STDDEV(price) AS price_volatility
FROM trades
GROUP BY exchange, trading_pair, hour;

-- Add refresh policies
SELECT add_continuous_aggregate_policy('candles_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('trades_hourly',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes');

-- ==============================================================================
-- FUNCTIONS AND TRIGGERS
-- ==============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$ language 'plpgsql';

-- Add triggers for updated_at
CREATE TRIGGER update_candles_updated_at BEFORE UPDATE ON candles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_funding_rates_updated_at BEFORE UPDATE ON funding_rates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_liquidations_updated_at BEFORE UPDATE ON liquidations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT USAGE ON SCHEMA market_data TO postgres;
GRANT USAGE ON SCHEMA metadata TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMAclass PostgresOrderBookRepository(OrderBookRepositoryInterface):
    """PostgreSQL implementation of order book repository"""
    
    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.logger = logger.bind(repository="orderbook")
    
    async def save_orderbook(self, orderbook: OrderBook) -> bool:
        """Save order book snapshot"""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # Insert main orderbook record
                    orderbook_id = await conn.fetchval("""
                        INSERT INTO orderbooks (
                            exchange, trading_pair, timestamp, best_bid, best_ask,
                            bid_count, ask_count, total_bid_volume, total_ask_volume,
                            spread, spread_percentage, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                        RETURNING id
                    """,
                    orderbook.exchange,
                    orderbook.trading_pair,
                    orderbook.timestamp,
                    float(orderbook.best_bid) if orderbook.best_bid else None,
                    float(orderbook.best_ask) if orderbook.best_ask else None,
                    len(orderbook.bids),
                    len(orderbook.asks),
                    float(orderbook.total_bid_volume) if orderbook.total_bid_volume else None,
                    float(orderbook.total_ask_volume) if orderbook.total_ask_volume else None,
                    float(orderbook.spread) if orderbook.spread else None,
                    float(orderbook.spread_percentage) if orderbook.spread_percentage else None,
                    orderbook.metadata
                    )
                    
                    # Insert bids
                    if orderbook.bids:
                        bid_data = [
                            (orderbook_id, float(bid.price), float(bid.quantity), bid.order_count)
                            for bid in orderbook.bids
                        ]
                        await conn.executemany("""
                            INSERT INTO orderbook_entries (orderbook_id, price, quantity, order_count, side)
                            VALUES ($1, $2, $3, $4, 'bid')
                        """, bid_data)
                    
                    # Insert asks
                    if orderbook.asks:
                        ask_data = [
                            (orderbook_id, float(ask.price), float(ask.quantity), ask.order_count)
                            for ask in orderbook.asks
                        ]
                        await conn.executemany("""
                            INSERT INTO orderbook_entries (orderbook_id, price, quantity, order_count, side)
                            VALUES ($1, $2, $3, $4, 'ask')
                        """, ask_data)
            
            self.logger.debug(
                "OrderBook saved",
                exchange=orderbook.exchange,
                trading_pair=orderbook.trading_pair,
                timestamp=orderbook.timestamp,
                bids=len(orderbook.bids),
                asks=len(orderbook.asks)
            )
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to save orderbook",
                error=str(e),
                exchange=orderbook.exchange,
                trading_pair=orderbook.trading_pair
            )
            return False
    
    async def get_latest_orderbook(
        self,
        exchange: str,
        trading_pair: str
    ) -> Optional[OrderBook]:
        """Get the most recent order book"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get latest orderbook record
                orderbook_row = await conn.fetchrow("""
                    SELECT id, exchange, trading_pair, timestamp, best_bid, best_ask,
                           bid_count, ask_count, total_bid_volume, total_ask_volume,
                           spread, spread_percentage, metadata
                    FROM orderbooks
                    WHERE exchange = $1 AND trading_pair = $2
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, exchange, trading_pair)
                
                if not orderbook_row:
                    return None
                
                # Get orderbook entries
                entries = await conn.fetch("""
                    SELECT price, quantity, order_count, side
                    FROM orderbook_entries
                    WHERE orderbook_id = $1
                    ORDER BY 
                        CASE WHEN side = 'bid' THEN price END DESC,
                        CASE WHEN side = 'ask' THEN price END ASC
                """, orderbook_row['id'])
                
                return self._build_orderbook(orderbook_row, entries)
                
        except Exception as e:
            self.logger.error("Failed to get latest orderbook", error=str(e))
            return None
    
    async def get_orderbook_at_time(
        self,
        exchange: str,
        trading_pair: str,
        timestamp: datetime
    ) -> Optional[OrderBook]:
        """Get order book closest to specified time"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get orderbook closest to timestamp
                orderbook_row = await conn.fetchrow("""
                    SELECT id, exchange, trading_pair, timestamp, best_bid, best_ask,
                           bid_count, ask_count, total_bid_volume, total_ask_volume,
                           spread, spread_percentage, metadata
                    FROM orderbooks
                    WHERE exchange = $1 AND trading_pair = $2 AND timestamp <= $3
                    ORDER BY timestamp DESC
                    LIMIT 1
                """, exchange, trading_pair, timestamp)
                
                if not orderbook_row:
                    return None
                
                # Get orderbook entries
                entries = await conn.fetch("""
                    SELECT price, quantity, order_count, side
                    FROM orderbook_entries
                    WHERE orderbook_id = $1
                    ORDER BY 
                        CASE WHEN side = 'bid' THEN price END DESC,
                        CASE WHEN side = 'ask' THEN price END ASC
                """, orderbook_row['id'])
                
                return self._build_orderbook(orderbook_row, entries)
                
        except Exception as e:
            self.logger.error("Failed to get orderbook at time", error=str(e))
            return None
    
    async def get_orderbook_history(
        self,
        exchange: str,
        trading_pair: str,
        start_time: datetime,
        end_time: datetime,
        limit: Optional[int] = None
    ) -> List[OrderBook]:
        """Get order book snapshots within time range"""
        try:
            async with self.db_pool.acquire() as conn:
                query = """
                    SELECT id, exchange, trading_pair, timestamp, best_bid, best_ask,
                           bid_count, ask_count, total_bid_volume, total_ask_volume,
                           spread, spread_percentage, metadata
                    FROM orderbooks
                    WHERE exchange = $1 AND trading_pair = $2 
                          AND timestamp >= $3 AND timestamp <= $4
                    ORDER BY timestamp ASC
                """
                
                params = [exchange, trading_pair, start_time, end_time]
                
                if limit:
                    query += " LIMIT $5"
                    params.append(limit)
                
                orderbook_rows = await conn.fetch(query, *params)
                
                orderbooks = []
                for row in orderbook_rows:
                    # Get entries for this orderbook
                    entries = await conn.fetch("""
                        SELECT price, quantity, order_count, side
                        FROM orderbook_entries
                        WHERE orderbook_id = $1
                        ORDER BY 
                            CASE WHEN side = 'bid' THEN price END DESC,
                            CASE WHEN side = 'ask' THEN price END ASC
                    """, row['id'])
                    
                    orderbook = self._build_orderbook(row, entries)
                    if orderbook:
                        orderbooks.append(orderbook)
                
                return orderbooks
                
        except Exception as e:
            self.logger.error("Failed to get orderbook history", error=str(e))
            return []
    
    async def delete_old_orderbooks(
        self,
        exchange: str,
        trading_pair: str,
        before_time: datetime
    ) -> int:
        """Delete order books before specified time"""
        try:
            async with self.db_pool.acquire() as conn:
                async with conn.transaction():
                    # First delete entries
                    await conn.execute("""
                        DELETE FROM orderbook_entries 
                        WHERE orderbook_id IN (
                            SELECT id FROM orderbooks
                            WHERE exchange = $1 AND trading_pair = $2 AND timestamp < $3
                        )
                    """, exchange, trading_pair, before_time)
                    
                    # Then delete orderbooks
                    result = await conn.execute("""
                        DELETE FROM orderbooks
                        WHERE exchange = $1 AND trading_pair = $2 AND timestamp < $3
                    """, exchange, trading_pair, before_time)
                    
                    deleted_count = int(result.split()[-1])
                    self.logger.info(f"Deleted {deleted_count} orderbooks")
                    return deleted_count
                    
        except Exception as e:
            self.logger.error("Failed to delete old orderbooks", error=str(e))
            return 0
    
    def _build_orderbook(self, orderbook_row, entries) -> Optional[OrderBook]:
        """Build OrderBook entity from database rows"""
        try:
            bids = []
            asks = []
            
            for entry in entries:
                order_entry = OrderBookEntry(
                    price=Decimal(str(entry['price'])),
                    quantity=Decimal(str(entry['quantity'])),
                    order_count=entry['order_count']
                )
                
                if entry['side'] == 'bid':
                    bids.append(order_entry)
                else:
                    asks.append(order_entry)
            
            return OrderBook(
                exchange=orderbook_row['exchange'],
                trading_pair=orderbook_row['trading_pair'],
                timestamp=orderbook_row['timestamp'],
                bids=bids,
                asks=asks,
                metadata=orderbook_row['metadata'] or {}
            )
            
        except Exception as e:
            self.logger.error("Failed to build orderbook", error=str(e))
            return None