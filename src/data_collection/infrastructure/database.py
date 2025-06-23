"""
Gerenciador de banco de dados corrigido.

Este módulo implementa o gerenciamento de conexões com PostgreSQL
com pool de conexões, health checks, e operações otimizadas.
"""
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import asynccontextmanager
import threading

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.pool import NullPool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

from src.data_collection.domain.entities.candle import Candle
from src.data_collection.domain.entities.orderbook import OrderBook
from src.data_collection.domain.entities.trade import Trade


logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Erro base para operações de banco de dados."""
    pass


class ConnectionError(DatabaseError):
    """Erro de conexão com banco de dados."""
    pass


class QueryError(DatabaseError):
    """Erro em query SQL."""
    pass


class DatabaseManager:
    """
    Gerenciador de banco de dados com pool de conexões.
    
    Implementa operações otimizadas para armazenamento de dados de mercado
    com suporte a transações, batch operations e health checks.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "neural_crypto_bot",
        user: str = "postgres",
        password: str = "postgres",
        min_connections: int = 5,
        max_connections: int = 20,
        enable_ssl: bool = False,
        schema: str = "public",
        statement_timeout: int = 30000,
        query_timeout: int = 30000
    ):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            host: Host do PostgreSQL
            port: Porta do PostgreSQL
            database: Nome do banco de dados
            user: Usuário
            password: Senha
            min_connections: Mínimo de conexões no pool
            max_connections: Máximo de conexões no pool
            enable_ssl: Habilitar SSL
            schema: Schema padrão
            statement_timeout: Timeout para statements (ms)
            query_timeout: Timeout para queries (ms)
        """
        if not HAS_ASYNCPG and not HAS_SQLALCHEMY:
            raise DatabaseError("asyncpg ou sqlalchemy são necessários")
        
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.enable_ssl = enable_ssl
        self.schema = schema
        self.statement_timeout = statement_timeout
        self.query_timeout = query_timeout
        
        # Pool de conexões
        self._pool = None
        self._engine = None
        self._session_maker = None
        
        # Estado
        self._initialized = False
        self._closed = False
        self._init_lock = asyncio.Lock()
        
        # Estatísticas
        self._stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_connections': 0,
            'active_connections': 0,
            'avg_query_time': 0.0,
            'last_query_time': None
        }
        self._stats_lock = threading.Lock()
        
        # Configurar URL de conexão
        ssl_mode = "require" if enable_ssl else "disable"
        self._database_url = (
            f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
            f"?ssl={ssl_mode}&statement_timeout={statement_timeout}&query_timeout={query_timeout}"
        )
        
        logger.info(f"DatabaseManager configurado para {host}:{port}/{database}")
    
    async def initialize(self) -> None:
        """Inicializa o pool de conexões."""
        async with self._init_lock:
            if self._initialized:
                return
            
            try:
                if HAS_SQLALCHEMY:
                    await self._initialize_sqlalchemy()
                elif HAS_ASYNCPG:
                    await self._initialize_asyncpg()
                else:
                    raise DatabaseError("Nenhuma biblioteca de banco disponível")
                
                # Testar conexão
                await self._test_connection()
                
                # Verificar/criar tabelas
                await self._ensure_tables()
                
                self._initialized = True
                logger.info("DatabaseManager inicializado com sucesso")
                
            except Exception as e:
                logger.error(f"Erro na inicialização do banco: {e}")
                raise ConnectionError(f"Falha ao conectar: {e}")
    
    async def _initialize_sqlalchemy(self) -> None:
        """Inicializa usando SQLAlchemy."""
        self._engine = create_async_engine(
            self._database_url,
            pool_size=self.max_connections,
            max_overflow=0,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            poolclass=NullPool if self.max_connections == 0 else None
        )
        
        self._session_maker = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("SQLAlchemy engine configurado")
    
    async def _initialize_asyncpg(self) -> None:
        """Inicializa usando asyncpg."""
        self._pool = await asyncpg.create_pool(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            min_size=self.min_connections,
            max_size=self.max_connections,
            ssl="require" if self.enable_ssl else "disable",
            statement_timeout=self.statement_timeout / 1000,  # asyncpg usa segundos
            query_timeout=self.query_timeout / 1000
        )
        
        logger.info("asyncpg pool configurado")
    
    async def _test_connection(self) -> None:
        """Testa conectividade básica."""
        try:
            result = await self.execute_query("SELECT 1 as test")
            if result and result[0]['test'] == 1:
                logger.info("Teste de conectividade: OK")
            else:
                raise ConnectionError("Teste de conectividade falhou")
        except Exception as e:
            raise ConnectionError(f"Teste de conectividade falhou: {e}")
    
    async def _ensure_tables(self) -> None:
        """Verifica/cria tabelas necessárias."""
        tables_sql = [
            # Tabela de candles
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.candles (
                id BIGSERIAL PRIMARY KEY,
                exchange VARCHAR(50) NOT NULL,
                trading_pair VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open_price DECIMAL(20,8) NOT NULL,
                high_price DECIMAL(20,8) NOT NULL,
                low_price DECIMAL(20,8) NOT NULL,
                close_price DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                close_timestamp TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(exchange, trading_pair, timeframe, timestamp)
            );
            """,
            
            # Índices para candles
            f"""
            CREATE INDEX IF NOT EXISTS idx_candles_exchange_pair_timeframe 
            ON {self.schema}.candles(exchange, trading_pair, timeframe);
            """,
            
            f"""
            CREATE INDEX IF NOT EXISTS idx_candles_timestamp 
            ON {self.schema}.candles(timestamp);
            """,
            
            # Tabela de trades
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.trades (
                id BIGSERIAL PRIMARY KEY,
                exchange VARCHAR(50) NOT NULL,
                trading_pair VARCHAR(20) NOT NULL,
                trade_id VARCHAR(100) NOT NULL,
                price DECIMAL(20,8) NOT NULL,
                amount DECIMAL(20,8) NOT NULL,
                cost DECIMAL(20,8) NOT NULL,
                side VARCHAR(10) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(exchange, trading_pair, trade_id)
            );
            """,
            
            # Índices para trades
            f"""
            CREATE INDEX IF NOT EXISTS idx_trades_exchange_pair 
            ON {self.schema}.trades(exchange, trading_pair);
            """,
            
            f"""
            CREATE INDEX IF NOT EXISTS idx_trades_timestamp 
            ON {self.schema}.trades(timestamp);
            """,
            
            # Tabela de orderbooks (snapshot)
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.orderbooks (
                id BIGSERIAL PRIMARY KEY,
                exchange VARCHAR(50) NOT NULL,
                trading_pair VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                bids JSONB NOT NULL,
                asks JSONB NOT NULL,
                spread_percentage DECIMAL(10,6),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """,
            
            # Índices para orderbooks
            f"""
            CREATE INDEX IF NOT EXISTS idx_orderbooks_exchange_pair 
            ON {self.schema}.orderbooks(exchange, trading_pair);
            """,
            
            f"""
            CREATE INDEX IF NOT EXISTS idx_orderbooks_timestamp 
            ON {self.schema}.orderbooks(timestamp);
            """
        ]
        
        for sql in tables_sql:
            try:
                await self.execute_query(sql)
            except Exception as e:
                logger.warning(f"Erro ao criar/verificar tabela: {e}")
        
        logger.info("Tabelas verificadas/criadas")
    
    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executa uma query SQL.
        
        Args:
            query: Query SQL
            params: Parâmetros da query
            
        Returns:
            List[Dict[str, Any]]: Resultados da query
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            if self._engine:
                result = await self._execute_with_sqlalchemy(query, params)
            elif self._pool:
                result = await self._execute_with_asyncpg(query, params)
            else:
                raise DatabaseError("Nenhuma conexão disponível")
            
            # Atualizar estatísticas
            execution_time = time.time() - start_time
            self._update_stats(True, execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(False, execution_time)
            logger.error(f"Erro na query: {e}")
            raise QueryError(f"Erro na execução da query: {e}")
    
    async def _execute_with_sqlalchemy(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executa query usando SQLAlchemy."""
        async with self._session_maker() as session:
            try:
                if params:
                    result = await session.execute(sa.text(query), params)
                else:
                    result = await session.execute(sa.text(query))
                
                # Converter resultado para lista de dicts
                if result.returns_rows:
                    rows = result.fetchall()
                    return [dict(row._mapping) for row in rows]
                else:
                    await session.commit()
                    return []
                    
            except Exception as e:
                await session.rollback()
                raise
    
    async def _execute_with_asyncpg(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Executa query usando asyncpg."""
        async with self._pool.acquire() as conn:
            if params:
                # Converter dict para lista ordenada para asyncpg
                param_values = list(params.values()) if params else []
                # Substituir named parameters por $1, $2, etc.
                formatted_query = query
                for i, key in enumerate(params.keys(), 1):
                    formatted_query = formatted_query.replace(f":{key}", f"${i}")
                
                result = await conn.fetch(formatted_query, *param_values)
            else:
                result = await conn.fetch(query)
            
            # Converter Record para dict
            return [dict(row) for row in result]
    
    @asynccontextmanager
    async def transaction(self):
        """Context manager para transações."""
        if not self._initialized:
            await self.initialize()
        
        if self._engine:
            async with self._session_maker() as session:
                async with session.begin():
                    yield session
        elif self._pool:
            async with self._pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        else:
            raise DatabaseError("Nenhuma conexão disponível")
    
    async def execute_batch(
        self,
        query: str,
        params_list: List[Dict[str, Any]]
    ) -> None:
        """
        Executa query em batch para melhor performance.
        
        Args:
            query: Query SQL
            params_list: Lista de parâmetros para cada execução
        """
        if not params_list:
            return
        
        start_time = time.time()
        
        try:
            async with self.transaction() as conn:
                if self._engine:
                    # SQLAlchemy
                    for params in params_list:
                        await conn.execute(sa.text(query), params)
                else:
                    # asyncpg - usar executemany para melhor performance
                    # Converter named parameters para positional
                    first_params = params_list[0]
                    formatted_query = query
                    for i, key in enumerate(first_params.keys(), 1):
                        formatted_query = formatted_query.replace(f":{key}", f"${i}")
                    
                    param_tuples = [
                        tuple(params[key] for key in first_params.keys())
                        for params in params_list
                    ]
                    
                    await conn.executemany(formatted_query, param_tuples)
            
            execution_time = time.time() - start_time
            self._update_stats(True, execution_time)
            
            logger.debug(f"Batch executado: {len(params_list)} registros em {execution_time:.2f}s")
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_stats(False, execution_time)
            logger.error(f"Erro no batch: {e}")
            raise QueryError(f"Erro na execução do batch: {e}")
    
    # Métodos específicos para entidades de domínio
    
    async def save_candle(self, candle: Candle) -> None:
        """Salva um candle no banco."""
        query = f"""
        INSERT INTO {self.schema}.candles 
        (exchange, trading_pair, timeframe, timestamp, open_price, high_price, 
         low_price, close_price, volume, close_timestamp)
        VALUES 
        (:exchange, :trading_pair, :timeframe, :timestamp, :open_price, :high_price,
         :low_price, :close_price, :volume, :close_timestamp)
        ON CONFLICT (exchange, trading_pair, timeframe, timestamp) 
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume,
            close_timestamp = EXCLUDED.close_timestamp
        """
        
        params = {
            'exchange': candle.exchange,
            'trading_pair': candle.trading_pair,
            'timeframe': candle.timeframe.value,
            'timestamp': candle.timestamp,
            'open_price': float(candle.open_price),
            'high_price': float(candle.high_price),
            'low_price': float(candle.low_price),
            'close_price': float(candle.close_price),
            'volume': float(candle.volume),
            'close_timestamp': candle.close_timestamp
        }
        
        await self.execute_query(query, params)
    
    async def save_candles_batch(self, candles: List[Candle]) -> None:
        """Salva múltiplos candles em batch."""
        if not candles:
            return
        
        query = f"""
        INSERT INTO {self.schema}.candles 
        (exchange, trading_pair, timeframe, timestamp, open_price, high_price, 
         low_price, close_price, volume, close_timestamp)
        VALUES 
        (:exchange, :trading_pair, :timeframe, :timestamp, :open_price, :high_price,
         :low_price, :close_price, :volume, :close_timestamp)
        ON CONFLICT (exchange, trading_pair, timeframe, timestamp) 
        DO UPDATE SET
            open_price = EXCLUDED.open_price,
            high_price = EXCLUDED.high_price,
            low_price = EXCLUDED.low_price,
            close_price = EXCLUDED.close_price,
            volume = EXCLUDED.volume,
            close_timestamp = EXCLUDED.close_timestamp
        """
        
        params_list = []
        for candle in candles:
            params_list.append({
                'exchange': candle.exchange,
                'trading_pair': candle.trading_pair,
                'timeframe': candle.timeframe.value,
                'timestamp': candle.timestamp,
                'open_price': float(candle.open_price),
                'high_price': float(candle.high_price),
                'low_price': float(candle.low_price),
                'close_price': float(candle.close_price),
                'volume': float(candle.volume),
                'close_timestamp': candle.close_timestamp
            })
        
        await self.execute_batch(query, params_list)
        logger.info(f"Salvos {len(candles)} candles em batch")
    
    async def save_trade(self, trade: Trade) -> None:
        """Salva um trade no banco."""
        query = f"""
        INSERT INTO {self.schema}.trades 
        (exchange, trading_pair, trade_id, price, amount, cost, side, timestamp)
        VALUES 
        (:exchange, :trading_pair, :trade_id, :price, :amount, :cost, :side, :timestamp)
        ON CONFLICT (exchange, trading_pair, trade_id) 
        DO NOTHING
        """
        
        params = {
            'exchange': trade.exchange,
            'trading_pair': trade.trading_pair,
            'trade_id': trade.trade_id,
            'price': float(trade.price),
            'amount': float(trade.amount),
            'cost': float(trade.cost),
            'side': trade.side.value,
            'timestamp': trade.timestamp
        }
        
        await self.execute_query(query, params)
    
    async def save_trades_batch(self, trades: List[Trade]) -> None:
        """Salva múltiplos trades em batch."""
        if not trades:
            return
        
        query = f"""
        INSERT INTO {self.schema}.trades 
        (exchange, trading_pair, trade_id, price, amount, cost, side, timestamp)
        VALUES 
        (:exchange, :trading_pair, :trade_id, :price, :amount, :cost, :side, :timestamp)
        ON CONFLICT (exchange, trading_pair, trade_id) 
        DO NOTHING
        """
        
        params_list = []
        for trade in trades:
            params_list.append({
                'exchange': trade.exchange,
                'trading_pair': trade.trading_pair,
                'trade_id': trade.trade_id,
                'price': float(trade.price),
                'amount': float(trade.amount),
                'cost': float(trade.cost),
                'side': trade.side.value,
                'timestamp': trade.timestamp
            })
        
        await self.execute_batch(query, params_list)
        logger.info(f"Salvos {len(trades)} trades em batch")
    
    async def save_orderbook(self, orderbook: OrderBook) -> None:
        """Salva um orderbook no banco."""
        import json
        
        # Converter bids e asks para JSON
        bids_json = [
            {"price": float(level.price), "amount": float(level.amount)}
            for level in orderbook.bids
        ]
        
        asks_json = [
            {"price": float(level.price), "amount": float(level.amount)}
            for level in orderbook.asks
        ]
        
        query = f"""
        INSERT INTO {self.schema}.orderbooks 
        (exchange, trading_pair, timestamp, bids, asks, spread_percentage)
        VALUES 
        (:exchange, :trading_pair, :timestamp, :bids, :asks, :spread_percentage)
        """
        
        params = {
            'exchange': orderbook.exchange,
            'trading_pair': orderbook.trading_pair,
            'timestamp': orderbook.timestamp,
            'bids': json.dumps(bids_json),
            'asks': json.dumps(asks_json),
            'spread_percentage': float(orderbook.spread_percentage) if hasattr(orderbook, 'spread_percentage') else None
        }
        
        await self.execute_query(query, params)
    
    def _update_stats(self, success: bool, execution_time: float) -> None:
        """Atualiza estatísticas de operações."""
        with self._stats_lock:
            self._stats['total_queries'] += 1
            
            if success:
                self._stats['successful_queries'] += 1
            else:
                self._stats['failed_queries'] += 1
            
            # Atualizar tempo médio de query
            total_time = self._stats['avg_query_time'] * (self._stats['total_queries'] - 1)
            self._stats['avg_query_time'] = (total_time + execution_time) / self._stats['total_queries']
            
            self._stats['last_query_time'] = datetime.utcnow()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do banco."""
        with self._stats_lock:
            stats = self._stats.copy()
        
        # Adicionar estatísticas do pool
        if self._pool:
            stats['pool_size'] = self._pool.get_size()
            stats['pool_available'] = self._pool.get_available_connections()
            stats['pool_max_size'] = self._pool.get_max_size()
        elif self._engine:
            pool = self._engine.pool
            stats['pool_size'] = pool.size()
            stats['pool_checked_out'] = pool.checkedout()
            stats['pool_checked_in'] = pool.checkedin()
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """Verifica saúde da conexão com banco."""
        try:
            start_time = time.time()
            
            # Teste básico de conectividade
            result = await self.execute_query("SELECT NOW() as current_time, 1 as test")
            
            query_time = (time.time() - start_time) * 1000  # ms
            
            # Obter estatísticas
            stats = await self.get_stats()
            
            return {
                'healthy': True,
                'database': self.database,
                'host': self.host,
                'query_time_ms': query_time,
                'current_time': result[0]['current_time'].isoformat() if result else None,
                'stats': stats,
                'initialized': self._initialized
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'database': self.database,
                'host': self.host,
                'error': str(e),
                'initialized': self._initialized
            }
    
    async def shutdown(self) -> None:
        """Fecha todas as conexões."""
        if self._closed:
            return
        
        try:
            if self._engine:
                await self._engine.dispose()
                logger.info("SQLAlchemy engine fechado")
            
            if self._pool:
                await self._pool.close()
                logger.info("asyncpg pool fechado")
            
            self._closed = True
            self._initialized = False
            
            logger.info("DatabaseManager finalizado")
            
        except Exception as e:
            logger.error(f"Erro ao fechar conexões: {e}")
    
    def __del__(self):
        """Cleanup no destructor."""
        if not self._closed and self._initialized:
            # Não podemos usar await aqui, apenas log warning
            logger.warning("DatabaseManager não foi fechado adequadamente")


# Factory function para facilitar criação
def create_database_manager(config: Dict[str, Any]) -> DatabaseManager:
    """
    Cria instância do DatabaseManager a partir de configuração.
    
    Args:
        config: Configuração do banco
        
    Returns:
        DatabaseManager: Instância configurada
    """
    return DatabaseManager(
        host=config.get('host', 'localhost'),
        port=config.get('port', 5432),
        database=config.get('database', 'neural_crypto_bot'),
        user=config.get('user', 'postgres'),
        password=config.get('password', 'postgres'),
        min_connections=config.get('min_connections', 5),
        max_connections=config.get('max_connections', 20),
        enable_ssl=config.get('enable_ssl', False),
        schema=config.get('schema', 'public'),
        statement_timeout=config.get('statement_timeout', 30000),
        query_timeout=config.get('query_timeout', 30000)
    )