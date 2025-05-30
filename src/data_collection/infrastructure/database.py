"""
Conexão e operações de banco de dados.

Este módulo implementa a conexão com o banco de dados e
operações básicas de persistência para os dados coletados.
"""
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple

import asyncpg
from asyncpg import Pool, Connection
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook, OrderBookLevel
from src.data_collection.domain.entities.trade import Trade, TradeSide


logger = logging.getLogger(__name__)
Base = declarative_base()


class DatabaseManager:
    """
    Gerenciador de conexão e operações de banco de dados.
    
    Implementa funcionalidades para conectar ao banco de dados
    e realizar operações de CRUD para os dados coletados.
    """
    
    def __init__(
        self, 
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        min_connections: int = 5,
        max_connections: int = 20,
        enable_ssl: bool = False,
        schema: str = "public"
    ):
        """
        Inicializa o gerenciador de banco de dados.
        
        Args:
            host: Host do banco de dados
            port: Porta do banco de dados
            database: Nome do banco de dados
            user: Usuário do banco de dados
            password: Senha do banco de dados
            min_connections: Número mínimo de conexões no pool
            max_connections: Número máximo de conexões no pool
            enable_ssl: Se True, habilita SSL na conexão
            schema: Schema do banco de dados
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.enable_ssl = enable_ssl
        self.schema = schema
        
        # Pool de conexões para asyncpg
        self._pool: Optional[Pool] = None
        
        # Engine SQLAlchemy para ORM
        self._engine = None
        self._session_factory = None
        
        # Lock para inicialização
        self._init_lock = asyncio.Lock()
        
        # Flag para controle de estado
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        Inicializa a conexão com o banco de dados.
        
        Esta função deve ser chamada antes de usar qualquer
        operação do banco de dados.
        """
        async with self._init_lock:
            if self._initialized:
                return
                
            try:
                logger.info(f"Conectando ao banco de dados {self.database} em {self.host}:{self.port}")
                
                # Inicializa o pool de conexões asyncpg
                ssl = None
                if self.enable_ssl:
                    ssl = True  # Pode ser configurado com mais detalhes se necessário
                
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=self.min_connections,
                    max_size=self.max_connections,
                    ssl=ssl,
                    command_timeout=60
                )
                
                # Inicializa o engine SQLAlchemy
                db_url = f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
                self._engine = create_async_engine(
                    db_url,
                    echo=False,
                    pool_pre_ping=True,
                    pool_size=self.min_connections,
                    max_overflow=self.max_connections - self.min_connections
                )
                
                self._session_factory = sessionmaker(
                    self._engine, 
                    class_=AsyncSession, 
                    expire_on_commit=False
                )
                
                # Verifica se as tabelas necessárias existem
                await self._ensure_tables()
                
                logger.info("Conexão com o banco de dados estabelecida")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Erro ao conectar ao banco de dados: {str(e)}", exc_info=True)
                raise
    
    async def close(self) -> None:
        """
        Fecha a conexão com o banco de dados.
        """
        if self._pool:
            await self._pool.close()
            self._pool = None
            
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            
        self._initialized = False
        logger.info("Conexão com o banco de dados fechada")
    
    async def get_connection(self) -> Connection:
        """
        Obtém uma conexão do pool.
        
        Returns:
            Connection: Conexão do pool
            
        Raises:
            ConnectionError: Se não for possível obter uma conexão
        """
        if not self._initialized:
            await self.initialize()
            
        if not self._pool:
            raise ConnectionError("Pool de conexões não inicializado")
            
        return await self._pool.acquire()
    
    async def release_connection(self, connection: Connection) -> None:
        """
        Libera uma conexão de volta para o pool.
        
        Args:
            connection: Conexão a ser liberada
        """
        if self._pool and connection:
            await self._pool.release(connection)
    
    async def get_session(self) -> AsyncSession:
        """
        Obtém uma sessão SQLAlchemy.
        
        Returns:
            AsyncSession: Sessão SQLAlchemy
            
        Raises:
            ConnectionError: Se não for possível obter uma sessão
        """
        if not self._initialized:
            await self.initialize()
            
        if not self._session_factory:
            raise ConnectionError("Session factory não inicializado")
            
        return self._session_factory()
    
    async def _ensure_tables(self) -> None:
        """
        Garante que as tabelas necessárias existem no banco de dados.
        
        Cria as tabelas se não existirem.
        """
        async with self._pool.acquire() as conn:
            # Verifica se o schema existe
            schema_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = $1)",
                self.schema
            )
            
            if not schema_exists:
                await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            
            # Cria a tabela de candles
            await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.candles (
                exchange TEXT NOT NULL,
                trading_pair TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                timeframe TEXT NOT NULL,
                open NUMERIC(30, 15) NOT NULL,
                high NUMERIC(30, 15) NOT NULL,
                low NUMERIC(30, 15) NOT NULL,
                close NUMERIC(30, 15) NOT NULL,
                volume NUMERIC(30, 15) NOT NULL,
                trades INTEGER,
                raw_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (exchange, trading_pair, timeframe, timestamp)
            )
            """)
            
            # Cria índices para a tabela de candles
            await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS candles_exchange_trading_pair_timeframe_timestamp_idx
            ON {self.schema}.candles (exchange, trading_pair, timeframe, timestamp)
            """)
            
            # Cria a tabela de orderbooks
            await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.orderbooks (
                exchange TEXT NOT NULL,
                trading_pair TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                bids JSONB NOT NULL,
                asks JSONB NOT NULL,
                raw_data JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (exchange, trading_pair, timestamp)
            )
            """)
            
            # Cria índices para a tabela de orderbooks
            await conn.execute(f"""
            CREATE INDEX IF NOT EXISTS orderbooks_exchange_trading_pair_timestamp_idx
            ON {self.schema}.orderbooks (exchange, trading_pair, timestamp)
            """)
            
            logger.info("Tabelas verificadas/criadas com sucesso")
    
    async def save_trade(self, trade: Trade) -> None:
        """
        Salva uma transação no banco de dados.
        
        Args:
            trade: Entidade Trade a ser salva
            
        Raises:
            Exception: Se ocorrer um erro ao salvar
        """
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Converte o raw_data para JSON se não for None
                raw_data = None
                if trade.raw_data:
                    try:
                        import json
                        raw_data = json.dumps(trade.raw_data)
                    except Exception as e:
                        logger.warning(f"Erro ao converter raw_data para JSON: {str(e)}")
                
                # Insere o trade na tabela
                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.trades
                    (id, exchange, trading_pair, price, amount, cost, timestamp, side, taker, raw_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (exchange, trading_pair, id) DO UPDATE
                    SET price = $4, amount = $5, cost = $6, timestamp = $7, side = $8, taker = $9, raw_data = $10
                    """,
                    trade.id,
                    trade.exchange,
                    trade.trading_pair,
                    float(trade.price),  # Converte Decimal para float para o banco
                    float(trade.amount),
                    float(trade.cost),
                    trade.timestamp,
                    trade.side.value,
                    trade.taker,
                    raw_data
                )
                
            except Exception as e:
                logger.error(f"Erro ao salvar trade: {str(e)}", exc_info=True)
                raise
    
    async def save_trades_batch(self, trades: List[Trade]) -> None:
        """
        Salva um lote de transações no banco de dados.
        
        Args:
            trades: Lista de entidades Trade a serem salvas
            
        Raises:
            Exception: Se ocorrer um erro ao salvar
        """
        if not trades:
            return
            
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Prepara os dados para inserção em batch
                values = []
                import json
                
                for trade in trades:
                    # Converte o raw_data para JSON se não for None
                    raw_data = None
                    if trade.raw_data:
                        try:
                            raw_data = json.dumps(trade.raw_data)
                        except Exception as e:
                            logger.warning(f"Erro ao converter raw_data para JSON: {str(e)}")
                    
                    values.append((
                        trade.id,
                        trade.exchange,
                        trade.trading_pair,
                        float(trade.price),
                        float(trade.amount),
                        float(trade.cost),
                        trade.timestamp,
                        trade.side.value,
                        trade.taker,
                        raw_data
                    ))
                
                # Inicia uma transação
                async with conn.transaction():
                    # Insere os trades em batch
                    await conn.executemany(
                        f"""
                        INSERT INTO {self.schema}.trades
                        (id, exchange, trading_pair, price, amount, cost, timestamp, side, taker, raw_data)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        ON CONFLICT (exchange, trading_pair, id) DO UPDATE
                        SET price = $4, amount = $5, cost = $6, timestamp = $7, side = $8, taker = $9, raw_data = $10
                        """,
                        values
                    )
                
            except Exception as e:
                logger.error(f"Erro ao salvar lote de trades: {str(e)}", exc_info=True)
                raise
    
    async def get_trades(
        self,
        exchange: str,
        trading_pair: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Trade]:
        """
        Recupera transações do banco de dados.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de transações a retornar (opcional)
            
        Returns:
            List[Trade]: Lista de entidades Trade
            
        Raises:
            Exception: Se ocorrer um erro ao recuperar os dados
        """
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Constrói a query com os filtros
                query = f"""
                SELECT id, exchange, trading_pair, price, amount, cost, timestamp, side, taker, raw_data
                FROM {self.schema}.trades
                WHERE exchange = $1 AND trading_pair = $2
                """
                
                params = [exchange, trading_pair]
                param_index = 3
                
                if start_time:
                    query += f" AND timestamp >= ${param_index}"
                    params.append(start_time)
                    param_index += 1
                
                if end_time:
                    query += f" AND timestamp <= ${param_index}"
                    params.append(end_time)
                    param_index += 1
                
                # Ordena por timestamp
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT ${param_index}"
                    params.append(limit)
                
                # Executa a query
                rows = await conn.fetch(query, *params)
                
                # Converte os resultados para entidades Trade
                trades = []
                for row in rows:
                    side = TradeSide(row['side'])
                    
                    # Converte JSON para dict se não for None
                    raw_data = None
                    if row['raw_data']:
                        import json
                        raw_data = json.loads(row['raw_data'])
                    
                    trade = Trade(
                        id=row['id'],
                        exchange=row['exchange'],
                        trading_pair=row['trading_pair'],
                        price=Decimal(str(row['price'])),
                        amount=Decimal(str(row['amount'])),
                        cost=Decimal(str(row['cost'])),
                        timestamp=row['timestamp'],
                        side=side,
                        taker=row['taker'],
                        raw_data=raw_data
                    )
                    
                    trades.append(trade)
                
                return trades
                
            except Exception as e:
                logger.error(f"Erro ao recuperar trades: {str(e)}", exc_info=True)
                raise
    
    async def save_candle(self, candle: Candle) -> None:
        """
        Salva uma vela no banco de dados.
        
        Args:
            candle: Entidade Candle a ser salva
            
        Raises:
            Exception: Se ocorrer um erro ao salvar
        """
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Converte o raw_data para JSON se não for None
                raw_data = None
                if candle.raw_data:
                    try:
                        import json
                        raw_data = json.dumps(candle.raw_data)
                    except Exception as e:
                        logger.warning(f"Erro ao converter raw_data para JSON: {str(e)}")
                
                # Insere a vela na tabela
                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.candles
                    (exchange, trading_pair, timestamp, timeframe, open, high, low, close, volume, trades, raw_data)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (exchange, trading_pair, timeframe, timestamp) DO UPDATE
                    SET open = $5, high = $6, low = $7, close = $8, volume = $9, trades = $10, raw_data = $11
                    """,
                    candle.exchange,
                    candle.trading_pair,
                    candle.timestamp,
                    candle.timeframe.value,
                    float(candle.open),
                    float(candle.high),
                    float(candle.low),
                    float(candle.close),
                    float(candle.volume),
                    candle.trades,
                    raw_data
                )
                
            except Exception as e:
                logger.error(f"Erro ao salvar candle: {str(e)}", exc_info=True)
                raise
    
    async def save_candles_batch(self, candles: List[Candle]) -> None:
        """
        Salva um lote de velas no banco de dados.
        
        Args:
            candles: Lista de entidades Candle a serem salvas
            
        Raises:
            Exception: Se ocorrer um erro ao salvar
        """
        if not candles:
            return
            
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Prepara os dados para inserção em batch
                values = []
                import json
                
                for candle in candles:
                    # Converte o raw_data para JSON se não for None
                    raw_data = None
                    if candle.raw_data:
                        try:
                            raw_data = json.dumps(candle.raw_data)
                        except Exception as e:
                            logger.warning(f"Erro ao converter raw_data para JSON: {str(e)}")
                    
                    values.append((
                        candle.exchange,
                        candle.trading_pair,
                        candle.timestamp,
                        candle.timeframe.value,
                        float(candle.open),
                        float(candle.high),
                        float(candle.low),
                        float(candle.close),
                        float(candle.volume),
                        candle.trades,
                        raw_data
                    ))
                
                # Inicia uma transação
                async with conn.transaction():
                    # Insere as velas em batch
                    await conn.executemany(
                        f"""
                        INSERT INTO {self.schema}.candles
                        (exchange, trading_pair, timestamp, timeframe, open, high, low, close, volume, trades, raw_data)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (exchange, trading_pair, timeframe, timestamp) DO UPDATE
                        SET open = $5, high = $6, low = $7, close = $8, volume = $9, trades = $10, raw_data = $11
                        """,
                        values
                    )
                
            except Exception as e:
                logger.error(f"Erro ao salvar lote de candles: {str(e)}", exc_info=True)
                raise
    
    async def get_candles(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Candle]:
        """
        Recupera velas do banco de dados.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timeframe: Timeframe das velas
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de velas a retornar (opcional)
            
        Returns:
            List[Candle]: Lista de entidades Candle
            
        Raises:
            Exception: Se ocorrer um erro ao recuperar os dados
        """
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Constrói a query com os filtros
                query = f"""
                SELECT exchange, trading_pair, timestamp, timeframe, open, high, low, close, volume, trades, raw_data
                FROM {self.schema}.candles
                WHERE exchange = $1 AND trading_pair = $2 AND timeframe = $3
                """
                
                params = [exchange, trading_pair, timeframe.value]
                param_index = 4
                
                if start_time:
                    query += f" AND timestamp >= ${param_index}"
                    params.append(start_time)
                    param_index += 1
                
                if end_time:
                    query += f" AND timestamp <= ${param_index}"
                    params.append(end_time)
                    param_index += 1
                
                # Ordena por timestamp
                query += " ORDER BY timestamp ASC"
                
                if limit:
                    query += f" LIMIT ${param_index}"
                    params.append(limit)
                
                # Executa a query
                rows = await conn.fetch(query, *params)
                
                # Converte os resultados para entidades Candle
                candles = []
                for row in rows:
                    timeframe_value = TimeFrame(row['timeframe'])
                    
                    # Converte JSON para dict se não for None
                    raw_data = None
                    if row['raw_data']:
                        import json
                        raw_data = json.loads(row['raw_data'])
                    
                    candle = Candle(
                        exchange=row['exchange'],
                        trading_pair=row['trading_pair'],
                        timestamp=row['timestamp'],
                        timeframe=timeframe_value,
                        open=Decimal(str(row['open'])),
                        high=Decimal(str(row['high'])),
                        low=Decimal(str(row['low'])),
                        close=Decimal(str(row['close'])),
                        volume=Decimal(str(row['volume'])),
                        trades=row['trades'],
                        raw_data=raw_data
                    )
                    
                    candles.append(candle)
                
                return candles
                
            except Exception as e:
                logger.error(f"Erro ao recuperar candles: {str(e)}", exc_info=True)
                raise
    
    async def save_orderbook(self, orderbook: OrderBook) -> None:
        """
        Salva um orderbook no banco de dados.
        
        Args:
            orderbook: Entidade OrderBook a ser salva
            
        Raises:
            Exception: Se ocorrer um erro ao salvar
        """
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                import json
                
                # Converte os bids e asks para formato JSON
                bids_json = json.dumps([
                    {"price": float(bid.price), "amount": float(bid.amount), "count": bid.count}
                    for bid in orderbook.bids
                ])
                
                asks_json = json.dumps([
                    {"price": float(ask.price), "amount": float(ask.amount), "count": ask.count}
                    for ask in orderbook.asks
                ])
                
                # Converte o raw_data para JSON se não for None
                raw_data = None
                if orderbook.raw_data:
                    try:
                        raw_data = json.dumps(orderbook.raw_data)
                    except Exception as e:
                        logger.warning(f"Erro ao converter raw_data para JSON: {str(e)}")
                
                # Insere o orderbook na tabela
                await conn.execute(
                    f"""
                    INSERT INTO {self.schema}.orderbooks
                    (exchange, trading_pair, timestamp, bids, asks, raw_data)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (exchange, trading_pair, timestamp) DO UPDATE
                    SET bids = $4, asks = $5, raw_data = $6
                    """,
                    orderbook.exchange,
                    orderbook.trading_pair,
                    orderbook.timestamp,
                    bids_json,
                    asks_json,
                    raw_data
                )
                
            except Exception as e:
                logger.error(f"Erro ao salvar orderbook: {str(e)}", exc_info=True)
                raise
    
    async def get_orderbooks(
        self,
        exchange: str,
        trading_pair: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        interval_seconds: Optional[int] = None
    ) -> List[OrderBook]:
        """
        Recupera orderbooks do banco de dados.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            start_time: Timestamp inicial (opcional)
            end_time: Timestamp final (opcional)
            limit: Número máximo de orderbooks a retornar (opcional)
            interval_seconds: Intervalo em segundos entre os orderbooks (opcional)
            
        Returns:
            List[OrderBook]: Lista de entidades OrderBook
            
        Raises:
            Exception: Se ocorrer um erro ao recuperar os dados
        """
        if not self._initialized:
            await self.initialize()
            
        async with self._pool.acquire() as conn:
            try:
                # Constrói a query base
                if interval_seconds:
                    # Com intervalo, usa time_bucket do TimescaleDB se disponível
                    query = f"""
                    SELECT DISTINCT ON (time_bucket('{interval_seconds} seconds', timestamp)) 
                    exchange, trading_pair, timestamp, bids, asks, raw_data
                    FROM {self.schema}.orderbooks
                    WHERE exchange = $1 AND trading_pair = $2
                    """
                else:
                    # Sem intervalo, recupera todos os registros
                    query = f"""
                    SELECT exchange, trading_pair, timestamp, bids, asks, raw_data
                    FROM {self.schema}.orderbooks
                    WHERE exchange = $1 AND trading_pair = $2
                    """
                
                params = [exchange, trading_pair]
                param_index = 3
                
                if start_time:
                    query += f" AND timestamp >= ${param_index}"
                    params.append(start_time)
                    param_index += 1
                
                if end_time:
                    query += f" AND timestamp <= ${param_index}"
                    params.append(end_time)
                    param_index += 1
                
                # Adiciona a ordenação
                if interval_seconds:
                    query += " ORDER BY time_bucket('{interval_seconds} seconds', timestamp), timestamp ASC"
                else:
                    query += " ORDER BY timestamp ASC"
                
                if limit:
                    query += f" LIMIT ${param_index}"
                    params.append(limit)
                
                # Executa a query
                rows = await conn.fetch(query, *params)
                
                # Converte os resultados para entidades OrderBook
                orderbooks = []
                for row in rows:
                    import json
                    
                    # Converte os bids e asks de JSON para objetos
                    bids_data = json.loads(row['bids'])
                    asks_data = json.loads(row['asks'])
                    
                    bids = [
                        OrderBookLevel(
                            price=Decimal(str(bid['price'])),
                            amount=Decimal(str(bid['amount'])),
                            count=bid.get('count')
                        )
                        for bid in bids_data
                    ]
                    
                    asks = [
                        OrderBookLevel(
                            price=Decimal(str(ask['price'])),
                            amount=Decimal(str(ask['amount'])),
                            count=ask.get('count')
                        )
                        for ask in asks_data
                    ]
                    
                    # Converte JSON para dict se não for None
                    raw_data = None
                    if row['raw_data']:
                        raw_data = json.loads(row['raw_data'])
                    
                    orderbook = OrderBook(
                        exchange=row['exchange'],
                        trading_pair=row['trading_pair'],
                        timestamp=row['timestamp'],
                        bids=bids,
                        asks=asks,
                        raw_data=raw_data
                    )
                    
                    orderbooks.append(orderbook)
                
                return orderbooks
                
            except Exception as e:
                logger.error(f"Erro ao recuperar orderbooks: {str(e)}", exc_info=True)
                raise
    
    async def get_latest_candle(
        self,
        exchange: str,
        trading_pair: str,
        timeframe: TimeFrame
    ) -> Optional[Candle]:
        """
        Recupera a vela mais recente do banco de dados.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            timeframe: Timeframe da vela
            
        Returns:
            Optional[Candle]: A vela mais recente ou None se não houver dados
            
        Raises:
            Exception: Se ocorrer um erro ao recuperar os dados
        """
        candles = await self.get_candles(
            exchange=exchange,
            trading_pair=trading_pair,
            timeframe=timeframe,
            limit=1
        )
        
        return candles[0] if candles else None
    
    async def get_latest_orderbook(
        self,
        exchange: str,
        trading_pair: str
    ) -> Optional[OrderBook]:
        """
        Recupera o orderbook mais recente do banco de dados.
        
        Args:
            exchange: Nome da exchange
            trading_pair: Par de negociação
            
        Returns:
            Optional[OrderBook]: O orderbook mais recente ou None se não houver dados
            
        Raises:
            Exception: Se ocorrer um erro ao recuperar os dados
        """
        orderbooks = await self.get_orderbooks(
            exchange=exchange,
            trading_pair=trading_pair,
            limit=1
        )
        
        return orderbooks[0] if orderbooks else None