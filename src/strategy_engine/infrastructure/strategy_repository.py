"""
Strategy Repository - Infrastructure layer for strategy persistence

This module implements repository patterns for strategy, position, and signal
persistence with support for multiple storage backends and caching.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Set
from uuid import UUID
from datetime import datetime, timezone
from decimal import Decimal
import asyncio
import json
import pickle
from dataclasses import asdict

# Import domain entities
from ..domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from ..domain.entities.position import Position, PositionStatus, PositionType
from ..domain.entities.signal import Signal, SignalStatus, SignalType


class RepositoryError(Exception):
    """Base exception for repository operations"""
    pass


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found"""
    pass


class ConcurrencyError(RepositoryError):
    """Raised when there's a concurrency conflict"""
    pass


class IStrategyRepository(ABC):
    """Abstract interface for strategy repository"""
    
    @abstractmethod
    async def save(self, strategy: Strategy) -> None:
        """Save a strategy"""
        pass
    
    @abstractmethod
    async def get_by_id(self, strategy_id: UUID) -> Optional[Strategy]:
        """Get strategy by ID"""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Strategy]:
        """Get strategy by name"""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Strategy]:
        """Get all strategies"""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Get strategies by status"""
        pass
    
    @abstractmethod
    async def get_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """Get strategies by type"""
        pass
    
    @abstractmethod
    async def delete(self, strategy_id: UUID) -> bool:
        """Delete a strategy"""
        pass
    
    @abstractmethod
    async def exists(self, strategy_id: UUID) -> bool:
        """Check if strategy exists"""
        pass


class IPositionRepository(ABC):
    """Abstract interface for position repository"""
    
    @abstractmethod
    async def save(self, position: Position) -> None:
        """Save a position"""
        pass
    
    @abstractmethod
    async def get_by_id(self, position_id: UUID) -> Optional[Position]:
        """Get position by ID"""
        pass
    
    @abstractmethod
    async def get_by_strategy(self, strategy_id: UUID) -> List[Position]:
        """Get positions by strategy ID"""
        pass
    
    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions by symbol"""
        pass
    
    @abstractmethod
    async def get_open_positions(self, strategy_id: Optional[UUID] = None) -> List[Position]:
        """Get open positions, optionally filtered by strategy"""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: PositionStatus) -> List[Position]:
        """Get positions by status"""
        pass
    
    @abstractmethod
    async def delete(self, position_id: UUID) -> bool:
        """Delete a position"""
        pass


class ISignalRepository(ABC):
    """Abstract interface for signal repository"""
    
    @abstractmethod
    async def save(self, signal: Signal) -> None:
        """Save a signal"""
        pass
    
    @abstractmethod
    async def get_by_id(self, signal_id: UUID) -> Optional[Signal]:
        """Get signal by ID"""
        pass
    
    @abstractmethod
    async def get_by_strategy(self, strategy_id: UUID) -> List[Signal]:
        """Get signals by strategy ID"""
        pass
    
    @abstractmethod
    async def get_pending_signals(self, strategy_id: Optional[UUID] = None) -> List[Signal]:
        """Get pending signals, optionally filtered by strategy"""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: SignalStatus) -> List[Signal]:
        """Get signals by status"""
        pass
    
    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> List[Signal]:
        """Get signals by symbol"""
        pass
    
    @abstractmethod
    async def get_recent_signals(
        self, 
        hours: int = 24, 
        strategy_id: Optional[UUID] = None
    ) -> List[Signal]:
        """Get recent signals within specified hours"""
        pass
    
    @abstractmethod
    async def delete(self, signal_id: UUID) -> bool:
        """Delete a signal"""
        pass


class InMemoryStrategyRepository(IStrategyRepository):
    """In-memory implementation of strategy repository for testing/development"""
    
    def __init__(self):
        self._strategies: Dict[UUID, Strategy] = {}
        self._name_index: Dict[str, UUID] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, strategy: Strategy) -> None:
        """Save a strategy"""
        async with self._lock:
            self._strategies[strategy.id] = strategy
            self._name_index[strategy.name] = strategy.id
    
    async def get_by_id(self, strategy_id: UUID) -> Optional[Strategy]:
        """Get strategy by ID"""
        return self._strategies.get(strategy_id)
    
    async def get_by_name(self, name: str) -> Optional[Strategy]:
        """Get strategy by name"""
        strategy_id = self._name_index.get(name)
        if strategy_id:
            return self._strategies.get(strategy_id)
        return None
    
    async def get_all(self) -> List[Strategy]:
        """Get all strategies"""
        return list(self._strategies.values())
    
    async def get_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Get strategies by status"""
        return [s for s in self._strategies.values() if s.status == status]
    
    async def get_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """Get strategies by type"""
        return [s for s in self._strategies.values() if s.strategy_type == strategy_type]
    
    async def delete(self, strategy_id: UUID) -> bool:
        """Delete a strategy"""
        async with self._lock:
            strategy = self._strategies.pop(strategy_id, None)
            if strategy:
                self._name_index.pop(strategy.name, None)
                return True
            return False
    
    async def exists(self, strategy_id: UUID) -> bool:
        """Check if strategy exists"""
        return strategy_id in self._strategies


class InMemoryPositionRepository(IPositionRepository):
    """In-memory implementation of position repository"""
    
    def __init__(self):
        self._positions: Dict[UUID, Position] = {}
        self._strategy_index: Dict[UUID, Set[UUID]] = {}
        self._symbol_index: Dict[str, Set[UUID]] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, position: Position) -> None:
        """Save a position"""
        async with self._lock:
            self._positions[position.id] = position
            
            # Update indexes
            if position.strategy_id not in self._strategy_index:
                self._strategy_index[position.strategy_id] = set()
            self._strategy_index[position.strategy_id].add(position.id)
            
            if position.symbol not in self._symbol_index:
                self._symbol_index[position.symbol] = set()
            self._symbol_index[position.symbol].add(position.id)
    
    async def get_by_id(self, position_id: UUID) -> Optional[Position]:
        """Get position by ID"""
        return self._positions.get(position_id)
    
    async def get_by_strategy(self, strategy_id: UUID) -> List[Position]:
        """Get positions by strategy ID"""
        position_ids = self._strategy_index.get(strategy_id, set())
        return [self._positions[pid] for pid in position_ids if pid in self._positions]
    
    async def get_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions by symbol"""
        position_ids = self._symbol_index.get(symbol, set())
        return [self._positions[pid] for pid in position_ids if pid in self._positions]
    
    async def get_open_positions(self, strategy_id: Optional[UUID] = None) -> List[Position]:
        """Get open positions"""
        positions = self._positions.values()
        if strategy_id:
            positions = [p for p in positions if p.strategy_id == strategy_id]
        return [p for p in positions if p.status == PositionStatus.OPEN]
    
    async def get_by_status(self, status: PositionStatus) -> List[Position]:
        """Get positions by status"""
        return [p for p in self._positions.values() if p.status == status]
    
    async def delete(self, position_id: UUID) -> bool:
        """Delete a position"""
        async with self._lock:
            position = self._positions.pop(position_id, None)
            if position:
                # Clean up indexes
                if position.strategy_id in self._strategy_index:
                    self._strategy_index[position.strategy_id].discard(position_id)
                if position.symbol in self._symbol_index:
                    self._symbol_index[position.symbol].discard(position_id)
                return True
            return False


class InMemorySignalRepository(ISignalRepository):
    """In-memory implementation of signal repository"""
    
    def __init__(self):
        self._signals: Dict[UUID, Signal] = {}
        self._strategy_index: Dict[UUID, Set[UUID]] = {}
        self._symbol_index: Dict[str, Set[UUID]] = {}
        self._lock = asyncio.Lock()
    
    async def save(self, signal: Signal) -> None:
        """Save a signal"""
        async with self._lock:
            self._signals[signal.id] = signal
            
            # Update indexes
            if signal.strategy_id not in self._strategy_index:
                self._strategy_index[signal.strategy_id] = set()
            self._strategy_index[signal.strategy_id].add(signal.id)
            
            if signal.symbol not in self._symbol_index:
                self._symbol_index[signal.symbol] = set()
            self._symbol_index[signal.symbol].add(signal.id)
    
    async def get_by_id(self, signal_id: UUID) -> Optional[Signal]:
        """Get signal by ID"""
        return self._signals.get(signal_id)
    
    async def get_by_strategy(self, strategy_id: UUID) -> List[Signal]:
        """Get signals by strategy ID"""
        signal_ids = self._strategy_index.get(strategy_id, set())
        return [self._signals[sid] for sid in signal_ids if sid in self._signals]
    
    async def get_pending_signals(self, strategy_id: Optional[UUID] = None) -> List[Signal]:
        """Get pending signals"""
        signals = self._signals.values()
        if strategy_id:
            signals = [s for s in signals if s.strategy_id == strategy_id]
        return [s for s in signals if s.status in [SignalStatus.PENDING, SignalStatus.VALIDATED]]
    
    async def get_by_status(self, status: SignalStatus) -> List[Signal]:
        """Get signals by status"""
        return [s for s in self._signals.values() if s.status == status]
    
    async def get_by_symbol(self, symbol: str) -> List[Signal]:
        """Get signals by symbol"""
        signal_ids = self._symbol_index.get(symbol, set())
        return [self._signals[sid] for sid in signal_ids if sid in self._signals]
    
    async def get_recent_signals(
        self, 
        hours: int = 24, 
        strategy_id: Optional[UUID] = None
    ) -> List[Signal]:
        """Get recent signals within specified hours"""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        signals = self._signals.values()
        
        if strategy_id:
            signals = [s for s in signals if s.strategy_id == strategy_id]
        
        return [s for s in signals if s.generated_at.timestamp() >= cutoff]
    
    async def delete(self, signal_id: UUID) -> bool:
        """Delete a signal"""
        async with self._lock:
            signal = self._signals.pop(signal_id, None)
            if signal:
                # Clean up indexes
                if signal.strategy_id in self._strategy_index:
                    self._strategy_index[signal.strategy_id].discard(signal_id)
                if signal.symbol in self._symbol_index:
                    self._symbol_index[signal.symbol].discard(signal_id)
                return True
            return False


class CachedRepository:
    """Caching decorator for repositories to improve performance"""
    
    def __init__(self, repository, cache_ttl: int = 300):  # 5 minutes default TTL
        self._repository = repository
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_ttl = cache_ttl
        self._lock = asyncio.Lock()
    
    def _get_cache_key(self, method: str, *args, **kwargs) -> str:
        """Generate cache key for method call"""
        key_parts = [method] + [str(arg) for arg in args] + [f"{k}={v}" for k, v in sorted(kwargs.items())]
        return "|".join(key_parts)
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cache entry is still valid"""
        if key not in self._cache_timestamps:
            return False
        return (datetime.now().timestamp() - self._cache_timestamps[key]) < self._cache_ttl
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache if valid"""
        async with self._lock:
            if key in self._cache and self._is_cache_valid(key):
                return self._cache[key]
            return None
    
    async def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache"""
        async with self._lock:
            self._cache[key] = value
            self._cache_timestamps[key] = datetime.now().timestamp()
    
    async def _invalidate_cache(self, pattern: str = None) -> None:
        """Invalidate cache entries matching pattern"""
        async with self._lock:
            if pattern:
                keys_to_remove = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self._cache.pop(key, None)
                    self._cache_timestamps.pop(key, None)
            else:
                self._cache.clear()
                self._cache_timestamps.clear()
    
    def __getattr__(self, name):
        """Proxy method calls to underlying repository with caching"""
        attr = getattr(self._repository, name)
        
        if not callable(attr):
            return attr
        
        async def cached_method(*args, **kwargs):
            # Skip caching for write operations
            if name in ['save', 'delete']:
                result = await attr(*args, **kwargs)
                # Invalidate related cache entries
                await self._invalidate_cache()
                return result
            
            # Use caching for read operations
            cache_key = self._get_cache_key(name, *args, **kwargs)
            cached_result = await self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - call original method
            result = await attr(*args, **kwargs)
            await self._set_cache(cache_key, result)
            return result
        
        return cached_method


class RepositoryFactory:
    """Factory for creating repository instances"""
    
    def __init__(self, storage_type: str = "memory", cache_enabled: bool = True, **config):
        self.storage_type = storage_type
        self.cache_enabled = cache_enabled
        self.config = config
    
    def create_strategy_repository(self) -> IStrategyRepository:
        """Create strategy repository instance"""
        if self.storage_type == "memory":
            repo = InMemoryStrategyRepository()
        elif self.storage_type == "postgres":
            repo = PostgreSQLStrategyRepository(**self.config)
        elif self.storage_type == "redis":
            repo = RedisStrategyRepository(**self.config)
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
        
        if self.cache_enabled:
            repo = CachedRepository(repo, cache_ttl=self.config.get("cache_ttl", 300))
        
        return repo
    
    def create_position_repository(self) -> IPositionRepository:
        """Create position repository instance"""
        if self.storage_type == "memory":
            repo = InMemoryPositionRepository()
        elif self.storage_type == "postgres":
            repo = PostgreSQLPositionRepository(**self.config)
        elif self.storage_type == "redis":
            repo = RedisPositionRepository(**self.config)
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
        
        if self.cache_enabled:
            repo = CachedRepository(repo, cache_ttl=self.config.get("cache_ttl", 300))
        
        return repo
    
    def create_signal_repository(self) -> ISignalRepository:
        """Create signal repository instance"""
        if self.storage_type == "memory":
            repo = InMemorySignalRepository()
        elif self.storage_type == "postgres":
            repo = PostgreSQLSignalRepository(**self.config)
        elif self.storage_type == "redis":
            repo = RedisSignalRepository(**self.config)
        else:
            raise ValueError(f"Unknown storage type: {self.storage_type}")
        
        if self.cache_enabled:
            repo = CachedRepository(repo, cache_ttl=self.config.get("cache_ttl", 300))
        
        return repo


class RedisStrategyRepository(IStrategyRepository):
    """Redis-based strategy repository for distributed deployment"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379", **kwargs):
        import redis.asyncio as redis
        self.redis = redis.from_url(redis_url, **kwargs)
        self.key_prefix = "strategy:"
        self.index_prefix = "strategy_index:"
    
    def _get_key(self, strategy_id: UUID) -> str:
        """Get Redis key for strategy"""
        return f"{self.key_prefix}{strategy_id}"
    
    def _get_index_key(self, index_type: str, value: str) -> str:
        """Get Redis key for index"""
        return f"{self.index_prefix}{index_type}:{value}"
    
    async def save(self, strategy: Strategy) -> None:
        """Save strategy to Redis"""
        key = self._get_key(strategy.id)
        data = json.dumps(strategy.to_dict())
        
        pipe = self.redis.pipeline()
        
        # Save strategy data
        pipe.set(key, data)
        
        # Update indexes
        pipe.sadd(self._get_index_key("status", strategy.status.value), str(strategy.id))
        pipe.sadd(self._get_index_key("type", strategy.strategy_type.value), str(strategy.id))
        pipe.sadd(self._get_index_key("name", strategy.name), str(strategy.id))
        pipe.sadd(f"{self.index_prefix}all", str(strategy.id))
        
        await pipe.execute()
    
    async def get_by_id(self, strategy_id: UUID) -> Optional[Strategy]:
        """Get strategy by ID from Redis"""
        key = self._get_key(strategy_id)
        data = await self.redis.get(key)
        
        if data:
            strategy_dict = json.loads(data)
            return Strategy.from_dict(strategy_dict)
        return None
    
    async def get_by_name(self, name: str) -> Optional[Strategy]:
        """Get strategy by name from Redis"""
        strategy_ids = await self.redis.smembers(self._get_index_key("name", name))
        if strategy_ids:
            strategy_id = UUID(strategy_ids.pop().decode())
            return await self.get_by_id(strategy_id)
        return None
    
    async def get_all(self) -> List[Strategy]:
        """Get all strategies from Redis"""
        strategy_ids = await self.redis.smembers(f"{self.index_prefix}all")
        strategies = []
        
        for sid_bytes in strategy_ids:
            strategy_id = UUID(sid_bytes.decode())
            strategy = await self.get_by_id(strategy_id)
            if strategy:
                strategies.append(strategy)
        
        return strategies
    
    async def get_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Get strategies by status from Redis"""
        strategy_ids = await self.redis.smembers(self._get_index_key("status", status.value))
        strategies = []
        
        for sid_bytes in strategy_ids:
            strategy_id = UUID(sid_bytes.decode())
            strategy = await self.get_by_id(strategy_id)
            if strategy:
                strategies.append(strategy)
        
        return strategies
    
    async def get_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """Get strategies by type from Redis"""
        strategy_ids = await self.redis.smembers(self._get_index_key("type", strategy_type.value))
        strategies = []
        
        for sid_bytes in strategy_ids:
            strategy_id = UUID(sid_bytes.decode())
            strategy = await self.get_by_id(strategy_id)
            if strategy:
                strategies.append(strategy)
        
        return strategies
    
    async def delete(self, strategy_id: UUID) -> bool:
        """Delete strategy from Redis"""
        key = self._get_key(strategy_id)
        
        # Get strategy first to clean up indexes
        strategy = await self.get_by_id(strategy_id)
        if not strategy:
            return False
        
        pipe = self.redis.pipeline()
        
        # Delete strategy data
        pipe.delete(key)
        
        # Clean up indexes
        pipe.srem(self._get_index_key("status", strategy.status.value), str(strategy_id))
        pipe.srem(self._get_index_key("type", strategy.strategy_type.value), str(strategy_id))
        pipe.srem(self._get_index_key("name", strategy.name), str(strategy_id))
        pipe.srem(f"{self.index_prefix}all", str(strategy_id))
        
        results = await pipe.execute()
        return results[0] > 0  # First result is from DELETE command
    
    async def exists(self, strategy_id: UUID) -> bool:
        """Check if strategy exists in Redis"""
        key = self._get_key(strategy_id)
        return await self.redis.exists(key) > 0


class PostgreSQLStrategyRepository(IStrategyRepository):
    """PostgreSQL-based strategy repository for production use"""
    
    def __init__(self, database_url: str, **kwargs):
        import asyncpg
        self.database_url = database_url
        self.pool = None
    
    async def _get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            import asyncpg
            self.pool = await asyncpg.create_pool(self.database_url)
        return self.pool
    
    async def save(self, strategy: Strategy) -> None:
        """Save strategy to PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            # Upsert strategy
            await conn.execute("""
                INSERT INTO strategies (
                    id, name, description, version, strategy_type, status, 
                    risk_tolerance, allocated_capital, used_capital, max_capital_usage,
                    created_at, updated_at, started_at, stopped_at, author,
                    module_path, class_name, parameters, state, performance,
                    symbols, exchanges, tags, metadata, required_indicators,
                    required_data_sources, dependencies
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                    $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27
                ) ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    description = EXCLUDED.description,
                    version = EXCLUDED.version,
                    strategy_type = EXCLUDED.strategy_type,
                    status = EXCLUDED.status,
                    risk_tolerance = EXCLUDED.risk_tolerance,
                    allocated_capital = EXCLUDED.allocated_capital,
                    used_capital = EXCLUDED.used_capital,
                    max_capital_usage = EXCLUDED.max_capital_usage,
                    updated_at = EXCLUDED.updated_at,
                    started_at = EXCLUDED.started_at,
                    stopped_at = EXCLUDED.stopped_at,
                    author = EXCLUDED.author,
                    module_path = EXCLUDED.module_path,
                    class_name = EXCLUDED.class_name,
                    parameters = EXCLUDED.parameters,
                    state = EXCLUDED.state,
                    performance = EXCLUDED.performance,
                    symbols = EXCLUDED.symbols,
                    exchanges = EXCLUDED.exchanges,
                    tags = EXCLUDED.tags,
                    metadata = EXCLUDED.metadata,
                    required_indicators = EXCLUDED.required_indicators,
                    required_data_sources = EXCLUDED.required_data_sources,
                    dependencies = EXCLUDED.dependencies
            """, 
                strategy.id, strategy.name, strategy.description, strategy.version,
                strategy.strategy_type.value, strategy.status.value, strategy.risk_tolerance.value,
                strategy.allocated_capital, strategy.used_capital, strategy.max_capital_usage,
                strategy.created_at, strategy.updated_at, strategy.started_at, strategy.stopped_at,
                strategy.author, strategy.module_path, strategy.class_name,
                json.dumps(strategy.parameters.to_dict() if hasattr(strategy.parameters, 'to_dict') else asdict(strategy.parameters)),
                json.dumps(asdict(strategy.state)),
                json.dumps(asdict(strategy.performance)),
                json.dumps(list(strategy.symbols)),
                json.dumps(list(strategy.exchanges)),
                json.dumps(strategy.tags),
                json.dumps(strategy.metadata),
                json.dumps(strategy.required_indicators),
                json.dumps(strategy.required_data_sources),
                json.dumps([str(dep) for dep in strategy.dependencies])
            )
    
    async def get_by_id(self, strategy_id: UUID) -> Optional[Strategy]:
        """Get strategy by ID from PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM strategies WHERE id = $1", 
                strategy_id
            )
            
            if row:
                return self._row_to_strategy(row)
            return None
    
    async def get_by_name(self, name: str) -> Optional[Strategy]:
        """Get strategy by name from PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM strategies WHERE name = $1", 
                name
            )
            
            if row:
                return self._row_to_strategy(row)
            return None
    
    async def get_all(self) -> List[Strategy]:
        """Get all strategies from PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM strategies ORDER BY created_at DESC")
            return [self._row_to_strategy(row) for row in rows]
    
    async def get_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Get strategies by status from PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM strategies WHERE status = $1 ORDER BY created_at DESC", 
                status.value
            )
            return [self._row_to_strategy(row) for row in rows]
    
    async def get_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """Get strategies by type from PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM strategies WHERE strategy_type = $1 ORDER BY created_at DESC", 
                strategy_type.value
            )
            return [self._row_to_strategy(row) for row in rows]
    
    async def delete(self, strategy_id: UUID) -> bool:
        """Delete strategy from PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM strategies WHERE id = $1", 
                strategy_id
            )
            return result == "DELETE 1"
    
    async def exists(self, strategy_id: UUID) -> bool:
        """Check if strategy exists in PostgreSQL"""
        pool = await self._get_connection()
        
        async with pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM strategies WHERE id = $1)", 
                strategy_id
            )
            return result
    
    def _row_to_strategy(self, row) -> Strategy:
        """Convert database row to Strategy object"""
        from ..domain.entities.strategy import (
            StrategyParameters, StrategyState, PerformanceMetrics,
            StrategyType, StrategyStatus, RiskTolerance
        )
        
        # Parse JSON fields
        parameters_data = json.loads(row['parameters'])
        state_data = json.loads(row['state'])
        performance_data = json.loads(row['performance'])
        
        # Reconstruct objects
        parameters = StrategyParameters(**parameters_data)
        state = StrategyState(**state_data)
        performance = PerformanceMetrics(**performance_data)
        
        return Strategy(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            version=row['version'],
            strategy_type=StrategyType(row['strategy_type']),
            status=StrategyStatus(row['status']),
            risk_tolerance=RiskTolerance(row['risk_tolerance']),
            parameters=parameters,
            symbols=set(json.loads(row['symbols'])),
            exchanges=set(json.loads(row['exchanges'])),
            state=state,
            performance=performance,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            started_at=row['started_at'],
            stopped_at=row['stopped_at'],
            allocated_capital=row['allocated_capital'],
            used_capital=row['used_capital'],
            max_capital_usage=row['max_capital_usage'],
            author=row['author'],
            tags=json.loads(row['tags']),
            metadata=json.loads(row['metadata']),
            required_indicators=json.loads(row['required_indicators']),
            required_data_sources=json.loads(row['required_data_sources']),
            dependencies=[UUID(dep) for dep in json.loads(row['dependencies'])],
            module_path=row['module_path'],
            class_name=row['class_name']
        )


# Repository implementations for Position and Signal would follow similar patterns
class RedisPositionRepository(IPositionRepository):
    """Redis implementation of position repository"""
    # Implementation similar to RedisStrategyRepository
    pass


class RedisSignalRepository(ISignalRepository):
    """Redis implementation of signal repository"""
    # Implementation similar to RedisStrategyRepository
    pass


class PostgreSQLPositionRepository(IPositionRepository):
    """PostgreSQL implementation of position repository"""
    # Implementation similar to PostgreSQLStrategyRepository
    pass


class PostgreSQLSignalRepository(ISignalRepository):
    """PostgreSQL implementation of signal repository"""
    # Implementation similar to PostgreSQLStrategyRepository
    pass