"""
Main Application - Strategy Engine Entry Point

This module provides the main application orchestration, dependency injection,
and startup configuration for the strategy engine system.
"""

from __future__ import annotations
import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4

# FastAPI and web framework imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import strategy engine components
from .domain.entities.strategy import Strategy, StrategyStatus, StrategyType, RiskTolerance
from .domain.entities.position import Position, PositionStatus
from .domain.entities.signal import Signal, SignalStatus, SignalDirection, SignalType
from .infrastructure.strategy_repository import RepositoryFactory
from .infrastructure.signal_publisher import SignalPublisherFactory
from .application.services.strategy_executor_service import StrategyExecutorService
from .application.services.backtest_service import BacktestService
from .application.use_cases.execute_strategy_use_case import ExecuteStrategyUseCase
from .application.use_cases.backtest_strategy_use_case import BacktestStrategyUseCase
from .application.use_cases.execute_strategy_use_case import ExecuteStrategyRequest
from .application.use_cases.backtest_strategy_use_case import BacktestStrategyRequest


# Pydantic models for API requests/responses
class CreateStrategyRequest(BaseModel):
    name: str = Field(..., description="Strategy name")
    description: str = Field("", description="Strategy description")
    strategy_type: str = Field(..., description="Strategy type")
    symbols: List[str] = Field(..., description="Trading symbols")
    exchanges: List[str] = Field(..., description="Exchanges to use")
    allocated_capital: float = Field(10000, description="Allocated capital")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")
    module_path: str = Field("", description="Strategy implementation module path")
    class_name: str = Field("", description="Strategy implementation class name")


class UpdateStrategyRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    symbols: Optional[List[str]] = None
    exchanges: Optional[List[str]] = None
    allocated_capital: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None


class ExecuteStrategyAPIRequest(BaseModel):
    market_data: Optional[Dict[str, Any]] = None
    force_start: bool = False
    paper_trading: bool = True


class BacktestStrategyAPIRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    initial_capital: float = 100000
    symbols: Optional[List[str]] = None
    parameters_override: Optional[Dict[str, Any]] = None
    include_slippage: bool = True
    include_fees: bool = True
    benchmark_symbol: Optional[str] = None


class StrategyResponse(BaseModel):
    id: str
    name: str
    description: str
    strategy_type: str
    status: str
    allocated_capital: float
    used_capital: float
    symbols: List[str]
    exchanges: List[str]
    created_at: datetime
    updated_at: datetime


class PositionResponse(BaseModel):
    id: str
    strategy_id: str
    symbol: str
    position_type: str
    status: str
    quantity: float
    average_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float


class SignalResponse(BaseModel):
    id: str
    strategy_id: str
    symbol: str
    direction: str
    signal_type: str
    quantity: float
    current_price: float
    status: str
    generated_at: datetime


class StrategyEngineApp:
    """Main application class for the Strategy Engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Core components
        self.repository_factory = None
        self.signal_publisher = None
        self.strategy_executor = None
        self.backtest_service = None
        self.execute_strategy_use_case = None
        self.backtest_strategy_use_case = None
        
        # Application state
        self.is_running = False
        self.background_tasks = set()
        
        # FastAPI app
        self.app = FastAPI(
            title="Neural Crypto Bot - Strategy Engine",
            description="Advanced cryptocurrency trading bot with ML capabilities",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self._setup_middleware()
        self._setup_routes()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "storage": {
                "type": "memory",  # memory, redis, postgres
                "cache_enabled": True,
                "cache_ttl": 300
            },
            "publisher": {
                "type": "composite",  # memory, kafka, redis, websocket, composite
                "kafka": {
                    "bootstrap_servers": "localhost:9092",
                    "signal_topic": "trading-signals",
                    "event_topic": "domain-events"
                },
                "redis": {
                    "redis_url": "redis://localhost:6379",
                    "signal_stream": "signals",
                    "event_stream": "events"
                },
                "websocket": {
                    "host": "localhost",
                    "port": 8765
                }
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "reload": False,
                "cors_origins": ["*"]
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup application logging"""
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"]
        )
        return logging.getLogger(__name__)
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config["api"]["cors_origins"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "Neural Crypto Bot - Strategy Engine",
                "version": "1.0.0",
                "status": "running" if self.is_running else "stopped",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.get("/health")
        async def health_check():
            health_status = {
                "status": "healthy" if self.is_running else "unhealthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {}
            }
            
            if self.signal_publisher:
                health_status["components"]["signal_publisher"] = await self.signal_publisher.health_check()
            
            return health_status
        
        # Strategy management routes
        @self.app.post("/strategies", response_model=StrategyResponse)
        async def create_strategy(request: CreateStrategyRequest):
            try:
                strategy = Strategy(
                    name=request.name,
                    description=request.description,
                    strategy_type=StrategyType(request.strategy_type.upper()),
                    symbols=set(request.symbols),
                    exchanges=set(request.exchanges),
                    allocated_capital=Decimal(str(request.allocated_capital)),
                    module_path=request.module_path,
                    class_name=request.class_name
                )
                
                # Apply parameters if provided
                if request.parameters:
                    for key, value in request.parameters.items():
                        if hasattr(strategy.parameters, key):
                            setattr(strategy.parameters, key, value)
                
                await self.repository_factory.create_strategy_repository().save(strategy)
                
                return StrategyResponse(
                    id=str(strategy.id),
                    name=strategy.name,
                    description=strategy.description,
                    strategy_type=strategy.strategy_type.value,
                    status=strategy.status.value,
                    allocated_capital=float(strategy.allocated_capital),
                    used_capital=float(strategy.used_capital),
                    symbols=list(strategy.symbols),
                    exchanges=list(strategy.exchanges),
                    created_at=strategy.created_at,
                    updated_at=strategy.updated_at
                )
                
            except Exception as e:
                self.logger.error(f"Error creating strategy: {e}")
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/strategies", response_model=List[StrategyResponse])
        async def list_strategies():
            try:
                strategies = await self.repository_factory.create_strategy_repository().get_all()
                return [
                    StrategyResponse(
                        id=str(s.id),
                        name=s.name,
                        description=s.description,
                        strategy_type=s.strategy_type.value,
                        status=s.status.value,
                        allocated_capital=float(s.allocated_capital),
                        used_capital=float(s.used_capital),
                        symbols=list(s.symbols),
                        exchanges=list(s.exchanges),
                        created_at=s.created_at,
                        updated_at=s.updated_at
                    )
                    for s in strategies
                ]
            except Exception as e:
                self.logger.error(f"Error listing strategies: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/{strategy_id}", response_model=StrategyResponse)
        async def get_strategy(strategy_id: str):
            try:
                strategy = await self.repository_factory.create_strategy_repository().get_by_id(UUID(strategy_id))
                if not strategy:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                return StrategyResponse(
                    id=str(strategy.id),
                    name=strategy.name,
                    description=strategy.description,
                    strategy_type=strategy.strategy_type.value,
                    status=strategy.status.value,
                    allocated_capital=float(strategy.allocated_capital),
                    used_capital=float(strategy.used_capital),
                    symbols=list(strategy.symbols),
                    exchanges=list(strategy.exchanges),
                    created_at=strategy.created_at,
                    updated_at=strategy.updated_at
                )
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error getting strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.put("/strategies/{strategy_id}", response_model=StrategyResponse)
        async def update_strategy(strategy_id: str, request: UpdateStrategyRequest):
            try:
                strategy_repo = self.repository_factory.create_strategy_repository()
                strategy = await strategy_repo.get_by_id(UUID(strategy_id))
                if not strategy:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                # Update fields if provided
                if request.name is not None:
                    strategy.name = request.name
                if request.description is not None:
                    strategy.description = request.description
                if request.symbols is not None:
                    strategy.symbols = set(request.symbols)
                if request.exchanges is not None:
                    strategy.exchanges = set(request.exchanges)
                if request.allocated_capital is not None:
                    strategy.allocated_capital = Decimal(str(request.allocated_capital))
                if request.parameters is not None:
                    for key, value in request.parameters.items():
                        if hasattr(strategy.parameters, key):
                            setattr(strategy.parameters, key, value)
                
                strategy.updated_at = datetime.now(timezone.utc)
                await strategy_repo.save(strategy)
                
                return StrategyResponse(
                    id=str(strategy.id),
                    name=strategy.name,
                    description=strategy.description,
                    strategy_type=strategy.strategy_type.value,
                    status=strategy.status.value,
                    allocated_capital=float(strategy.allocated_capital),
                    used_capital=float(strategy.used_capital),
                    symbols=list(strategy.symbols),
                    exchanges=list(strategy.exchanges),
                    created_at=strategy.created_at,
                    updated_at=strategy.updated_at
                )
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error updating strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/strategies/{strategy_id}")
        async def delete_strategy(strategy_id: str):
            try:
                # Stop strategy if running
                if self.strategy_executor:
                    await self.strategy_executor.stop_strategy(UUID(strategy_id), "Strategy deleted")
                
                # Delete from repository
                success = await self.repository_factory.create_strategy_repository().delete(UUID(strategy_id))
                if not success:
                    raise HTTPException(status_code=404, detail="Strategy not found")
                
                return {"message": "Strategy deleted successfully"}
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error deleting strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Strategy execution routes
        @self.app.post("/strategies/{strategy_id}/execute")
        async def execute_strategy(strategy_id: str, request: ExecuteStrategyAPIRequest):
            try:
                use_case_request = ExecuteStrategyRequest(
                    strategy_id=UUID(strategy_id),
                    market_data=request.market_data,
                    force_start=request.force_start,
                    paper_trading=request.paper_trading
                )
                
                response = await self.execute_strategy_use_case.execute(use_case_request)
                
                if not response.success:
                    raise HTTPException(status_code=400, detail={"errors": response.errors, "warnings": response.warnings})
                
                return {
                    "success": True,
                    "strategy_id": str(response.strategy_id),
                    "execution_result": response.execution_result.to_dict() if response.execution_result else None,
                    "warnings": response.warnings
                }
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error executing strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/strategies/{strategy_id}/stop")
        async def stop_strategy(strategy_id: str, reason: str = "Manual stop"):
            try:
                response = await self.execute_strategy_use_case.stop_strategy(UUID(strategy_id), reason)
                
                if not response.success:
                    raise HTTPException(status_code=400, detail={"errors": response.errors})
                
                return {
                    "success": True,
                    "strategy_id": str(response.strategy_id),
                    "message": f"Strategy stopped: {reason}"
                }
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error stopping strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/strategies/{strategy_id}/status")
        async def get_strategy_status(strategy_id: str):
            try:
                status = await self.execute_strategy_use_case.get_strategy_status(UUID(strategy_id))
                return status
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error getting strategy status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Backtesting routes
        @self.app.post("/strategies/{strategy_id}/backtest")
        async def backtest_strategy(strategy_id: str, request: BacktestStrategyAPIRequest):
            try:
                use_case_request = BacktestStrategyRequest(
                    strategy_id=UUID(strategy_id),
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=Decimal(str(request.initial_capital)),
                    symbols=request.symbols,
                    parameters_override=request.parameters_override,
                    include_slippage=request.include_slippage,
                    include_fees=request.include_fees,
                    benchmark_symbol=request.benchmark_symbol
                )
                
                response = await self.backtest_strategy_use_case.execute(use_case_request)
                
                if not response.success:
                    raise HTTPException(status_code=400, detail={"errors": response.errors, "warnings": response.warnings})
                
                return {
                    "success": True,
                    "strategy_id": str(response.strategy_id),
                    "backtest_result": response.backtest_result.to_dict() if response.backtest_result else None,
                    "warnings": response.warnings
                }
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error backtesting strategy: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Position management routes
        @self.app.get("/strategies/{strategy_id}/positions", response_model=List[PositionResponse])
        async def get_strategy_positions(strategy_id: str):
            try:
                positions = await self.repository_factory.create_position_repository().get_by_strategy(UUID(strategy_id))
                return [
                    PositionResponse(
                        id=str(p.id),
                        strategy_id=str(p.strategy_id),
                        symbol=p.symbol,
                        position_type=p.position_type.value,
                        status=p.status.value,
                        quantity=float(p.quantity),
                        average_entry_price=float(p.average_entry_price),
                        current_price=float(p.current_price),
                        unrealized_pnl=float(p.unrealized_pnl),
                        realized_pnl=float(p.realized_pnl),
                        total_pnl=float(p.total_pnl)
                    )
                    for p in positions
                ]
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error getting strategy positions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Signal management routes
        @self.app.get("/strategies/{strategy_id}/signals", response_model=List[SignalResponse])
        async def get_strategy_signals(strategy_id: str, hours: int = 24):
            try:
                signals = await self.repository_factory.create_signal_repository().get_recent_signals(hours, UUID(strategy_id))
                return [
                    SignalResponse(
                        id=str(s.id),
                        strategy_id=str(s.strategy_id),
                        symbol=s.symbol,
                        direction=s.direction.value,
                        signal_type=s.signal_type.value,
                        quantity=float(s.quantity),
                        current_price=float(s.current_price),
                        status=s.status.value,
                        generated_at=s.generated_at
                    )
                    for s in signals
                ]
                
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid strategy ID")
            except Exception as e:
                self.logger.error(f"Error getting strategy signals: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def initialize(self):
        """Initialize application components"""
        try:
            self.logger.info("Initializing Strategy Engine...")
            
            # Create repository factory
            self.repository_factory = RepositoryFactory(
                storage_type=self.config["storage"]["type"],
                cache_enabled=self.config["storage"]["cache_enabled"],
                **self.config["storage"]
            )
            
            # Create signal publisher
            publisher_config = self.config["publisher"]
            if publisher_config["type"] == "composite":
                # Configure composite publisher with multiple backends
                publishers_config = {
                    "publishers": [
                        {"type": "memory"},
                        {"type": "websocket", **publisher_config.get("websocket", {})},
                    ]
                }
                # Add Kafka and Redis if configured
                if "kafka" in publisher_config:
                    publishers_config["publishers"].append({"type": "kafka", **publisher_config["kafka"]})
                if "redis" in publisher_config:
                    publishers_config["publishers"].append({"type": "redis", **publisher_config["redis"]})
                
                self.signal_publisher = SignalPublisherFactory.create_publisher("composite", **publishers_config)
            else:
                self.signal_publisher = SignalPublisherFactory.create_publisher(
                    publisher_config["type"], 
                    **publisher_config.get(publisher_config["type"], {})
                )
            
            # Initialize signal publisher
            await self.signal_publisher.start()
            
            # Create application services
            self.strategy_executor = StrategyExecutorService(
                strategy_repository=self.repository_factory.create_strategy_repository(),
                position_repository=self.repository_factory.create_position_repository(),
                signal_repository=self.repository_factory.create_signal_repository(),
                signal_publisher=self.signal_publisher
            )
            
            self.backtest_service = BacktestService(
                strategy_repository=self.repository_factory.create_strategy_repository(),
                position_repository=self.repository_factory.create_position_repository(),
                signal_repository=self.repository_factory.create_signal_repository()
            )
            
            # Create use cases
            self.execute_strategy_use_case = ExecuteStrategyUseCase(self.strategy_executor)
            self.backtest_strategy_use_case = BacktestStrategyUseCase(self.backtest_service)
            
            self.is_running = True
            self.logger.info("Strategy Engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Strategy Engine: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown application components"""
        try:
            self.logger.info("Shutting down Strategy Engine...")
            self.is_running = False
            
            # Stop all running strategies
            if self.strategy_executor:
                strategies = await self.repository_factory.create_strategy_repository().get_by_status(StrategyStatus.ACTIVE)
                for strategy in strategies:
                    await self.strategy_executor.stop_strategy(strategy.id, "Application shutdown")
            
            # Stop signal publisher
            if self.signal_publisher:
                await self.signal_publisher.stop()
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            self.logger.info("Strategy Engine shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def run(self):
        """Run the application"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.initialize()
            yield
            # Shutdown
            await self.shutdown()
        
        # Set lifespan for FastAPI app
        self.app.router.lifespan_context = lifespan
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run the application
        uvicorn.run(
            self.app,
            host=self.config["api"]["host"],
            port=self.config["api"]["port"],
            reload=self.config["api"]["reload"],
            log_level=self.config["logging"]["level"].lower()
        )


# Extension methods for BacktestResult and ExecutionResult
def backtest_result_to_dict(self) -> Dict[str, Any]:
    """Convert BacktestResult to dictionary"""
    return {
        "strategy_id": str(self.strategy_id),
        "start_date": self.start_date.isoformat(),
        "end_date": self.end_date.isoformat(),
        "initial_capital": float(self.initial_capital),
        "final_capital": float(self.final_capital),
        "total_return": float(self.total_return),
        "total_return_pct": float(self.total_return_pct),
        "sharpe_ratio": float(self.sharpe_ratio),
        "max_drawdown": float(self.max_drawdown),
        "win_rate": float(self.win_rate),
        "total_trades": self.total_trades,
        "winning_trades": self.winning_trades,
        "losing_trades": self.losing_trades,
        "avg_trade_return": float(self.avg_trade_return),
        "volatility": float(self.volatility),
        "trades": self.trades,
        "equity_curve": self.equity_curve,
        "performance_metrics": self.performance_metrics
    }

def execution_result_to_dict(self) -> Dict[str, Any]:
    """Convert ExecutionResult to dictionary"""
    return {
        "strategy_id": str(self.strategy_id),
        "signals_generated": self.signals_generated,
        "signals_executed": self.signals_executed,
        "positions_opened": self.positions_opened,
        "positions_closed": self.positions_closed,
        "total_pnl": float(self.total_pnl),
        "execution_time": self.execution_time,
        "errors": self.errors,
        "warnings": self.warnings
    }

# Monkey patch the to_dict methods
from .application.services.backtest_service import BacktestResult
from .application.services.strategy_executor_service import ExecutionResult
BacktestResult.to_dict = backtest_result_to_dict
ExecutionResult.to_dict = execution_result_to_dict


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Crypto Bot - Strategy Engine")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    if not config:
        config = {}
    
    if "api" not in config:
        config["api"] = {}
    
    config["api"]["host"] = args.host
    config["api"]["port"] = args.port
    config["api"]["reload"] = args.reload
    
    if "logging" not in config:
        config["logging"] = {}
    config["logging"]["level"] = args.log_level.upper()
    
    # Create and run application
    app = StrategyEngineApp(config)
    app.run()


if __name__ == "__main__":
    main()