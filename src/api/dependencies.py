"""
Neural Crypto Bot - API Dependencies

Este módulo gerencia as dependências injetadas nos controllers da API.
Implementa o padrão de injeção de dependência usando FastAPI.
"""
from typing import Annotated, AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from analytics.application.services.analytics_service import AnalyticsService
from common.infrastructure.database.unit_of_work import UnitOfWork
from common.utils.config import get_settings
from execution_service.application.services.execution_optimization_service import (
    ExecutionOptimizationService,
)
from execution_service.application.services.order_routing_service import OrderRoutingService
from execution_service.application.use_cases.execute_order_use_case import ExecuteOrderUseCase
from strategy_engine.application.services.backtest_service import BacktestService
from strategy_engine.application.services.strategy_executor_service import StrategyExecutorService
from strategy_engine.application.use_cases.backtest_strategy_use_case import BacktestStrategyUseCase
from strategy_engine.application.use_cases.execute_strategy_use_case import ExecuteStrategyUseCase
from strategy_engine.infrastructure.strategy_repository import StrategyRepository

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Injeção de dependência para o token de autenticação
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Obtém o usuário atual a partir do token de autenticação.
    
    Args:
        token: Token de autenticação JWT.
        
    Returns:
        Objeto de usuário autenticado.
        
    Raises:
        HTTPException: Se o token for inválido ou expirado.
    """
    # Implementação de verificação de token e recuperação de usuário
    # Em um sistema real, verificaria a validade do token JWT e buscaria o usuário
    # No momento, apenas simulamos o comportamento básico
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciais de autenticação não fornecidas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Mock de usuário para desenvolvimento
    return {"username": "test_user", "email": "test@example.com", "roles": ["trader"]}

# UnitOfWork para transações de banco de dados
async def get_unit_of_work() -> AsyncGenerator[UnitOfWork, None]:
    """
    Cria e gerencia um UnitOfWork para operações transacionais.
    
    Yields:
        UnitOfWork: Objeto para gerenciar transações de banco de dados.
    """
    async with UnitOfWork() as uow:
        yield uow

# Repository dependencies
async def get_strategy_repository(
    uow: UnitOfWork = Depends(get_unit_of_work)
) -> StrategyRepository:
    """
    Fornece um repositório de estratégias.
    
    Args:
        uow: Unit of Work para gerenciar transações.
        
    Returns:
        StrategyRepository: Repositório para operações com estratégias.
    """
    return StrategyRepository(uow)

# Service dependencies
async def get_strategy_executor_service() -> StrategyExecutorService:
    """
    Fornece o serviço de execução de estratégias.
    
    Returns:
        StrategyExecutorService: Serviço para execução de estratégias.
    """
    return StrategyExecutorService()

async def get_backtest_service() -> BacktestService:
    """
    Fornece o serviço de backtesting.
    
    Returns:
        BacktestService: Serviço para backtesting de estratégias.
    """
    return BacktestService()

async def get_analytics_service() -> AnalyticsService:
    """
    Fornece o serviço de analytics.
    
    Returns:
        AnalyticsService: Serviço para análise de desempenho.
    """
    return AnalyticsService()

async def get_order_routing_service() -> OrderRoutingService:
    """
    Fornece o serviço de roteamento de ordens.
    
    Returns:
        OrderRoutingService: Serviço para roteamento otimizado de ordens.
    """
    return OrderRoutingService()

async def get_execution_optimization_service() -> ExecutionOptimizationService:
    """
    Fornece o serviço de otimização de execução.
    
    Returns:
        ExecutionOptimizationService: Serviço para otimização de execução de ordens.
    """
    return ExecutionOptimizationService()

# Use case dependencies
async def get_execute_strategy_use_case(
    strategy_executor_service: StrategyExecutorService = Depends(get_strategy_executor_service),
) -> ExecuteStrategyUseCase:
    """
    Fornece o caso de uso para execução de estratégias.
    
    Args:
        strategy_executor_service: Serviço de execução de estratégias.
        
    Returns:
        ExecuteStrategyUseCase: Caso de uso para execução de estratégias.
    """
    return ExecuteStrategyUseCase(strategy_executor_service)

async def get_backtest_strategy_use_case(
    backtest_service: BacktestService = Depends(get_backtest_service),
) -> BacktestStrategyUseCase:
    """
    Fornece o caso de uso para backtesting de estratégias.
    
    Args:
        backtest_service: Serviço de backtesting.
        
    Returns:
        BacktestStrategyUseCase: Caso de uso para backtesting de estratégias.
    """
    return BacktestStrategyUseCase(backtest_service)

async def get_execute_order_use_case(
    order_routing_service: OrderRoutingService = Depends(get_order_routing_service),
    execution_optimization_service: ExecutionOptimizationService = Depends(get_execution_optimization_service),
) -> ExecuteOrderUseCase:
    """
    Fornece o caso de uso para execução de ordens.
    
    Args:
        order_routing_service: Serviço de roteamento de ordens.
        execution_optimization_service: Serviço de otimização de execução.
        
    Returns:
        ExecuteOrderUseCase: Caso de uso para execução de ordens.
    """
    return ExecuteOrderUseCase(order_routing_service, execution_optimization_service)

# Tipos anotados comuns para uso nos controllers
CurrentUser = Annotated[dict, Depends(get_current_user)]
StrategyRepositoryDep = Annotated[StrategyRepository, Depends(get_strategy_repository)]
ExecuteStrategyUCDep = Annotated[ExecuteStrategyUseCase, Depends(get_execute_strategy_use_case)]
BacktestStrategyUCDep = Annotated[BacktestStrategyUseCase, Depends(get_backtest_strategy_use_case)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
ExecuteOrderUCDep = Annotated[ExecuteOrderUseCase, Depends(get_execute_order_use_case)]