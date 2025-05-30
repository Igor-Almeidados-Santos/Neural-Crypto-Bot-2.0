"""
Módulo principal do serviço de execução.

Este módulo inicializa e configura o serviço de execução, e
fornece endpoints para gerenciamento de ordens e execuções.
"""
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.execution_service.application.services.execution_optimization_service import (
    ExecutionOptimizationService,
)
from src.execution_service.application.services.order_routing_service import (
    OrderRoutingService,
)
from src.execution_service.application.use_cases.execute_order_use_case import (
    ExecuteOrderUseCase,
)
from src.execution_service.domain.entities.order import Order
from src.execution_service.domain.value_objects.execution_parameters import (
    create_execution_parameters,
)
from src.execution_service.infrastructure.exchange_client import ExchangeClient
from src.execution_service.infrastructure.execution_event_publisher import (
    ExecutionEventPublisher,
)
from src.execution_service.infrastructure.order_repository import OrderRepository

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OrderRequest(BaseModel):
    """Modelo para solicitação de execução de ordem."""
    
    trading_pair: str = Field(..., description="Par de trading (ex: BTC/USDT)")
    side: str = Field(..., description="Lado da ordem ('buy' ou 'sell')")
    order_type: str = Field(..., description="Tipo da ordem ('market' ou 'limit')")
    quantity: float = Field(..., gt=0, description="Quantidade a ser comprada/vendida")
    price: Optional[float] = Field(None, description="Preço para ordens limit")
    exchange: str = Field("auto", description="Exchange para execução (ou 'auto' para roteamento)")
    algorithm: Optional[str] = Field(None, description="Algoritmo de execução")
    algorithm_params: Optional[Dict] = Field(None, description="Parâmetros do algoritmo")
    strategy_id: Optional[str] = Field(None, description="ID da estratégia")
    metadata: Optional[Dict] = Field({}, description="Metadados adicionais")


class OrderResponse(BaseModel):
    """Modelo para resposta de execução de ordem."""
    
    id: str
    trading_pair: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    average_price: Optional[float]
    fees: float
    exchange: str
    exchange_order_id: Optional[str]
    strategy_id: Optional[str]
    executed_at: Optional[str]
    error_message: Optional[str]
    child_orders_count: int


class CancelOrderRequest(BaseModel):
    """Modelo para solicitação de cancelamento de ordem."""
    
    exchange: str = Field(..., description="Exchange onde a ordem foi colocada")
    order_id: str = Field(..., description="ID da ordem na exchange")
    trading_pair: str = Field(..., description="Par de trading")


class OrderStatusRequest(BaseModel):
    """Modelo para solicitação de status de ordem."""
    
    exchange: str = Field(..., description="Exchange onde a ordem foi colocada")
    order_id: str = Field(..., description="ID da ordem na exchange")
    trading_pair: str = Field(..., description="Par de trading")


class ServiceStatus(BaseModel):
    """Modelo para status do serviço."""
    
    status: str
    version: str
    exchanges: List[str]
    algorithms: List[str]


# Instância global do FastAPI
app = FastAPI(
    title="Execution Service API",
    description="Serviço de execução de ordens para o sistema de trading",
    version="1.0.0",
)

# Adicionar middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais para componentes do serviço
exchange_client = None
order_repository = None
order_routing_service = None
execution_optimization_service = None
event_publisher = None
execute_order_use_case = None


async def get_exchange_client():
    """Dependency para obter o cliente de exchanges."""
    return exchange_client


async def get_order_repository():
    """Dependency para obter o repositório de ordens."""
    return order_repository


async def get_order_routing_service():
    """Dependency para obter o serviço de roteamento de ordens."""
    return order_routing_service


async def get_execution_optimization_service():
    """Dependency para obter o serviço de otimização de execução."""
    return execution_optimization_service


async def get_event_publisher():
    """Dependency para obter o publisher de eventos."""
    return event_publisher


async def get_execute_order_use_case():
    """Dependency para obter o caso de uso de execução de ordens."""
    return execute_order_use_case


@app.post("/orders", response_model=OrderResponse, tags=["Execution"])
async def execute_order(
    order_request: OrderRequest,
    use_case: ExecuteOrderUseCase = Depends(get_execute_order_use_case),
):
    """
    Executa uma ordem de trading.
    
    Args:
        order_request: Dados da ordem a ser executada.
        use_case: Caso de uso de execução de ordens.
        
    Returns:
        OrderResponse: Dados da ordem após a execução.
        
    Raises:
        HTTPException: Se houver um erro na execução.
    """
    try:
        # Criar objeto Order a partir da requisição
        order = Order(
            trading_pair=order_request.trading_pair,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=order_request.quantity,
            price=order_request.price,
            exchange=order_request.exchange,
            strategy_id=order_request.strategy_id,
            status="pending",
            metadata=order_request.metadata or {},
        )
        
        # Adicionar algoritmo e parâmetros aos metadados, se especificados
        if order_request.algorithm:
            order.metadata["algorithm"] = order_request.algorithm
        if order_request.algorithm_params:
            order.metadata["execution_params"] = order_request.algorithm_params
        
        # Executar a ordem
        result = await use_case.execute(order)
        
        # Converter para resposta
        return OrderResponse(
            id=result.id,
            trading_pair=result.trading_pair,
            side=result.side,
            order_type=result.order_type,
            quantity=result.quantity,
            price=result.price,
            status=result.status,
            filled_quantity=result.filled_quantity,
            average_price=result.average_price,
            fees=result.fees,
            exchange=result.exchange,
            exchange_order_id=result.exchange_order_id,
            strategy_id=result.strategy_id,
            executed_at=result.executed_at.isoformat() if result.executed_at else None,
            error_message=result.error_message,
            child_orders_count=len(result.child_orders),
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erro ao executar ordem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/orders/{order_id}", response_model=OrderResponse, tags=["Execution"])
async def get_order(
    order_id: str = Path(..., description="ID da ordem"),
    repository: OrderRepository = Depends(get_order_repository),
):
    """
    Obtém os detalhes de uma ordem pelo ID.
    
    Args:
        order_id: ID da ordem.
        repository: Repositório de ordens.
        
    Returns:
        OrderResponse: Dados da ordem.
        
    Raises:
        HTTPException: Se a ordem não for encontrada.
    """
    order = await repository.get_by_id(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Ordem não encontrada")
    
    return OrderResponse(
        id=order.id,
        trading_pair=order.trading_pair,
        side=order.side,
        order_type=order.order_type,
        quantity=order.quantity,
        price=order.price,
        status=order.status,
        filled_quantity=order.filled_quantity,
        average_price=order.average_price,
        fees=order.fees,
        exchange=order.exchange,
        exchange_order_id=order.exchange_order_id,
        strategy_id=order.strategy_id,
        executed_at=order.executed_at.isoformat() if order.executed_at else None,
        error_message=order.error_message,
        child_orders_count=len(order.child_orders),
    )


@app.get("/orders", response_model=List[OrderResponse], tags=["Execution"])
async def list_orders(
    page: int = Query(1, ge=1, description="Número da página"),
    size: int = Query(10, ge=1, le=100, description="Tamanho da página"),
    status: Optional[str] = Query(None, description="Filtrar por status"),
    trading_pair: Optional[str] = Query(None, description="Filtrar por par de trading"),
    exchange: Optional[str] = Query(None, description="Filtrar por exchange"),
    strategy_id: Optional[str] = Query(None, description="Filtrar por estratégia"),
    repository: OrderRepository = Depends(get_order_repository),
):
    """
    Lista ordens com paginação e filtros.
    
    Args:
        page: Número da página.
        size: Tamanho da página.
        status: Filtrar por status.
        trading_pair: Filtrar por par de trading.
        exchange: Filtrar por exchange.
        strategy_id: Filtrar por estratégia.
        repository: Repositório de ordens.
        
    Returns:
        List[OrderResponse]: Lista de ordens.
    """
    # Construir filtros
    filters = {}
    if status:
        filters["status"] = status
    if trading_pair:
        filters["trading_pair"] = trading_pair
    if exchange:
        filters["exchange"] = exchange
    if strategy_id:
        filters["strategy_id"] = strategy_id
    
    # Buscar ordens
    orders, _ = await repository.list_with_pagination(page, size, filters)
    
    # Converter para resposta
    return [
        OrderResponse(
            id=order.id,
            trading_pair=order.trading_pair,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
            fees=order.fees,
            exchange=order.exchange,
            exchange_order_id=order.exchange_order_id,
            strategy_id=order.strategy_id,
            executed_at=order.executed_at.isoformat() if order.executed_at else None,
            error_message=order.error_message,
            child_orders_count=len(order.child_orders),
        )
        for order in orders
    ]


@app.post("/orders/cancel", response_model=OrderResponse, tags=["Execution"])
async def cancel_order(
    request: CancelOrderRequest,
    exchange_client: ExchangeClient = Depends(get_exchange_client),
    repository: OrderRepository = Depends(get_order_repository),
):
    """
    Cancela uma ordem na exchange.
    
    Args:
        request: Dados da ordem a ser cancelada.
        exchange_client: Cliente de exchanges.
        repository: Repositório de ordens.
        
    Returns:
        OrderResponse: Dados da ordem após o cancelamento.
        
    Raises:
        HTTPException: Se houver um erro no cancelamento.
    """
    try:
        # Buscar a ordem no repositório pelo ID na exchange
        order = await repository.get_by_exchange_order_id(request.exchange, request.order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Ordem não encontrada")
        
        # Verificar se a ordem pode ser cancelada
        if order.status not in ["pending", "partial"]:
            raise HTTPException(status_code=400, detail=f"Ordem não pode ser cancelada. Status: {order.status}")
        
        # Cancelar a ordem na exchange
        cancel_result = await exchange_client.cancel_order(
            exchange=request.exchange,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
        )
        
        # Atualizar a ordem no repositório
        order.status = "cancelled"
        order.filled_quantity = cancel_result.get("filled_quantity", order.filled_quantity)
        await repository.save(order)
        
        return OrderResponse(
            id=order.id,
            trading_pair=order.trading_pair,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
            fees=order.fees,
            exchange=order.exchange,
            exchange_order_id=order.exchange_order_id,
            strategy_id=order.strategy_id,
            executed_at=order.executed_at.isoformat() if order.executed_at else None,
            error_message=order.error_message,
            child_orders_count=len(order.child_orders),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao cancelar ordem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao cancelar ordem: {str(e)}")


@app.post("/orders/status", response_model=OrderResponse, tags=["Execution"])
async def check_order_status(
    request: OrderStatusRequest,
    exchange_client: ExchangeClient = Depends(get_exchange_client),
    repository: OrderRepository = Depends(get_order_repository),
):
    """
    Verifica o status de uma ordem na exchange.
    
    Args:
        request: Dados da ordem.
        exchange_client: Cliente de exchanges.
        repository: Repositório de ordens.
        
    Returns:
        OrderResponse: Dados atualizados da ordem.
        
    Raises:
        HTTPException: Se houver um erro na verificação.
    """
    try:
        # Buscar a ordem no repositório pelo ID na exchange
        order = await repository.get_by_exchange_order_id(request.exchange, request.order_id)
        if not order:
            raise HTTPException(status_code=404, detail="Ordem não encontrada")
        
        # Verificar o status na exchange
        status_result = await exchange_client.get_order_status(
            exchange=request.exchange,
            order_id=request.order_id,
            trading_pair=request.trading_pair,
        )
        
        # Atualizar a ordem no repositório
        order.status = status_result.get("status", order.status)
        order.filled_quantity = status_result.get("filled_quantity", order.filled_quantity)
        order.average_price = status_result.get("average_price", order.average_price)
        order.fees = status_result.get("fees", order.fees)
        
        if order.status in ["filled", "partial"] and order.executed_at is None:
            order.executed_at = datetime.utcnow()
        
        await repository.save(order)
        
        return OrderResponse(
            id=order.id,
            trading_pair=order.trading_pair,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
            fees=order.fees,
            exchange=order.exchange,
            exchange_order_id=order.exchange_order_id,
            strategy_id=order.strategy_id,
            executed_at=order.executed_at.isoformat() if order.executed_at else None,
            error_message=order.error_message,
            child_orders_count=len(order.child_orders),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro ao verificar status da ordem: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao verificar status da ordem: {str(e)}")


@app.get("/status", response_model=ServiceStatus, tags=["Service"])
async def get_status(
    exchange_client: ExchangeClient = Depends(get_exchange_client),
):
    """
    Obtém o status do serviço.
    
    Args:
        exchange_client: Cliente de exchanges.
        
    Returns:
        ServiceStatus: Status do serviço.
    """
    return ServiceStatus(
        status="running",
        version="1.0.0",
        exchanges=list(exchange_client.exchanges.keys()),
        algorithms=["twap", "iceberg", "smart_routing", "direct"],
    )


async def initialize():
    """
    Inicializa os componentes do serviço.
    """
    global exchange_client, order_repository, order_routing_service
    global execution_optimization_service, event_publisher, execute_order_use_case
    
    try:
        # Carregar configuração
        config_path = os.environ.get("CONFIG_PATH", "config/execution_service.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Inicializar cliente de exchanges
        exchange_client = ExchangeClient(config["exchanges"])
        await exchange_client.initialize()
        
        # Inicializar repositório de ordens
        order_repository = OrderRepository(config["database"]["url"])
        await order_repository.initialize()
        
        # Inicializar publisher de eventos
        event_publisher = ExecutionEventPublisher(config["kafka"])
        
        # Inicializar serviços
        order_routing_service = OrderRoutingService(exchange_client, config["routing"])
        execution_optimization_service = ExecutionOptimizationService(exchange_client, config["optimization"])
        
        # Inicializar caso de uso
        execute_order_use_case = ExecuteOrderUseCase(
            exchange_client=exchange_client,
            order_repository=order_repository,
            order_routing_service=order_routing_service,
            execution_optimization_service=execution_optimization_service,
            event_publisher=event_publisher,
            config=config["execution"],
        )
        
        logger.info("Serviço de execução inicializado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao inicializar serviço: {str(e)}")
        sys.exit(1)


async def cleanup():
    """
    Limpa recursos antes de encerrar o serviço.
    """
    try:
        if exchange_client:
            await exchange_client.close()
            
        if order_repository:
            await order_repository.close()
            
        if event_publisher:
            event_publisher.close()
            
        logger.info("Recursos liberados com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao liberar recursos: {str(e)}")


def handle_signals():
    """
    Configura handlers para sinais do sistema.
    """
    def signal_handler(sig, frame):
        logger.info(f"Sinal recebido: {sig}. Encerrando serviço...")
        asyncio.create_task(cleanup())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def startup_event():
    """
    Evento executado na inicialização do FastAPI.
    """
    await initialize()


async def shutdown_event():
    """
    Evento executado no encerramento do FastAPI.
    """
    await cleanup()


# Registrar eventos de ciclo de vida
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)


if __name__ == "__main__":
    # Configurar handlers de sinais
    handle_signals()
    
    # Obter porta do ambiente ou usar padrão
    port = int(os.environ.get("PORT", 8000))
    
    # Iniciar servidor Uvicorn
    uvicorn.run("src.execution_service.main:app", host="0.0.0.0", port=port, reload=False)