"""
Neural Crypto Bot - Strategy Controller

Este módulo implementa o controller para gerenciamento de estratégias de trading.
"""
import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status

from src.api.dependencies import (
    BacktestStrategyUCDep, 
    CurrentUser, 
    ExecuteOrderUCDep, 
    ExecuteStrategyUCDep, 
    StrategyRepositoryDep
)
from src.api.dtos.strategy_dto import (
    BacktestRequest, 
    BacktestResult, 
    OrderRequest, 
    OrderResponse, 
    SignalRequest, 
    SignalResponse, 
    StrategyCreate, 
    StrategyListResponse, 
    StrategyResponse, 
    StrategyUpdate
)
from src.common.utils.validation import validate_trading_pair

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    strategy_data: StrategyCreate = Body(...),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Cria uma nova estratégia de trading.
    
    Args:
        strategy_data: Dados da estratégia a ser criada.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        StrategyResponse: Dados da estratégia criada.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a criação.
    """
    logger.info(f"Criando estratégia: {strategy_data.name}")
    
    # Valida todos os pares de trading
    for pair in strategy_data.trading_pairs:
        if not validate_trading_pair(pair):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Par de trading inválido: {pair}. Formato esperado: 'BTC/USDT'"
            )
    
    try:
        # Converte DTO para entidade e salva
        strategy_entity = await strategy_repo.create_from_dto(strategy_data, current_user["username"])
        
        # Converte entidade de volta para DTO de resposta
        return StrategyResponse.model_validate(strategy_entity.to_dict())
    
    except Exception as e:
        logger.error(f"Erro ao criar estratégia: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao criar estratégia: {str(e)}"
        )


@router.get("", response_model=StrategyListResponse)
async def list_strategies(
    page: int = Query(1, ge=1, description="Número da página"),
    size: int = Query(10, ge=1, le=100, description="Itens por página"),
    type: Optional[str] = Query(None, description="Filtrar por tipo de estratégia"),
    is_active: Optional[bool] = Query(None, description="Filtrar por status ativo/inativo"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyListResponse:
    """
    Lista estratégias de trading com paginação e filtros opcionais.
    
    Args:
        page: Número da página atual.
        size: Quantidade de itens por página.
        type: Filtro opcional por tipo de estratégia.
        is_active: Filtro opcional por status de ativação.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        StrategyListResponse: Lista de estratégias com metadados de paginação.
    """
    logger.info(f"Listando estratégias: page={page}, size={size}")
    
    # Define os filtros baseados nos parâmetros
    filters = {}
    if type:
        filters["type"] = type
    if is_active is not None:
        filters["is_active"] = is_active
    
    # Obtém as estratégias do repositório
    strategies, total = await strategy_repo.list_with_pagination(
        page=page,
        size=size,
        filters=filters,
        user_id=current_user.get("username")
    )
    
    # Converte entidades para DTOs de resposta
    strategy_responses = [StrategyResponse.model_validate(s.to_dict()) for s in strategies]
    
    return StrategyListResponse(
        items=strategy_responses,
        total=total,
        page=page,
        size=size
    )


@router.get("/{strategy_id}", response_model=StrategyResponse)
async def get_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Obtém detalhes de uma estratégia específica.
    
    Args:
        strategy_id: ID da estratégia.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        StrategyResponse: Dados da estratégia.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada.
    """
    logger.info(f"Buscando estratégia: {strategy_id}")
    
    strategy = await strategy_repo.get_by_id(strategy_id)
    
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado à estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para acessar esta estratégia"
        )
    
    return StrategyResponse.model_validate(strategy.to_dict())


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    update_data: StrategyUpdate = Body(...),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Atualiza uma estratégia existente.
    
    Args:
        strategy_id: ID da estratégia a ser atualizada.
        update_data: Dados atualizados da estratégia.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        StrategyResponse: Dados atualizados da estratégia.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou o usuário não tiver permissão.
    """
    logger.info(f"Atualizando estratégia: {strategy_id}")
    
    # Valida pares de trading, se fornecidos
    if update_data.trading_pairs:
        for pair in update_data.trading_pairs:
            if not validate_trading_pair(pair):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Par de trading inválido: {pair}. Formato esperado: 'BTC/USDT'"
                )
    
    # Verifica se a estratégia existe
    strategy = await strategy_repo.get_by_id(strategy_id)
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado para atualizar estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para atualizar esta estratégia"
        )
    
    try:
        # Atualiza a estratégia
        updated_strategy = await strategy_repo.update(strategy_id, update_data)
        return StrategyResponse.model_validate(updated_strategy.to_dict())
    
    except Exception as e:
        logger.error(f"Erro ao atualizar estratégia: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao atualizar estratégia: {str(e)}"
        )


@router.delete("/{strategy_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> None:
    """
    Remove uma estratégia existente.
    
    Args:
        strategy_id: ID da estratégia a ser removida.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Raises:
        HTTPException: Se a estratégia não for encontrada ou o usuário não tiver permissão.
    """
    logger.info(f"Removendo estratégia: {strategy_id}")
    
    # Verifica se a estratégia existe
    strategy = await strategy_repo.get_by_id(strategy_id)
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado para remover estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para remover esta estratégia"
        )
    
    # Verifica se a estratégia está ativa
    if strategy.is_active:
        logger.warning(f"Tentativa de remover estratégia ativa: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não é possível excluir uma estratégia ativa. Desative-a primeiro."
        )
    
    # Remove a estratégia
    await strategy_repo.delete(strategy_id)


@router.post("/{strategy_id}/activate", response_model=StrategyResponse)
async def activate_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Ativa uma estratégia para começar a executar trades.
    
    Args:
        strategy_id: ID da estratégia a ser ativada.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        StrategyResponse: Dados atualizados da estratégia.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a ativação.
    """
    logger.info(f"Ativando estratégia: {strategy_id}")
    
    # Verifica se a estratégia existe
    strategy = await strategy_repo.get_by_id(strategy_id)
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado para ativar estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para ativar esta estratégia"
        )
    
    try:
        # Ativa a estratégia
        strategy.is_active = True
        strategy.status = "active"
        updated_strategy = await strategy_repo.save(strategy)
        
        # Notifica o serviço de execução sobre a ativação da estratégia
        # Aqui você poderia integrar com seu sistema de mensageria para notificar outros serviços
        # Exemplo: await message_broker.publish("strategy.activated", strategy_id)
        
        return StrategyResponse.model_validate(updated_strategy.to_dict())
    
    except Exception as e:
        logger.error(f"Erro ao ativar estratégia: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao ativar estratégia: {str(e)}"
        )


@router.post("/{strategy_id}/deactivate", response_model=StrategyResponse)
async def deactivate_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Desativa uma estratégia para parar de executar trades.
    
    Args:
        strategy_id: ID da estratégia a ser desativada.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        StrategyResponse: Dados atualizados da estratégia.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a desativação.
    """
    logger.info(f"Desativando estratégia: {strategy_id}")
    
    # Verifica se a estratégia existe
    strategy = await strategy_repo.get_by_id(strategy_id)
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado para desativar estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para desativar esta estratégia"
        )
    
    try:
        # Desativa a estratégia
        strategy.is_active = False
        strategy.status = "paused"
        updated_strategy = await strategy_repo.save(strategy)
        
        # Notifica o serviço de execução sobre a desativação da estratégia
        # Exemplo: await message_broker.publish("strategy.deactivated", strategy_id)
        
        return StrategyResponse.model_validate(updated_strategy.to_dict())
    
    except Exception as e:
        logger.error(f"Erro ao desativar estratégia: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao desativar estratégia: {str(e)}"
        )


@router.post("/{strategy_id}/backtest", response_model=BacktestResult)
async def run_backtest(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    backtest_params: BacktestRequest = Body(...),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
    backtest_uc: BacktestStrategyUCDep = Depends(),
) -> BacktestResult:
    """
    Executa backtesting em uma estratégia.
    
    Args:
        strategy_id: ID da estratégia para backtest.
        backtest_params: Parâmetros de configuração do backtest.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
        backtest_uc: Caso de uso para execução de backtest.
    
    Returns:
        BacktestResult: Resultados do backtest.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou ocorrer um erro durante o backtest.
    """
    logger.info(f"Executando backtest para estratégia: {strategy_id}")
    
    # Verifica se a estratégia existe
    strategy = await strategy_repo.get_by_id(strategy_id)
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado para backtest de estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para executar backtest nesta estratégia"
        )
    
    try:
        # Executa o backtest
        result = await backtest_uc.execute(strategy, backtest_params)
        return result
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para backtest: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao executar backtest: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar backtest: {str(e)}"
        )


@router.post("/{strategy_id}/signals", response_model=List[SignalResponse])
async def generate_signals(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    request_data: SignalRequest = Body(...),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
    execute_strategy_uc: ExecuteStrategyUCDep = Depends(),
) -> List[SignalResponse]:
    """
    Gera sinais de trading para uma estratégia.
    
    Args:
        strategy_id: ID da estratégia.
        request_data: Dados da requisição para geração de sinais.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
        execute_strategy_uc: Caso de uso para execução de estratégia.
    
    Returns:
        List[SignalResponse]: Lista de sinais gerados.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou ocorrer um erro durante a geração.
    """
    logger.info(f"Gerando sinais para estratégia: {strategy_id}")
    
    # Verifica se a estratégia existe
    strategy = await strategy_repo.get_by_id(strategy_id)
    if not strategy:
        logger.warning(f"Estratégia não encontrada: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Estratégia com ID {strategy_id} não encontrada"
        )
    
    # Verifica permissão do usuário
    if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
        logger.warning(f"Acesso não autorizado para gerar sinais: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para gerar sinais com esta estratégia"
        )
    
    try:
        # Define os pares para geração de sinais
        trading_pairs = request_data.trading_pairs or strategy.trading_pairs
        
        # Gera os sinais
        signals = await execute_strategy_uc.generate_signals(
            strategy=strategy,
            trading_pairs=trading_pairs,
            generate_orders=request_data.generate_orders
        )
        
        return signals
    
    except Exception as e:
        logger.error(f"Erro ao gerar sinais: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar sinais: {str(e)}"
        )


@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order_data: OrderRequest = Body(...),
    current_user: CurrentUser = Depends(),
    execute_order_uc: ExecuteOrderUCDep = Depends(),
) -> OrderResponse:
    """
    Cria e executa uma ordem de trading.
    
    Args:
        order_data: Dados da ordem a ser criada.
        current_user: Usuário autenticado atual.
        execute_order_uc: Caso de uso para execução de ordens.
    
    Returns:
        OrderResponse: Dados da ordem criada e seu status.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a criação ou execução da ordem.
    """
    logger.info(f"Criando ordem: {order_data.trading_pair}, {order_data.side}, {order_data.type}")
    
    # Valida o par de trading
    if not validate_trading_pair(order_data.trading_pair):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Par de trading inválido: {order_data.trading_pair}. Formato esperado: 'BTC/USDT'"
        )
    
    # Validações específicas por tipo de ordem
    if order_data.type != "market" and not order_data.price:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preço é obrigatório para ordens do tipo {order_data.type}"
        )
    
    try:
        # Cria e executa a ordem
        order_result = await execute_order_uc.execute(
            order_data=order_data,
            user_id=current_user.get("username")
        )
        
        return order_result
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para ordem: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao criar/executar ordem: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar ordem: {str(e)}"
        )