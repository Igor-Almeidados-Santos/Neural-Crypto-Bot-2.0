"""
Neural Crypto Bot - Strategy Controller V2

Este módulo implementa a versão 2 do controller para gerenciamento de estratégias de trading.
Fornece funcionalidades aprimoradas e novos endpoints em relação à versão 1.
"""
import logging
from typing import Dict, List, Optional, Union, Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query, status
from fastapi.responses import JSONResponse

from api.dependencies import (
    BacktestStrategyUCDep, 
    CurrentUser, 
    ExecuteOrderUCDep, 
    ExecuteStrategyUCDep, 
    StrategyRepositoryDep
)
from api.dtos.strategy_dto import (
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
from common.infrastructure.cache.cache_manager import cached, get_cache
from common.utils.validation import validate_trading_pair

logger = logging.getLogger(__name__)

# Criação do router com configurações de documentação
router = APIRouter(
    responses={
        status.HTTP_401_UNAUTHORIZED: {"description": "Não autenticado"},
        status.HTTP_403_FORBIDDEN: {"description": "Sem permissão para realizar esta ação"},
        status.HTTP_404_NOT_FOUND: {"description": "Recurso não encontrado"},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {"description": "Erro de validação nos dados enviados"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Erro interno no servidor"},
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Muitas requisições. Tente novamente mais tarde."}
    }
)


@router.post("", response_model=StrategyResponse, status_code=status.HTTP_201_CREATED)
async def create_strategy(
    strategy_data: StrategyCreate = Body(..., description="Dados da estratégia a ser criada", examples=[{
        "name": "Mean Reversion BTC",
        "description": "Estratégia de reversão à média para Bitcoin",
        "type": "mean_reversion",
        "trading_pairs": ["BTC/USDT"],
        "timeframe": "1h",
        "parameters": [
            {"name": "lookback_period", "value": 24},
            {"name": "std_dev_threshold", "value": 2.0}
        ],
        "max_position_size": 0.1,
        "max_drawdown_percent": 5.0,
        "is_active": False
    }]),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Cria uma nova estratégia de trading.
    
    Esta operação permite criar uma nova estratégia de trading com parâmetros 
    personalizados, mas não a ativa automaticamente.
    
    Args:
        strategy_data: Dados da estratégia a ser criada.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        Dados da estratégia criada com ID único.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a criação.
    """
    logger.info(f"[V2] Criando estratégia: {strategy_data.name}")
    
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
@cached(ttl=60, tags=["strategies", "list"])
async def list_strategies(
    page: int = Query(1, ge=1, description="Número da página"),
    size: int = Query(10, ge=1, le=100, description="Itens por página"),
    type: Optional[str] = Query(None, description="Filtrar por tipo de estratégia"),
    is_active: Optional[bool] = Query(None, description="Filtrar por status ativo/inativo"),
    sort_by: str = Query("created_at", description="Campo para ordenação"),
    sort_dir: str = Query("desc", description="Direção da ordenação (asc, desc)"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyListResponse:
    """
    Lista estratégias de trading com paginação e filtros.
    
    Esta operação retorna uma lista paginada de estratégias de trading
    do usuário atual, com opções de filtragem e ordenação.
    
    Args:
        page: Número da página atual.
        size: Quantidade de itens por página.
        type: Filtro opcional por tipo de estratégia.
        is_active: Filtro opcional por status de ativação.
        sort_by: Campo para ordenação dos resultados.
        sort_dir: Direção da ordenação (asc, desc).
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        Lista de estratégias com metadados de paginação.
    """
    logger.info(f"[V2] Listando estratégias: page={page}, size={size}, sort_by={sort_by}, sort_dir={sort_dir}")
    
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
        user_id=current_user.get("username"),
        sort_by=sort_by,
        sort_direction=sort_dir
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
@cached(ttl=30, tags=["strategies", "detail"])
async def get_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    include_stats: bool = Query(True, description="Incluir estatísticas da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Obtém detalhes de uma estratégia específica.
    
    Esta operação retorna informações completas sobre uma estratégia,
    incluindo seus parâmetros e estatísticas de desempenho se solicitado.
    
    Args:
        strategy_id: ID da estratégia.
        include_stats: Incluir estatísticas de desempenho.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        Dados detalhados da estratégia.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou o usuário não tiver permissão.
    """
    logger.info(f"[V2] Buscando estratégia: {strategy_id}, include_stats={include_stats}")
    
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
    
    # Converte para DTO
    strategy_dict = strategy.to_dict()
    
    # Adiciona estatísticas se solicitado
    if include_stats:
        stats = await strategy_repo.get_strategy_stats(strategy_id)
        strategy_dict["stats"] = stats
    
    return StrategyResponse.model_validate(strategy_dict)


@router.put("/{strategy_id}", response_model=StrategyResponse)
async def update_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    update_data: StrategyUpdate = Body(..., description="Dados atualizados da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Atualiza uma estratégia existente.
    
    Esta operação permite modificar os parâmetros e configurações de uma
    estratégia existente.
    
    Args:
        strategy_id: ID da estratégia a ser atualizada.
        update_data: Dados atualizados da estratégia.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        Dados atualizados da estratégia.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou o usuário não tiver permissão.
    """
    logger.info(f"[V2] Atualizando estratégia: {strategy_id}")
    
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
        
        # Invalida o cache relacionado à estratégia
        cache = get_cache()
        await cache.invalidate_by_tag(f"strategy:{strategy_id}")
        await cache.invalidate_by_tag("strategies:list")
        
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
    force: bool = Query(False, description="Forçar exclusão mesmo se a estratégia estiver ativa"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> None:
    """
    Remove uma estratégia existente.
    
    Esta operação remove permanentemente uma estratégia do sistema.
    Estratégias ativas não podem ser removidas a menos que o parâmetro 'force' seja True.
    
    Args:
        strategy_id: ID da estratégia a ser removida.
        force: Se True, permite remover estratégias ativas.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Raises:
        HTTPException: Se a estratégia não for encontrada, o usuário não tiver permissão
                      ou a estratégia estiver ativa e 'force' for False.
    """
    logger.info(f"[V2] Removendo estratégia: {strategy_id}, force={force}")
    
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
    if strategy.is_active and not force:
        logger.warning(f"Tentativa de remover estratégia ativa: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não é possível excluir uma estratégia ativa. Desative-a primeiro ou use o parâmetro 'force=true'."
        )
    
    # Remove a estratégia
    await strategy_repo.delete(strategy_id)
    
    # Invalida o cache relacionado à estratégia
    cache = get_cache()
    await cache.invalidate_by_tag(f"strategy:{strategy_id}")
    await cache.invalidate_by_tag("strategies:list")


@router.post("/{strategy_id}/activate", response_model=StrategyResponse)
async def activate_strategy(
    strategy_id: UUID = Path(..., description="ID da estratégia"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> StrategyResponse:
    """
    Ativa uma estratégia para começar a executar trades.
    
    Esta operação ativa uma estratégia previamente criada, permitindo
    que ela comece a gerar sinais e executar trades automaticamente.
    
    Args:
        strategy_id: ID da estratégia a ser ativada.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        Dados atualizados da estratégia.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a ativação.
    """
    logger.info(f"[V2] Ativando estratégia: {strategy_id}")
    
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
        
        # Invalida o cache relacionado à estratégia
        cache = get_cache()
        await cache.invalidate_by_tag(f"strategy:{strategy_id}")
        await cache.invalidate_by_tag("strategies:list")
        
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
    
    Esta operação desativa uma estratégia previamente ativada,
    fazendo com que ela pare de gerar sinais e executar trades.
    
    Args:
        strategy_id: ID da estratégia a ser desativada.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        Dados atualizados da estratégia.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a desativação.
    """
    logger.info(f"[V2] Desativando estratégia: {strategy_id}")
    
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
        
        # Invalida o cache relacionado à estratégia
        cache = get_cache()
        await cache.invalidate_by_tag(f"strategy:{strategy_id}")
        await cache.invalidate_by_tag("strategies:list")
        
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
    backtest_params: BacktestRequest = Body(..., description="Parâmetros do backtest"),
    background_tasks: BackgroundTasks = Depends(),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
    backtest_uc: BacktestStrategyUCDep = Depends(),
) -> BacktestResult:
    """
    Executa backtesting em uma estratégia.
    
    Esta operação simula o desempenho de uma estratégia em dados históricos,
    permitindo avaliar sua eficácia antes de usá-la com dinheiro real.
    Os backtests extensos podem ser executados em segundo plano.
    
    Args:
        strategy_id: ID da estratégia para backtest.
        backtest_params: Parâmetros de configuração do backtest.
        background_tasks: Tarefas em segundo plano.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
        backtest_uc: Caso de uso para execução de backtest.
    
    Returns:
        Resultados do backtest.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou ocorrer um erro durante o backtest.
    """
    logger.info(f"[V2] Executando backtest para estratégia: {strategy_id}")
    
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
        
        # Se o backtest incluir detalhes de trades e for muito grande,
        # processa o resultado detalhado em segundo plano e retorna um resultado resumido
        if backtest_params.include_detailed_trades and len(result.equity_curve) > 1000:
            background_tasks.add_task(
                process_detailed_backtest_results,
                result=result,
                strategy_id=strategy_id,
                user_id=current_user.get("username")
            )
            
            # Remove detalhes extensos para a resposta imediata
            result.detailed_trades = None
            result.equity_curve = result.equity_curve[:1000]  # Limita para os primeiros 1000 pontos
            
            logger.info(f"Detalhes completos do backtest sendo processados em segundo plano para: {strategy_id}")
        
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
    request_data: SignalRequest = Body(..., description="Dados para geração de sinais"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
    execute_strategy_uc: ExecuteStrategyUCDep = Depends(),
) -> List[SignalResponse]:
    """
    Gera sinais de trading para uma estratégia.
    
    Esta operação executa a estratégia para gerar sinais de trading
    com base nos dados de mercado atuais, sem executar ordens reais
    a menos que solicitado.
    
    Args:
        strategy_id: ID da estratégia.
        request_data: Dados da requisição para geração de sinais.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
        execute_strategy_uc: Caso de uso para execução de estratégia.
    
    Returns:
        Lista de sinais gerados.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou ocorrer um erro durante a geração.
    """
    logger.info(f"[V2] Gerando sinais para estratégia: {strategy_id}")
    
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


@router.post("/execute", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def execute_strategy(
    strategy_id: UUID = Query(..., description="ID da estratégia a executar"),
    trading_pairs: Optional[List[str]] = Query(None, description="Pares de trading específicos"),
    max_orders: int = Query(1, ge=1, le=10, description="Número máximo de ordens a gerar"),
    dry_run: bool = Query(False, description="Não executar ordens reais, apenas simular"),
    current_user: CurrentUser = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
    execute_strategy_uc: ExecuteStrategyUCDep = Depends(),
) -> OrderResponse:
    """
    Executa uma estratégia manualmente para gerar e executar ordens imediatamente.
    
    Esta é uma nova operação na V2 que permite executar uma estratégia sob demanda,
    gerando e potencialmente executando ordens com base nas condições atuais do mercado.
    
    Args:
        strategy_id: ID da estratégia a executar.
        trading_pairs: Lista opcional de pares específicos.
        max_orders: Número máximo de ordens a gerar.
        dry_run: Se True, apenas simula a execução sem enviar ordens reais.
        current_user: Usuário autenticado atual.
        strategy_repo: Repositório de estratégias.
        execute_strategy_uc: Caso de uso para execução de estratégia.
    
    Returns:
        Detalhes da ordem executada ou simulada.
        
    Raises:
        HTTPException: Se a estratégia não for encontrada ou ocorrer um erro durante a execução.
    """
    logger.info(f"[V2] Executando estratégia manualmente: {strategy_id}, dry_run={dry_run}")
    
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
        logger.warning(f"Acesso não autorizado para executar estratégia: {strategy_id}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Sem permissão para executar esta estratégia"
        )
    
    try:
        # Define os pares para execução da estratégia
        target_pairs = trading_pairs or strategy.trading_pairs
        
        # Executa a estratégia
        execution_result = await execute_strategy_uc.execute_manual(
            strategy=strategy,
            trading_pairs=target_pairs,
            max_orders=max_orders,
            dry_run=dry_run
        )
        
        return execution_result
    
    except Exception as e:
        logger.error(f"Erro ao executar estratégia: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao executar estratégia: {str(e)}"
        )


async def process_detailed_backtest_results(
    result: BacktestResult,
    strategy_id: UUID,
    user_id: str
) -> None:
    """
    Processa resultados detalhados de backtest em segundo plano.
    
    Esta função é executada como tarefa em segundo plano para processar
    e armazenar resultados extensivos de backtest.
    
    Args:
        result: Resultado do backtest a ser processado.
        strategy_id: ID da estratégia testada.
        user_id: ID do usuário que solicitou o backtest.
    """
    try:
        logger.info(f"Processando resultados detalhados do backtest para estratégia: {strategy_id}")
        
        # Aqui você implementaria a lógica para:
        # 1. Salvar resultados detalhados no banco de dados
        # 2. Gerar gráficos e visualizações
        # 3. Enviar notificação para o usuário quando concluído
        # 4. Armazenar métricas para análises futuras
        
        # Simulação de processamento
        await asyncio.sleep(2)  # Simula processamento
        
        logger.info(f"Processamento de backtest concluído para estratégia: {strategy_id}")
        
    except Exception as e:
        logger.error(f"Erro ao processar resultados detalhados do backtest: {str(e)}", exc_info=True)