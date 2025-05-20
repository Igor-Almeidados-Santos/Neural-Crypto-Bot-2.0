"""
Neural Crypto Bot - Analytics Controller

Este módulo implementa o controller para análise e métricas de desempenho.
"""
import logging
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status

from src.api.dependencies import AnalyticsServiceDep, CurrentUser, StrategyRepositoryDep
from src.api.dtos.analytics_dto import (
    AlertConfig,
    AlertInstance,
    AnalyticsRequest,
    DashboardConfig,
    ExposureAnalysis,
    MarketAnalysisRequest,
    MarketAnalysisResponse,
    PortfolioSummary,
    ReportRequest,
    ReportResponse,
    StrategyPerformanceResponse,
    TradeAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/performance", response_model=List[StrategyPerformanceResponse])
async def get_performance_metrics(
    request_data: AnalyticsRequest = Body(...),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> List[StrategyPerformanceResponse]:
    """
    Obtém métricas de desempenho para estratégias específicas ou para todo o portfolio.
    
    Args:
        request_data: Parâmetros para análise de desempenho.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        List[StrategyPerformanceResponse]: Lista de métricas de desempenho por estratégia.
        
    Raises:
        HTTPException: Se ocorrer um erro durante o cálculo das métricas.
    """
    logger.info("Analisando métricas de desempenho")
    
    try:
        # Verifica acesso às estratégias solicitadas
        if request_data.strategy_ids:
            for strategy_id in request_data.strategy_ids:
                strategy = await strategy_repo.get_by_id(strategy_id)
                if not strategy:
                    logger.warning(f"Estratégia não encontrada: {strategy_id}")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Estratégia com ID {strategy_id} não encontrada"
                    )
                
                # Verifica permissão do usuário
                if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
                    logger.warning(f"Acesso não autorizado para analytics: {strategy_id}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Sem permissão para acessar analytics da estratégia {strategy_id}"
                    )
        
        # Realiza a análise de desempenho
        performance_metrics = await analytics_service.get_performance_metrics(
            user_id=current_user.get("username"),
            strategy_ids=request_data.strategy_ids,
            time_range=request_data.time_range,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            metrics=request_data.metrics,
        )
        
        return performance_metrics
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para análise: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao calcular métricas de desempenho: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao calcular métricas de desempenho: {str(e)}"
        )


@router.get("/portfolio/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> PortfolioSummary:
    """
    Obtém um resumo do portfolio atual.
    
    Args:
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        PortfolioSummary: Resumo do portfolio com principais métricas.
        
    Raises:
        HTTPException: Se ocorrer um erro durante o cálculo do resumo.
    """
    logger.info("Obtendo resumo do portfolio")
    
    try:
        # Obtém resumo do portfolio
        portfolio_summary = await analytics_service.get_portfolio_summary(
            user_id=current_user.get("username")
        )
        
        return portfolio_summary
    
    except Exception as e:
        logger.error(f"Erro ao obter resumo do portfolio: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter resumo do portfolio: {str(e)}"
        )


@router.get("/exposure", response_model=ExposureAnalysis)
async def get_exposure_analysis(
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> ExposureAnalysis:
    """
    Obtém análise de exposição atual do portfolio.
    
    Args:
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        ExposureAnalysis: Análise detalhada de exposição.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a análise.
    """
    logger.info("Obtendo análise de exposição")
    
    try:
        # Obtém análise de exposição
        exposure_analysis = await analytics_service.get_exposure_analysis(
            user_id=current_user.get("username")
        )
        
        return exposure_analysis
    
    except Exception as e:
        logger.error(f"Erro ao obter análise de exposição: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter análise de exposição: {str(e)}"
        )


@router.get("/trades", response_model=TradeAnalysisResponse)
async def get_trade_analysis(
    page: int = Query(1, ge=1, description="Número da página"),
    size: int = Query(20, ge=1, le=100, description="Itens por página"),
    strategy_id: Optional[UUID] = Query(None, description="Filtrar por estratégia"),
    trading_pair: Optional[str] = Query(None, description="Filtrar por par de trading"),
    start_date: Optional[str] = Query(None, description="Data de início (ISO format)"),
    end_date: Optional[str] = Query(None, description="Data de fim (ISO format)"),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> TradeAnalysisResponse:
    """
    Obtém análise de trades com paginação e filtros opcionais.
    
    Args:
        page: Número da página atual.
        size: Quantidade de itens por página.
        strategy_id: Filtro opcional por ID de estratégia.
        trading_pair: Filtro opcional por par de trading.
        start_date: Filtro opcional por data de início.
        end_date: Filtro opcional por data de fim.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        TradeAnalysisResponse: Análise de trades paginada.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a análise.
    """
    logger.info(f"Obtendo análise de trades: page={page}, size={size}")
    
    try:
        # Define os filtros baseados nos parâmetros
        filters = {}
        if strategy_id:
            filters["strategy_id"] = strategy_id
        if trading_pair:
            filters["trading_pair"] = trading_pair
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        
        # Obtém análise de trades
        trade_analysis = await analytics_service.get_trade_analysis(
            user_id=current_user.get("username"),
            page=page,
            size=size,
            filters=filters
        )
        
        return trade_analysis
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para análise de trades: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao obter análise de trades: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter análise de trades: {str(e)}"
        )


@router.post("/market", response_model=List[MarketAnalysisResponse])
async def get_market_analysis(
    request_data: MarketAnalysisRequest = Body(...),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> List[MarketAnalysisResponse]:
    """
    Obtém análise de mercado para pares de trading específicos.
    
    Args:
        request_data: Parâmetros para análise de mercado.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        List[MarketAnalysisResponse]: Lista de análises por par de trading.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a análise.
    """
    logger.info(f"Obtendo análise de mercado para {len(request_data.trading_pairs)} pares")
    
    try:
        # Valida pares de trading
        if not request_data.trading_pairs:
            raise ValueError("Pelo menos um par de trading deve ser especificado")
        
        # Obtém análise de mercado
        market_analysis = await analytics_service.get_market_analysis(
            trading_pairs=request_data.trading_pairs,
            time_range=request_data.time_range,
            start_date=request_data.start_date,
            end_date=request_data.end_date,
            indicators=request_data.indicators,
            include_sentiment=request_data.include_sentiment,
            include_correlations=request_data.include_correlations
        )
        
        return market_analysis
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para análise de mercado: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao obter análise de mercado: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter análise de mercado: {str(e)}"
        )


@router.post("/reports", response_model=ReportResponse)
async def generate_report(
    request_data: ReportRequest = Body(...),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
    strategy_repo: StrategyRepositoryDep = Depends(),
) -> ReportResponse:
    """
    Gera um relatório personalizado com métricas e análises.
    
    Args:
        request_data: Parâmetros para geração do relatório.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
        strategy_repo: Repositório de estratégias.
    
    Returns:
        ReportResponse: Informações sobre o relatório gerado.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a geração do relatório.
    """
    logger.info(f"Gerando relatório: {request_data.name}")
    
    try:
        # Verifica acesso às estratégias solicitadas
        if request_data.strategy_ids:
            for strategy_id in request_data.strategy_ids:
                strategy = await strategy_repo.get_by_id(strategy_id)
                if not strategy:
                    logger.warning(f"Estratégia não encontrada: {strategy_id}")
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Estratégia com ID {strategy_id} não encontrada"
                    )
                
                # Verifica permissão do usuário
                if strategy.owner_id != current_user.get("username") and "admin" not in current_user.get("roles", []):
                    logger.warning(f"Acesso não autorizado para relatório: {strategy_id}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Sem permissão para incluir estratégia {strategy_id} no relatório"
                    )
        
        # Gera o relatório
        report = await analytics_service.generate_report(
            user_id=current_user.get("username"),
            request=request_data
        )
        
        return report
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para relatório: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao gerar relatório: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao gerar relatório: {str(e)}"
        )


@router.get("/reports", response_model=List[ReportResponse])
async def list_reports(
    page: int = Query(1, ge=1, description="Número da página"),
    size: int = Query(10, ge=1, le=50, description="Itens por página"),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> List[ReportResponse]:
    """
    Lista relatórios gerados anteriormente.
    
    Args:
        page: Número da página atual.
        size: Quantidade de itens por página.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        List[ReportResponse]: Lista de relatórios.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a listagem.
    """
    logger.info(f"Listando relatórios: page={page}, size={size}")
    
    try:
        # Lista os relatórios
        reports = await analytics_service.list_reports(
            user_id=current_user.get("username"),
            page=page,
            size=size
        )
        
        return reports
    
    except Exception as e:
        logger.error(f"Erro ao listar relatórios: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar relatórios: {str(e)}"
        )


@router.get("/dashboards", response_model=List[DashboardConfig])
async def list_dashboards(
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> List[DashboardConfig]:
    """
    Lista configurações de dashboards disponíveis.
    
    Args:
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        List[DashboardConfig]: Lista de configurações de dashboard.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a listagem.
    """
    logger.info("Listando dashboards")
    
    try:
        # Lista dashboards
        dashboards = await analytics_service.list_dashboards(
            user_id=current_user.get("username")
        )
        
        return dashboards
    
    except Exception as e:
        logger.error(f"Erro ao listar dashboards: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar dashboards: {str(e)}"
        )


@router.post("/dashboards", response_model=DashboardConfig)
async def create_dashboard(
    dashboard: DashboardConfig = Body(...),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> DashboardConfig:
    """
    Cria uma nova configuração de dashboard.
    
    Args:
        dashboard: Configuração do dashboard a ser criado.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        DashboardConfig: Configuração do dashboard criado.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a criação.
    """
    logger.info(f"Criando dashboard: {dashboard.name}")
    
    try:
        # Cria o dashboard
        created_dashboard = await analytics_service.create_dashboard(
            user_id=current_user.get("username"),
            dashboard=dashboard
        )
        
        return created_dashboard
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para dashboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao criar dashboard: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao criar dashboard: {str(e)}"
        )


@router.get("/alerts", response_model=List[AlertConfig])
async def list_alerts(
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> List[AlertConfig]:
    """
    Lista configurações de alertas.
    
    Args:
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        List[AlertConfig]: Lista de configurações de alerta.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a listagem.
    """
    logger.info("Listando alertas")
    
    try:
        # Lista alertas
        alerts = await analytics_service.list_alerts(
            user_id=current_user.get("username")
        )
        
        return alerts
    
    except Exception as e:
        logger.error(f"Erro ao listar alertas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao listar alertas: {str(e)}"
        )


@router.post("/alerts", response_model=AlertConfig)
async def create_alert(
    alert_config: AlertConfig = Body(...),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> AlertConfig:
    """
    Cria uma nova configuração de alerta.
    
    Args:
        alert_config: Configuração do alerta a ser criado.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        AlertConfig: Configuração do alerta criado.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a criação.
    """
    logger.info(f"Criando alerta: {alert_config.name}")
    
    try:
        # Cria o alerta
        created_alert = await analytics_service.create_alert(
            user_id=current_user.get("username"),
            alert_config=alert_config
        )
        
        return created_alert
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para alerta: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao criar alerta: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao criar alerta: {str(e)}"
        )


@router.get("/alerts/history", response_model=List[AlertInstance])
async def get_alert_history(
    page: int = Query(1, ge=1, description="Número da página"),
    size: int = Query(20, ge=1, le=100, description="Itens por página"),
    level: Optional[str] = Query(None, description="Filtrar por nível de alerta"),
    start_date: Optional[str] = Query(None, description="Data de início (ISO format)"),
    end_date: Optional[str] = Query(None, description="Data de fim (ISO format)"),
    acknowledged: Optional[bool] = Query(None, description="Filtrar por status de reconhecimento"),
    current_user: CurrentUser = Depends(),
    analytics_service: AnalyticsServiceDep = Depends(),
) -> List[AlertInstance]:
    """
    Obtém histórico de alertas disparados.
    
    Args:
        page: Número da página atual.
        size: Quantidade de itens por página.
        level: Filtro opcional por nível de alerta.
        start_date: Filtro opcional por data de início.
        end_date: Filtro opcional por data de fim.
        acknowledged: Filtro opcional por status de reconhecimento.
        current_user: Usuário autenticado atual.
        analytics_service: Serviço de analytics.
    
    Returns:
        List[AlertInstance]: Lista de instâncias de alerta.
        
    Raises:
        HTTPException: Se ocorrer um erro durante a obtenção do histórico.
    """
    logger.info(f"Obtendo histórico de alertas: page={page}, size={size}")
    
    try:
        # Define os filtros baseados nos parâmetros
        filters = {}
        if level:
            filters["level"] = level
        if start_date:
            filters["start_date"] = start_date
        if end_date:
            filters["end_date"] = end_date
        if acknowledged is not None:
            filters["acknowledged"] = acknowledged
        
        # Obtém histórico de alertas
        alert_history = await analytics_service.get_alert_history(
            user_id=current_user.get("username"),
            page=page,
            size=size,
            filters=filters
        )
        
        return alert_history
    
    except ValueError as e:
        logger.warning(f"Parâmetros inválidos para histórico de alertas: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.error(f"Erro ao obter histórico de alertas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter histórico de alertas: {str(e)}"
        )