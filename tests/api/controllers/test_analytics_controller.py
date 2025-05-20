"""
Testes para o Analytics Controller.

Este módulo contém testes unitários para o controller de analytics.
"""
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from src.api.controllers import analytics_controller
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
    TimeRange,
    TradeAnalysisResponse,
)
from src.analytics.application.services.analytics_service import AnalyticsService
from src.strategy_engine.domain.entities.strategy import Strategy


@pytest.fixture
def app():
    """Cria uma instância da aplicação FastAPI para testes."""
    app = FastAPI()
    app.include_router(analytics_controller.router)
    return app


@pytest.fixture
def client(app):
    """Cria um cliente de teste para a aplicação FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_current_user():
    """Mock para simular um usuário autenticado."""
    return {"username": "test_user", "email": "test@example.com", "roles": ["trader"]}


@pytest.fixture
def mock_admin_user():
    """Mock para simular um usuário administrador."""
    return {"username": "admin_user", "email": "admin@example.com", "roles": ["admin", "trader"]}


@pytest.fixture
def mock_strategy_repo():
    """Mock para o repositório de estratégias."""
    repo = AsyncMock()
    
    # Setup do mock para get_by_id
    strategy = MagicMock(spec=Strategy)
    strategy.id = str(uuid.uuid4())
    strategy.name = "Test Strategy"
    strategy.owner_id = "test_user"
    
    repo.get_by_id.return_value = strategy
    
    return repo


@pytest.fixture
def mock_analytics_service():
    """Mock para o serviço de analytics."""
    service = AsyncMock(spec=AnalyticsService)
    
    # Setup do mock para get_performance_metrics
    performance = MagicMock(spec=StrategyPerformanceResponse)
    performance.strategy_id = str(uuid.uuid4())
    performance.strategy_name = "Test Strategy"
    performance.time_range = TimeRange.MONTH
    performance.start_date = datetime(2025, 4, 1)
    performance.end_date = datetime(2025, 5, 1)
    performance.initial_capital = 10000.0
    performance.final_capital = 12000.0
    
    service.get_performance_metrics.return_value = [performance]
    
    # Setup do mock para get_portfolio_summary
    portfolio = MagicMock(spec=PortfolioSummary)
    portfolio.total_capital = 50000.0
    portfolio.allocated_capital = 25000.0
    portfolio.available_capital = 25000.0
    portfolio.total_profit_loss = 5000.0
    portfolio.total_profit_loss_percent = 10.0
    
    service.get_portfolio_summary.return_value = portfolio
    
    # Setup do mock para get_exposure_analysis
    exposure = MagicMock(spec=ExposureAnalysis)
    exposure.total_exposure = 25000.0
    exposure.long_exposure = 20000.0
    exposure.short_exposure = 5000.0
    exposure.net_exposure = 15000.0
    exposure.gross_exposure = 25000.0
    
    service.get_exposure_analysis.return_value = exposure
    
    # Setup do mock para get_trade_analysis
    trade_analysis = MagicMock(spec=TradeAnalysisResponse)
    trade_analysis.trades = []
    trade_analysis.total = 0
    trade_analysis.page = 1
    trade_analysis.size = 20
    
    service.get_trade_analysis.return_value = trade_analysis
    
    # Setup do mock para get_market_analysis
    market_analysis = MagicMock(spec=MarketAnalysisResponse)
    market_analysis.trading_pair = "BTC/USDT"
    market_analysis.time_range = TimeRange.MONTH
    market_analysis.start_date = datetime(2025, 4, 1)
    market_analysis.end_date = datetime(2025, 5, 1)
    
    service.get_market_analysis.return_value = [market_analysis]
    
    # Setup do mock para generate_report
    report = MagicMock(spec=ReportResponse)
    report.id = str(uuid.uuid4())
    report.name = "Monthly Performance Report"
    report.created_at = datetime.utcnow()
    report.url = "https://example.com/reports/123456.pdf"
    
    service.generate_report.return_value = report
    
    # Setup do mock para list_reports
    service.list_reports.return_value = [report]
    
    # Setup do mock para list_dashboards
    dashboard = MagicMock(spec=DashboardConfig)
    dashboard.id = str(uuid.uuid4())
    dashboard.name = "Trading Dashboard"
    dashboard.is_default = True
    
    service.list_dashboards.return_value = [dashboard]
    
    # Setup do mock para create_dashboard
    service.create_dashboard.return_value = dashboard
    
    # Setup do mock para list_alerts
    alert = MagicMock(spec=AlertConfig)
    alert.id = str(uuid.uuid4())
    alert.name = "Drawdown Alert"
    alert.level = "warning"
    
    service.list_alerts.return_value = [alert]
    
    # Setup do mock para create_alert
    service.create_alert.return_value = alert
    
    # Setup do mock para get_alert_history
    alert_instance = MagicMock(spec=AlertInstance)
    alert_instance.id = str(uuid.uuid4())
    alert_instance.config_id = str(uuid.uuid4())
    alert_instance.triggered_at = datetime.utcnow()
    alert_instance.level = "warning"
    
    service.get_alert_history.return_value = [alert_instance]
    
    return service


@pytest.mark.asyncio
async def test_get_performance_metrics(client, mock_current_user, mock_analytics_service, mock_strategy_repo):
    """Testa a obtenção de métricas de desempenho."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service), \
         patch("src.api.controllers.analytics_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Prepara os dados da requisição
        request_data = {
            "strategy_ids": [str(uuid.uuid4())],
            "time_range": "month",
            "metrics": ["total_return", "sharpe_ratio", "max_drawdown"],
            "include_trades": False
        }
        
        # Faz a requisição
        response = client.post("/performance", json=request_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_analytics_service.get_performance_metrics.called
        
        # Verifica o tipo da resposta
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos principais da resposta
        performance = result[0]
        assert "strategy_id" in performance
        assert "strategy_name" in performance
        assert "time_range" in performance
        assert "start_date" in performance
        assert "end_date" in performance
        assert "initial_capital" in performance
        assert "final_capital" in performance


@pytest.mark.asyncio
async def test_get_performance_metrics_unauthorized_strategy(client, mock_current_user, mock_analytics_service, mock_strategy_repo):
    """Testa a obtenção de métricas de desempenho para uma estratégia não autorizada."""
    # Modifica o proprietário da estratégia para ser diferente do usuário atual
    strategy = mock_strategy_repo.get_by_id.return_value
    strategy.owner_id = "another_user"
    
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service), \
         patch("src.api.controllers.analytics_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Prepara os dados da requisição
        request_data = {
            "strategy_ids": [str(uuid.uuid4())],
            "time_range": "month",
            "metrics": ["total_return", "sharpe_ratio", "max_drawdown"],
            "include_trades": False
        }
        
        # Faz a requisição
        response = client.post("/performance", json=request_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert mock_strategy_repo.get_by_id.called
        assert not mock_analytics_service.get_performance_metrics.called


@pytest.mark.asyncio
async def test_get_portfolio_summary(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção do resumo do portfolio."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/portfolio/summary")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_portfolio_summary.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "total_capital" in result
        assert "allocated_capital" in result
        assert "available_capital" in result
        assert "total_profit_loss" in result
        assert "total_profit_loss_percent" in result


@pytest.mark.asyncio
async def test_get_exposure_analysis(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção da análise de exposição."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/exposure")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_exposure_analysis.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "total_exposure" in result
        assert "long_exposure" in result
        assert "short_exposure" in result
        assert "net_exposure" in result
        assert "gross_exposure" in result


@pytest.mark.asyncio
async def test_get_trade_analysis(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção da análise de trades."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/trades")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_trade_analysis.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "trades" in result
        assert "total" in result
        assert "page" in result
        assert "size" in result


@pytest.mark.asyncio
async def test_get_trade_analysis_with_filters(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção da análise de trades com filtros."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição com parâmetros de filtro
        strategy_id = str(uuid.uuid4())
        response = client.get(f"/trades?page=2&size=10&strategy_id={strategy_id}&trading_pair=BTC/USDT")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_trade_analysis.called
        
        # Verifica se os filtros foram passados corretamente
        _, kwargs = mock_analytics_service.get_trade_analysis.call_args
        assert kwargs["page"] == 2
        assert kwargs["size"] == 10
        assert "filters" in kwargs
        assert kwargs["filters"]["strategy_id"] == strategy_id
        assert kwargs["filters"]["trading_pair"] == "BTC/USDT"


@pytest.mark.asyncio
async def test_get_market_analysis(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção da análise de mercado."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Prepara os dados da requisição
        request_data = {
            "trading_pairs": ["BTC/USDT", "ETH/USDT"],
            "time_range": "month",
            "indicators": ["rsi", "macd", "bollinger_bands"],
            "include_sentiment": True,
            "include_correlations": True
        }
        
        # Faz a requisição
        response = client.post("/market", json=request_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_market_analysis.called
        
        # Verifica o tipo da resposta
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos principais da resposta
        market = result[0]
        assert "trading_pair" in market
        assert "time_range" in market
        assert "start_date" in market
        assert "end_date" in market


@pytest.mark.asyncio
async def test_generate_report(client, mock_current_user, mock_analytics_service, mock_strategy_repo):
    """Testa a geração de um relatório."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service), \
         patch("src.api.controllers.analytics_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Prepara os dados da requisição
        request_data = {
            "name": "Monthly Performance Report",
            "strategy_ids": [str(uuid.uuid4())],
            "time_range": "month",
            "sections": ["performance", "risk", "trades"],
            "format": "pdf",
            "include_charts": True,
            "include_trade_details": False
        }
        
        # Faz a requisição
        response = client.post("/reports", json=request_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_analytics_service.generate_report.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "id" in result
        assert "name" in result
        assert "created_at" in result
        assert "url" in result


@pytest.mark.asyncio
async def test_list_reports(client, mock_current_user, mock_analytics_service):
    """Testa a listagem de relatórios."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/reports")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.list_reports.called
        
        # Verifica o tipo da resposta
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos principais da resposta
        report = result[0]
        assert "id" in report
        assert "name" in report
        assert "created_at" in report
        assert "url" in report


@pytest.mark.asyncio
async def test_list_dashboards(client, mock_current_user, mock_analytics_service):
    """Testa a listagem de dashboards."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/dashboards")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.list_dashboards.called
        
        # Verifica o tipo da resposta
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos principais da resposta
        dashboard = result[0]
        assert "id" in dashboard
        assert "name" in dashboard
        assert "is_default" in dashboard


@pytest.mark.asyncio
async def test_create_dashboard(client, mock_current_user, mock_analytics_service):
    """Testa a criação de um dashboard."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Prepara os dados da requisição
        dashboard_data = {
            "id": str(uuid.uuid4()),
            "name": "Trading Dashboard",
            "description": "Main trading dashboard for daily monitoring",
            "layout": "grid",
            "is_default": True,
            "items": [
                {
                    "id": str(uuid.uuid4()),
                    "title": "Portfolio Value",
                    "type": "line_chart",
                    "data_source": "portfolio_history",
                    "refresh_interval": 300,
                    "size": {"width": 6, "height": 4},
                    "position": {"x": 0, "y": 0},
                    "config": {},
                    "last_updated": datetime.utcnow().isoformat()
                }
            ],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "created_by": "test_user"
        }
        
        # Faz a requisição
        response = client.post("/dashboards", json=dashboard_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.create_dashboard.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "id" in result
        assert "name" in result
        assert "is_default" in result


@pytest.mark.asyncio
async def test_list_alerts(client, mock_current_user, mock_analytics_service):
    """Testa a listagem de alertas."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/alerts")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.list_alerts.called
        
        # Verifica o tipo da resposta
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos principais da resposta
        alert = result[0]
        assert "id" in alert
        assert "name" in alert
        assert "level" in alert


@pytest.mark.asyncio
async def test_create_alert(client, mock_current_user, mock_analytics_service):
    """Testa a criação de um alerta."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Prepara os dados da requisição
        alert_data = {
            "id": str(uuid.uuid4()),
            "name": "Drawdown Alert",
            "description": "Alert when drawdown exceeds threshold",
            "type": "drawdown",
            "metric": "max_drawdown",
            "condition": ">",
            "threshold": 5.0,
            "level": "warning",
            "notification_channels": ["email", "app"],
            "cooldown_period": 3600,
            "is_active": True,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        # Faz a requisição
        response = client.post("/alerts", json=alert_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.create_alert.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "id" in result
        assert "name" in result
        assert "level" in result


@pytest.mark.asyncio
async def test_get_alert_history(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção do histórico de alertas."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição
        response = client.get("/alerts/history")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_alert_history.called
        
        # Verifica o tipo da resposta
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos principais da resposta
        alert = result[0]
        assert "id" in alert
        assert "config_id" in alert
        assert "triggered_at" in alert
        assert "level" in alert


@pytest.mark.asyncio
async def test_get_alert_history_with_filters(client, mock_current_user, mock_analytics_service):
    """Testa a obtenção do histórico de alertas com filtros."""
    # Configura os mocks
    with patch("src.api.controllers.analytics_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.analytics_controller.get_analytics_service", return_value=mock_analytics_service):
        
        # Faz a requisição com parâmetros de filtro
        response = client.get("/alerts/history?page=2&size=10&level=warning&acknowledged=false")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_analytics_service.get_alert_history.called
        
        # Verifica se os filtros foram passados corretamente
        _, kwargs = mock_analytics_service.get_alert_history.call_args
        assert kwargs["page"] == 2
        assert kwargs["size"] == 10
        assert "filters" in kwargs
        assert kwargs["filters"]["level"] == "warning"
        assert kwargs["filters"]["acknowledged"] is False