"""
Testes para o Strategy Controller.

Este módulo contém testes unitários para o controller de estratégias.
"""
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.controllers import strategy_controller
from api.dtos.strategy_dto import (
    BacktestRequest,
    BacktestResult,
    OrderRequest,
    OrderResponse,
    SignalRequest,
    SignalResponse,
    StrategyCreate,
    StrategyResponse,
    StrategyStatus,
    StrategyType,
    StrategyUpdate,
    TimeFrame,
)
from strategy_engine.domain.entities.strategy import Strategy


@pytest.fixture
def app():
    """Cria uma instância da aplicação FastAPI para testes."""
    app = FastAPI()
    app.include_router(strategy_controller.router)
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
    strategy.description = "Test strategy description"
    strategy.type = StrategyType.MEAN_REVERSION
    strategy.trading_pairs = ["BTC/USDT"]
    strategy.timeframe = TimeFrame.HOUR_1
    strategy.owner_id = "test_user"
    strategy.is_active = False
    strategy.status = StrategyStatus.PAUSED
    strategy.created_at = datetime.utcnow()
    strategy.updated_at = datetime.utcnow()
    strategy.to_dict = MagicMock(return_value={
        "id": strategy.id,
        "name": strategy.name,
        "description": strategy.description,
        "type": strategy.type,
        "trading_pairs": strategy.trading_pairs,
        "timeframe": strategy.timeframe,
        "owner_id": strategy.owner_id,
        "is_active": strategy.is_active,
        "status": strategy.status,
        "created_at": strategy.created_at,
        "updated_at": strategy.updated_at,
        "parameters": [],
    })
    
    repo.get_by_id.return_value = strategy
    
    # Setup do mock para create_from_dto
    repo.create_from_dto.return_value = strategy
    
    # Setup do mock para update
    repo.update.return_value = strategy
    
    # Setup do mock para list_with_pagination
    repo.list_with_pagination.return_value = ([strategy], 1)
    
    return repo


@pytest.fixture
def mock_execute_strategy_uc():
    """Mock para o caso de uso de execução de estratégia."""
    uc = AsyncMock()
    
    # Setup do mock para generate_signals
    signal = MagicMock(spec=SignalResponse)
    signal.strategy_id = str(uuid.uuid4())
    signal.timestamp = datetime.utcnow()
    signal.trading_pair = "BTC/USDT"
    signal.side = "long"
    signal.price = 64000.0
    signal.confidence = 0.85
    
    uc.generate_signals.return_value = [signal]
    
    return uc


@pytest.fixture
def mock_backtest_uc():
    """Mock para o caso de uso de backtest."""
    uc = AsyncMock()
    
    # Setup do mock para execute
    result = MagicMock(spec=BacktestResult)
    result.strategy_id = str(uuid.uuid4())
    result.start_date = datetime(2024, 1, 1)
    result.end_date = datetime(2025, 1, 1)
    result.initial_capital = 10000.0
    result.final_capital = 15000.0
    result.total_return = 0.5
    result.annualized_return = 0.4
    result.max_drawdown = 0.15
    result.sharpe_ratio = 1.8
    
    uc.execute.return_value = result
    
    return uc


@pytest.fixture
def mock_execute_order_uc():
    """Mock para o caso de uso de execução de ordem."""
    uc = AsyncMock()
    
    # Setup do mock para execute
    order = MagicMock(spec=OrderResponse)
    order.id = str(uuid.uuid4())
    order.strategy_id = str(uuid.uuid4())
    order.trading_pair = "BTC/USDT"
    order.side = "long"
    order.type = "market"
    order.quantity = 0.1
    order.status = "filled"
    
    uc.execute.return_value = order
    
    return uc


@pytest.mark.asyncio
async def test_create_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a criação de uma nova estratégia."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Prepara os dados da requisição
        strategy_data = {
            "name": "Test Strategy",
            "description": "Test strategy description",
            "type": "mean_reversion",
            "trading_pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "parameters": [],
            "max_position_size": 0.1,
            "max_drawdown_percent": 5.0,
            "take_profit_percent": 2.0,
            "stop_loss_percent": 1.0,
            "is_active": False
        }
        
        # Faz a requisição
        response = client.post("/", json=strategy_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_201_CREATED
        assert mock_strategy_repo.create_from_dto.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert result["name"] == "Test Strategy"
        assert result["description"] == "Test strategy description"
        assert result["type"] == "mean_reversion"
        assert result["trading_pairs"] == ["BTC/USDT"]
        assert result["timeframe"] == "1h"


@pytest.mark.asyncio
async def test_create_strategy_invalid_pair(client, mock_current_user, mock_strategy_repo):
    """Testa a criação de uma estratégia com par de trading inválido."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo), \
         patch("src.api.controllers.strategy_controller.validate_trading_pair", return_value=False):
        
        # Prepara os dados da requisição com par inválido
        strategy_data = {
            "name": "Test Strategy",
            "description": "Test strategy description",
            "type": "mean_reversion",
            "trading_pairs": ["INVALID_PAIR"],
            "timeframe": "1h",
            "parameters": [],
        }
        
        # Faz a requisição
        response = client.post("/", json=strategy_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Par de trading inválido" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a obtenção de uma estratégia por ID."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.get(f"/{strategy_id}")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert result["name"] == "Test Strategy"
        assert result["description"] == "Test strategy description"
        assert result["type"] == "mean_reversion"


@pytest.mark.asyncio
async def test_get_strategy_not_found(client, mock_current_user, mock_strategy_repo):
    """Testa a obtenção de uma estratégia inexistente."""
    # Configura o mock para retornar None (estratégia não encontrada)
    mock_strategy_repo.get_by_id.return_value = None
    
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.get(f"/{strategy_id}")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert mock_strategy_repo.get_by_id.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)


@pytest.mark.asyncio
async def test_get_strategy_unauthorized(client, mock_current_user, mock_strategy_repo):
    """Testa a obtenção de uma estratégia com usuário não autorizado."""
    # Modifica o proprietário da estratégia para ser diferente do usuário atual
    strategy = mock_strategy_repo.get_by_id.return_value
    strategy.owner_id = "another_user"
    
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.get(f"/{strategy_id}")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert mock_strategy_repo.get_by_id.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)


@pytest.mark.asyncio
async def test_list_strategies(client, mock_current_user, mock_strategy_repo):
    """Testa a listagem de estratégias."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Faz a requisição
        response = client.get("/")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.list_with_pagination.called
        
        # Verifica os campos principais da resposta
        result = response.json()
        assert "items" in result
        assert len(result["items"]) == 1
        assert result["total"] == 1
        assert result["page"] == 1
        assert result["size"] == 10
        
        # Verifica o item na lista
        item = result["items"][0]
        assert item["name"] == "Test Strategy"
        assert item["description"] == "Test strategy description"
        assert item["type"] == "mean_reversion"


@pytest.mark.asyncio
async def test_update_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a atualização de uma estratégia."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Prepara os dados da requisição
        update_data = {
            "name": "Updated Strategy",
            "description": "Updated description",
            "trading_pairs": ["ETH/USDT"],
            "timeframe": "4h",
            "max_position_size": 0.2,
        }
        
        # Faz a requisição
        response = client.put(f"/{strategy_id}", json=update_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_strategy_repo.update.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        
        # Verifica os campos da resposta
        result = response.json()
        assert result["name"] == "Test Strategy"  # Estamos usando o mock sempre retornando o mesmo objeto
        assert result["description"] == "Test strategy description"
        assert result["type"] == "mean_reversion"


@pytest.mark.asyncio
async def test_delete_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a remoção de uma estratégia."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.delete(f"/{strategy_id}")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_204_NO_CONTENT
        assert mock_strategy_repo.get_by_id.called
        assert mock_strategy_repo.delete.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        mock_strategy_repo.delete.assert_called_with(strategy_id)


@pytest.mark.asyncio
async def test_delete_active_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a tentativa de remoção de uma estratégia ativa."""
    # Modifica o estado da estratégia para ativo
    strategy = mock_strategy_repo.get_by_id.return_value
    strategy.is_active = True
    
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.delete(f"/{strategy_id}")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "estratégia ativa" in response.json()["detail"]
        assert mock_strategy_repo.get_by_id.called
        assert not mock_strategy_repo.delete.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)


@pytest.mark.asyncio
async def test_activate_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a ativação de uma estratégia."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.post(f"/{strategy_id}/activate")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_strategy_repo.save.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        
        # Verifica se a estratégia foi ativada
        strategy = mock_strategy_repo.get_by_id.return_value
        assert strategy.is_active is True
        assert strategy.status == "active"


@pytest.mark.asyncio
async def test_deactivate_strategy(client, mock_current_user, mock_strategy_repo):
    """Testa a desativação de uma estratégia."""
    # Configura a estratégia como ativa inicialmente
    strategy = mock_strategy_repo.get_by_id.return_value
    strategy.is_active = True
    strategy.status = StrategyStatus.ACTIVE
    
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Faz a requisição
        response = client.post(f"/{strategy_id}/deactivate")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_strategy_repo.save.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        
        # Verifica se a estratégia foi desativada
        strategy = mock_strategy_repo.get_by_id.return_value
        assert strategy.is_active is False
        assert strategy.status == "paused"


@pytest.mark.asyncio
async def test_run_backtest(client, mock_current_user, mock_strategy_repo, mock_backtest_uc):
    """Testa a execução de backtest em uma estratégia."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo), \
         patch("src.api.controllers.strategy_controller.get_backtest_strategy_use_case", return_value=mock_backtest_uc):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Prepara os dados da requisição
        backtest_data = {
            "strategy_id": str(strategy_id),
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2025-01-01T00:00:00Z",
            "initial_capital": 10000.0,
            "trading_fee": 0.001,
            "slippage": 0.0005,
            "use_historical_data": True,
            "include_detailed_trades": False
        }
        
        # Faz a requisição
        response = client.post(f"/{strategy_id}/backtest", json=backtest_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_backtest_uc.execute.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        
        # Verifica os campos da resposta
        result = response.json()
        assert "strategy_id" in result
        assert "initial_capital" in result
        assert "final_capital" in result
        assert "total_return" in result


@pytest.mark.asyncio
async def test_generate_signals(client, mock_current_user, mock_strategy_repo, mock_execute_strategy_uc):
    """Testa a geração de sinais para uma estratégia."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_strategy_repository", return_value=mock_strategy_repo), \
         patch("src.api.controllers.strategy_controller.get_execute_strategy_use_case", return_value=mock_execute_strategy_uc):
        
        # Gera um ID para o teste
        strategy_id = str(uuid.uuid4())
        
        # Prepara os dados da requisição
        signal_request = {
            "strategy_id": str(strategy_id),
            "trading_pairs": ["BTC/USDT", "ETH/USDT"],
            "generate_orders": False
        }
        
        # Faz a requisição
        response = client.post(f"/{strategy_id}/signals", json=signal_request)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_strategy_repo.get_by_id.called
        assert mock_execute_strategy_uc.generate_signals.called
        mock_strategy_repo.get_by_id.assert_called_with(strategy_id)
        
        # Verifica se a resposta contém sinais
        result = response.json()
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Verifica os campos do sinal
        signal = result[0]
        assert "strategy_id" in signal
        assert "trading_pair" in signal
        assert "side" in signal
        assert "price" in signal
        assert "confidence" in signal


@pytest.mark.asyncio
async def test_create_order(client, mock_current_user, mock_execute_order_uc):
    """Testa a criação de uma ordem de trading."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_execute_order_use_case", return_value=mock_execute_order_uc), \
         patch("src.api.controllers.strategy_controller.validate_trading_pair", return_value=True):
        
        # Prepara os dados da requisição
        order_data = {
            "trading_pair": "BTC/USDT",
            "side": "long",
            "type": "market",
            "quantity": 0.1,
            "price": None,
            "exchange": "binance"
        }
        
        # Faz a requisição
        response = client.post("/orders", json=order_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert mock_execute_order_uc.execute.called
        
        # Verifica os campos da resposta
        result = response.json()
        assert "id" in result
        assert "trading_pair" in result
        assert "side" in result
        assert "type" in result
        assert "quantity" in result
        assert "status" in result


@pytest.mark.asyncio
async def test_create_order_invalid_pair(client, mock_current_user, mock_execute_order_uc):
    """Testa a criação de uma ordem com par de trading inválido."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
        patch("src.api.controllers.strategy_controller.get_execute_order_use_case", return_value=mock_execute_order_uc), \
        patch("src.api.controllers.strategy_controller.validate_trading_pair", return_value=False):
        
        # Prepara os dados da requisição
        order_data = {
            "trading_pair": "INVALID_PAIR",
            "side": "long",
            "type": "market",
            "quantity": 0.1,
            "price": None,
            "exchange": "binance"
        }
        
        # Faz a requisição
        response = client.post("/orders", json=order_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Par de trading inválido" in response.json()["detail"]
        assert not mock_execute_order_uc.execute.called


@pytest.mark.asyncio
async def test_create_limit_order_without_price(client, mock_current_user, mock_execute_order_uc):
    """Testa a criação de uma ordem limit sem informar o preço."""
    # Configura os mocks
    with patch("src.api.controllers.strategy_controller.get_current_user", return_value=mock_current_user), \
         patch("src.api.controllers.strategy_controller.get_execute_order_use_case", return_value=mock_execute_order_uc), \
         patch("src.api.controllers.strategy_controller.validate_trading_pair", return_value=True):
        
        # Prepara os dados da requisição sem preço para uma ordem limit
        order_data = {
            "trading_pair": "BTC/USDT",
            "side": "long",
            "type": "limit",
            "quantity": 0.1,
            "price": None,
            "exchange": "binance"
        }
        
        # Faz a requisição
        response = client.post("/orders", json=order_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Preço é obrigatório" in response.json()["detail"]
        assert not mock_execute_order_uc.execute.called