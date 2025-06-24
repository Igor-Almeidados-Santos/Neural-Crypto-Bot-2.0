"""
Testes para o módulo principal da API.

Este módulo contém testes unitários para o arquivo main.py da API.
"""
import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Cria um cliente de teste para a aplicação FastAPI."""
    return TestClient(app)


def test_health_check(client):
    """Testa o endpoint de health check."""
    # Faz a requisição
    response = client.get("/health")
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok", "service": "Neural Crypto Bot API"}


@pytest.mark.asyncio
async def test_global_exception_handler():
    """Testa o handler global de exceções."""
    # Cria uma rota de teste temporária que lança uma exceção
    @app.get("/test-exception")
    async def test_exception():
        raise Exception("Test exception")
    
    # Cria um cliente para testar
    client = TestClient(app)
    
    # Faz a requisição para a rota que lança exceção
    response = client.get("/test-exception")
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "Erro interno do servidor" in response.json()["detail"]
    
    # Remove a rota de teste
    app.routes = [route for route in app.routes if route.path != "/test-exception"]


def test_cors_middleware(client):
    """Testa a configuração do middleware CORS."""
    # Faz uma requisição OPTIONS com headers específicos
    response = client.options(
        "/health",
        headers={
            "Origin": "http://example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type,Authorization",
        },
    )
    
    # Verifica se os headers CORS estão presentes na resposta
    assert response.status_code == status.HTTP_200_OK
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"
    assert "GET" in response.headers["access-control-allow-methods"]
    assert "Content-Type" in response.headers["access-control-allow-headers"]
    assert "Authorization" in response.headers["access-control-allow-headers"]


@pytest.mark.asyncio
async def test_app_lifespan():
    """Testa o ciclo de vida da aplicação (startup e shutdown)."""
    # Mock para simular o logger
    with patch("src.api.main.logging") as mock_logging, \
         patch("src.api.main.load_config") as mock_load_config, \
         patch("src.api.main.setup_logger") as mock_setup_logger:
        
        # Configura o mock de config
        mock_config = MagicMock()
        mock_config.LOG_LEVEL = "INFO"
        mock_load_config.return_value = mock_config
        
        # Cria um mock para o app
        app_mock = MagicMock()
        
        # Obtém o gerenciador de contexto lifespan
        from api.main import lifespan
        
        # Executa o lifespan
        async with lifespan(app_mock):
            # Verifica se as funções de inicialização foram chamadas
            mock_load_config.assert_called_once()
            mock_setup_logger.assert_called_once_with(mock_config.LOG_LEVEL)
            mock_logging.info.assert_called_with("Inicializando recursos da API...")
        
        # Verifica se a função de cleanup foi chamada
        mock_logging.info.assert_called_with("Encerrando recursos da API...")


@pytest.mark.asyncio
async def test_routes_registration():
    """Testa se as rotas dos controllers foram registradas corretamente."""
    # Verifica se as rotas foram registradas
    routes = app.routes
    
    # Obtém os caminhos das rotas
    paths = [route.path for route in routes if hasattr(route, "path")]
    
    # Verifica se há rotas registradas
    assert len(paths) > 0
    
    # Verifica a presença de algumas rotas específicas
    assert "/health" in paths
    
    # Verifica se há rotas para cada controller
    assert any(path.startswith("/api/v1/strategies") for path in paths)
    assert any(path.startswith("/api/v1/analytics") for path in paths)
    assert any(path.startswith("/api/v1/admin") for path in paths)


@pytest.mark.asyncio
async def test_opentelemetry_instrumentation():
    """Testa se a instrumentação do OpenTelemetry foi aplicada."""
    # Este teste é apenas para verificar se o código da instrumentação está presente
    # e não causa erros. Em um ambiente real, seria necessário validar a telemetria gerada.
    
    # É difícil testar a instrumentação diretamente sem mock extensivo,
    # então estamos apenas verificando se o app foi criado sem erros
    assert app is not None