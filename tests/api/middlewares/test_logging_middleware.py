"""
Testes para o Middleware de Logging.

Este módulo contém testes unitários para o middleware de logging da API.
"""
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from src.api.middlewares.logging_middleware import LoggingMiddleware


@pytest.fixture
def mock_settings():
    """Mock para as configurações da aplicação."""
    return MagicMock(
        LOG_LEVEL="INFO",
    )


@pytest.fixture
def app():
    """Cria uma instância da aplicação FastAPI para testes."""
    app = FastAPI()
    
    # Adiciona uma rota normal para teste
    @app.get("/test")
    async def test_route(request: Request):
        return {"message": "Success", "request_id": getattr(request.state, "request_id", None)}
    
    # Adiciona uma rota que lança exceção para teste
    @app.get("/error")
    async def error_route():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Test error")
    
    # Adiciona uma rota que lança exceção não tratada para teste
    @app.get("/unexpected-error")
    async def unexpected_error_route():
        raise ValueError("Unexpected test error")
    
    # Adiciona uma rota GET com query params para teste
    @app.get("/query-params")
    async def query_params_route(param1: str, param2: int = 0):
        return {"param1": param1, "param2": param2}
    
    # Adiciona uma rota POST para teste de logging de corpo
    @app.post("/body")
    async def body_route(body: dict):
        return body
    
    return app


@pytest.fixture
def client(app, mock_settings):
    """Cria um cliente de teste para a aplicação FastAPI com o middleware de logging."""
    # Configura o middleware
    app.add_middleware(LoggingMiddleware)
    
    # Patch a função get_settings para retornar os mocks
    with patch("src.api.middlewares.logging_middleware.get_settings", return_value=mock_settings):
        yield TestClient(app)


@pytest.mark.asyncio
async def test_logging_middleware_successful_request(client):
    """Testa o middleware de logging para uma requisição bem-sucedida."""
    # Configura o mock para o logger
    with patch("src.api.middlewares.logging_middleware.logger") as mock_logger:
        # Faz a requisição
        response = client.get("/test")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert "Success" in response.json()["message"]
        
        # Verifica se os logs foram registrados
        assert mock_logger.info.call_count >= 2  # Pelo menos 2 logs: início e fim da requisição
        
        # Verifica se o request_id está presente na resposta e nos headers
        assert "request_id" in response.json()
        assert response.json()["request_id"] is not None
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers


@pytest.mark.asyncio
async def test_logging_middleware_handled_error(client):
    """Testa o middleware de logging para uma requisição com erro tratado."""
    # Configura o mock para o logger
    with patch("src.api.middlewares.logging_middleware.logger") as mock_logger:
        # Faz a requisição
        response = client.get("/error")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Test error" in response.json()["detail"]
        
        # Verifica se os logs foram registrados
        assert mock_logger.info.call_count >= 2  # Pelo menos 2 logs: início e fim da requisição
        
        # Verifica se os headers de rastreabilidade estão presentes
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers


@pytest.mark.asyncio
async def test_logging_middleware_unhandled_error(client):
    """Testa o middleware de logging para uma requisição com erro não tratado."""
    # Configura o mock para o logger
    with patch("src.api.middlewares.logging_middleware.logger") as mock_logger:
        # Faz a requisição (que vai gerar um erro 500 devido ao erro não tratado)
        response = client.get("/unexpected-error")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        
        # Verifica se os logs foram registrados
        assert mock_logger.info.called  # Log de início da requisição
        assert mock_logger.error.called  # Log de erro
        
        # Verifica se o erro foi registrado com o traceback
        mock_logger.error.assert_called_once()
        error_call_args = mock_logger.error.call_args[0][0]
        assert "Request failed" in error_call_args
        assert "Unexpected test error" in error_call_args
        
        # Verifica se exc_info=True foi passado para incluir o traceback
        assert mock_logger.error.call_args[1]["exc_info"] is True


@pytest.mark.asyncio
async def test_logging_middleware_query_params(client):
    """Testa o middleware de logging para uma requisição com query params."""
    # Configura o mock para o logger
    with patch("src.api.middlewares.logging_middleware.logger") as mock_logger:
        # Faz a requisição com query params
        response = client.get("/query-params?param1=test&param2=123")
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"param1": "test", "param2": 123}
        
        # Verifica se os logs foram registrados
        assert mock_logger.info.call_count >= 2  # Pelo menos 2 logs: início e fim da requisição
        
        # Verifica se os query params foram capturados no log
        info_call_args = mock_logger.info.call_args_list[0][0][0]
        assert "Request started" in info_call_args
        assert "/query-params" in info_call_args


@pytest.mark.asyncio
async def test_logging_middleware_post_body(client):
    """Testa o middleware de logging para uma requisição POST com corpo."""
    # Este teste verifica apenas a funcionalidade principal, pois o método _log_request_body
    # é assíncrono e precisaria de um mock mais complexo para testar completamente
    
    # Configura o mock para o logger
    with patch("src.api.middlewares.logging_middleware.logger") as mock_logger:
        # Faz a requisição POST com body
        body_data = {"name": "Test", "value": 123}
        response = client.post("/body", json=body_data)
        
        # Verifica o resultado
        assert response.status_code == status.HTTP_200_OK
        assert response.json() == body_data
        
        # Verifica se os logs foram registrados
        assert mock_logger.info.call_count >= 2  # Pelo menos 2 logs: início e fim da requisição


@pytest.mark.asyncio
async def test_get_client_ip_method():
    """Testa o método _get_client_ip do middleware."""
    # Cria uma instância do middleware
    middleware = LoggingMiddleware(app=None)
    
    # Teste com X-Forwarded-For
    request = MagicMock()
    request.headers = {"X-Forwarded-For": "192.168.0.1, 10.0.0.1"}
    client_ip = middleware._get_client_ip(request)
    assert client_ip == "192.168.0.1"
    
    # Teste com X-Real-IP
    request = MagicMock()
    request.headers = {"X-Real-IP": "192.168.0.2"}
    client_ip = middleware._get_client_ip(request)
    assert client_ip == "192.168.0.2"
    
    # Teste com CF-Connecting-IP
    request = MagicMock()
    request.headers = {"CF-Connecting-IP": "192.168.0.3"}
    client_ip = middleware._get_client_ip(request)
    assert client_ip == "192.168.0.3"
    
    # Teste sem headers especiais
    request = MagicMock()
    request.headers = {}
    request.client = MagicMock(host="192.168.0.4")
    client_ip = middleware._get_client_ip(request)
    assert client_ip == "192.168.0.4"
    
    # Teste sem client
    request = MagicMock()
    request.headers = {}
    request.client = None
    client_ip = middleware._get_client_ip(request)
    assert client_ip == "unknown"


@pytest.mark.asyncio
async def test_logging_middleware_metrics(client):
    """Testa se as métricas Prometheus estão sendo atualizadas."""
    # Configura o mock para o Histogram do Prometheus
    with patch("src.api.middlewares.logging_middleware.REQUEST_LATENCY") as mock_histogram:
        # Faz a requisição
        response = client.get("/test")
        
        # Verifica se o método de observação foi chamado
        assert mock_histogram.labels.called
        mock_histogram.labels.assert_called_with(method="GET", endpoint="/test")
        assert mock_histogram.labels.return_value.observe.called


@pytest.mark.asyncio
async def test_log_request_body():
    """Testa o método _log_request_body do middleware."""
    # Cria uma instância do middleware
    middleware = LoggingMiddleware(app=None)
    
    # Mock para o settings
    mock_settings = MagicMock(LOG_LEVEL="INFO")
    
    # Teste com método GET (não deve ler o body)
    request = MagicMock()
    request.method = "GET"
    
    with patch("src.api.middlewares.logging_middleware.get_settings", return_value=mock_settings):
        result = await middleware._log_request_body(request)
        assert result == {"body": "[No body]"}
    
    # Teste com método POST e body JSON
    request = MagicMock()
    request.method = "POST"
    request.json = AsyncMock(return_value={"username": "test", "password": "secret"})
    
    with patch("src.api.middlewares.logging_middleware.get_settings", return_value=mock_settings):
        result = await middleware._log_request_body(request)
        assert "body" in result
        # Verifica se o campo sensível foi redatado
        assert "***REDACTED***" in result["body"]
    
    # Teste com método POST e body que falha ao ser lido como JSON
    request = MagicMock()
    request.method = "POST"
    request.json = AsyncMock(side_effect=ValueError("Invalid JSON"))
    request.body = AsyncMock(return_value=b"binary data")
    
    with patch("src.api.middlewares.logging_middleware.get_settings", return_value=mock_settings):
        result = await middleware._log_request_body(request)
        assert "body" in result
        assert "Binary data" in result["body"]