"""
Testes para o Middleware de Autenticação.

Este módulo contém testes unitários para o middleware de autenticação da API.
"""
import json
import time
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from jose import jwt

from api.middlewares.auth_middleware import AuthMiddleware


@pytest.fixture
def mock_settings():
    """Mock para as configurações da aplicação."""
    return MagicMock(
        SECRET_KEY="test-secret-key",
        JWT_ALGORITHM="HS256",
    )


@pytest.fixture
def valid_token(mock_settings):
    """Gera um token JWT válido para testes."""
    # Cria um payload com as informações necessárias
    payload = {
        "sub": "test_user",
        "email": "test@example.com",
        "roles": ["trader"],
        "exp": int(time.time()) + 3600  # Token válido por 1 hora
    }
    
    # Gera o token usando o secret e algoritmo da configuração
    return jwt.encode(
        payload,
        mock_settings.SECRET_KEY,
        algorithm=mock_settings.JWT_ALGORITHM
    )


@pytest.fixture
def expired_token(mock_settings):
    """Gera um token JWT expirado para testes."""
    # Cria um payload com as informações necessárias, mas com exp no passado
    payload = {
        "sub": "test_user",
        "email": "test@example.com",
        "roles": ["trader"],
        "exp": int(time.time()) - 3600  # Token expirado há 1 hora
    }
    
    # Gera o token usando o secret e algoritmo da configuração
    return jwt.encode(
        payload,
        mock_settings.SECRET_KEY,
        algorithm=mock_settings.JWT_ALGORITHM
    )


@pytest.fixture
def invalid_signature_token():
    """Gera um token JWT com assinatura inválida para testes."""
    # Cria um payload com as informações necessárias
    payload = {
        "sub": "test_user",
        "email": "test@example.com",
        "roles": ["trader"],
        "exp": int(time.time()) + 3600
    }
    
    # Gera o token usando um secret diferente
    return jwt.encode(
        payload,
        "wrong-secret-key",
        algorithm="HS256"
    )


@pytest.fixture
def app():
    """Cria uma instância da aplicação FastAPI para testes."""
    app = FastAPI()
    
    # Adiciona uma rota protegida para teste
    @app.get("/protected")
    async def protected_route(request: Request):
        return {"user": request.state.user, "message": "Success"}
    
    # Adiciona uma rota pública para teste
    @app.get("/public")
    async def public_route():
        return {"message": "Public route"}
    
    return app


@pytest.fixture
def client(app, mock_settings):
    """Cria um cliente de teste para a aplicação FastAPI com o middleware de autenticação."""
    # Configura o middleware
    app.add_middleware(
        AuthMiddleware,
        exclude_paths=["/public"]
    )
    
    # Patch a função get_settings para retornar os mocks
    with patch("src.api.middlewares.auth_middleware.get_settings", return_value=mock_settings):
        yield TestClient(app)


@pytest.mark.asyncio
async def test_auth_middleware_protected_route_no_token(client):
    """Testa acessar uma rota protegida sem token."""
    # Faz a requisição sem token
    response = client.get("/protected")
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token de autenticação não fornecido" in response.json()["detail"]
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_auth_middleware_protected_route_valid_token(client, valid_token):
    """Testa acessar uma rota protegida com token válido."""
    # Faz a requisição com token válido
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {valid_token}"}
    )
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_200_OK
    assert "Success" in response.json()["message"]
    assert "user" in response.json()
    assert response.json()["user"]["sub"] == "test_user"


@pytest.mark.asyncio
async def test_auth_middleware_protected_route_expired_token(client, expired_token):
    """Testa acessar uma rota protegida com token expirado."""
    # Faz a requisição com token expirado
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {expired_token}"}
    )
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token de autenticação inválido ou expirado" in response.json()["detail"]
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_auth_middleware_protected_route_invalid_token(client, invalid_signature_token):
    """Testa acessar uma rota protegida com token de assinatura inválida."""
    # Faz a requisição com token inválido
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {invalid_signature_token}"}
    )
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token de autenticação inválido" in response.json()["detail"]
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_auth_middleware_public_route(client):
    """Testa acessar uma rota pública sem token."""
    # Faz a requisição sem token
    response = client.get("/public")
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Public route"}


@pytest.mark.asyncio
async def test_auth_middleware_invalid_auth_header_format(client):
    """Testa acessar uma rota protegida com formato de header inválido."""
    # Faz a requisição com formato de header inválido
    response = client.get(
        "/protected",
        headers={"Authorization": "InvalidFormat"}
    )
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token de autenticação não fornecido" in response.json()["detail"]
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_auth_middleware_no_exp_claim(client, mock_settings):
    """Testa acessar uma rota protegida com token sem claim 'exp'."""
    # Cria um payload sem o campo 'exp'
    payload = {
        "sub": "test_user",
        "email": "test@example.com",
        "roles": ["trader"]
    }
    
    # Gera o token
    token = jwt.encode(
        payload,
        mock_settings.SECRET_KEY,
        algorithm=mock_settings.JWT_ALGORITHM
    )
    
    # Faz a requisição
    response = client.get(
        "/protected",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    # Verifica o resultado
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Token de autenticação inválido ou expirado" in response.json()["detail"]
    assert response.headers.get("WWW-Authenticate") == "Bearer"


@pytest.mark.asyncio
async def test_auth_middleware_extract_token():
    """Testa o método de extração de token do header de autorização."""
    # Cria uma instância do middleware
    middleware = AuthMiddleware(app=None)
    
    # Cria um request mock com header de autorização
    request = MagicMock()
    request.headers = {"Authorization": "Bearer test-token"}
    
    # Testa a extração do token
    token = middleware._extract_token(request)
    assert token == "test-token"


@pytest.mark.asyncio
async def test_auth_middleware_extract_token_missing_header():
    """Testa o método de extração de token com header ausente."""
    # Cria uma instância do middleware
    middleware = AuthMiddleware(app=None)
    
    # Cria um request mock sem header de autorização
    request = MagicMock()
    request.headers = {}
    
    # Testa a extração do token
    token = middleware._extract_token(request)
    assert token is None


@pytest.mark.asyncio
async def test_auth_middleware_extract_token_invalid_format():
    """Testa o método de extração de token com formato inválido."""
    # Cria uma instância do middleware
    middleware = AuthMiddleware(app=None)
    
    # Cria um request mock com header de formato inválido
    request = MagicMock()
    request.headers = {"Authorization": "InvalidFormat"}
    
    # Testa a extração do token
    token = middleware._extract_token(request)
    assert token is None


@pytest.mark.asyncio
async def test_auth_middleware_is_path_excluded():
    """Testa o método que verifica se um caminho está excluído da autenticação."""
    # Cria uma instância do middleware com caminhos excluídos
    middleware = AuthMiddleware(app=None, exclude_paths=["/public", "/docs"])
    
    # Testa caminhos excluídos
    assert middleware._is_path_excluded("/public") is True
    assert middleware._is_path_excluded("/docs") is True
    assert middleware._is_path_excluded("/docs/") is True
    assert middleware._is_path_excluded("/docs/oauth2-redirect") is True
    
    # Testa caminhos não excluídos
    assert middleware._is_path_excluded("/api/v1/strategies") is False
    assert middleware._is_path_excluded("/protected") is False


@pytest.mark.asyncio
async def test_auth_middleware_validate_token(mock_settings, valid_token):
    """Testa o método de validação de token."""
    # Cria uma instância do middleware
    middleware = AuthMiddleware(app=None)
    
    # Patch a função get_settings para retornar os mocks
    with patch("src.api.middlewares.auth_middleware.get_settings", return_value=mock_settings):
        # Valida um token válido
        user_data = middleware._validate_token(valid_token)
        
        # Verifica se os dados do usuário foram extraídos corretamente
        assert user_data is not None
        assert user_data["sub"] == "test_user"
        assert user_data["email"] == "test@example.com"
        assert "trader" in user_data["roles"]