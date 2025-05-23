"""
Neural Crypto Bot - API Version Middleware

Este módulo implementa o middleware de versionamento da API.
Responsável por rotear requisições para a versão correta da API e
controlar endpoints obsoletos.
"""
import logging
import re
from typing import Callable, Dict, List, Optional, Set, Union
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.version import APIVersion, VersionedEndpoint
from src.common.utils.config import get_settings

# Configure o logger
logger = logging.getLogger(__name__)


class VersionMiddleware(BaseHTTPMiddleware):
    """
    Middleware para gerenciar o versionamento da API.
    
    Este middleware é responsável por:
    1. Identificar a versão da API solicitada (via URL ou cabeçalho)
    2. Rotear requisições para a versão correta da API
    3. Lidar com endpoints obsoletos e redirecionamentos
    4. Adicionar cabeçalhos relacionados à versão
    """

    def __init__(
        self,
        app: ASGIApp,
        deprecated_endpoints: Optional[List[VersionedEndpoint]] = None,
        default_version: str = APIVersion.latest(),
    ):
        """
        Inicializa o middleware de versionamento.
        
        Args:
            app: A aplicação FastAPI.
            deprecated_endpoints: Lista de endpoints obsoletos e suas configurações.
            default_version: Versão padrão da API a ser usada quando não especificada.
        """
        super().__init__(app)
        self.settings = get_settings()
        self.deprecated_endpoints = deprecated_endpoints or []
        self.default_version = default_version
        
        # Verifica se a versão padrão é válida
        if self.default_version not in APIVersion.all():
            self.default_version = APIVersion.latest()
        
        # Compila expressões regulares para endpoints obsoletos
        self.deprecated_patterns = []
        for endpoint in self.deprecated_endpoints:
            pattern = endpoint.path.replace("{", "(?P<").replace("}", ">\\w+)")
            self.deprecated_patterns.append((re.compile(f"^{pattern}$"), endpoint))
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa a requisição, identificando a versão da API e gerenciando endpoints obsoletos.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            call_next: Função para processar o próximo middleware/endpoint.
            
        Returns:
            Objeto Response com a resposta da API ou redirecionamento.
        """
        # Extrai a versão da API da requisição
        api_version = self._extract_version(request)
        
        # Armazena a versão na requisição para uso posterior
        request.state.api_version = api_version
        
        # Verifica se a requisição está acessando um endpoint obsoleto
        path = request.url.path
        deprecated_endpoint = self._check_deprecated_endpoint(path, api_version)
        
        if deprecated_endpoint:
            # Se o endpoint está obsoleto e tem redirecionamento, redireciona
            redirect_path = deprecated_endpoint.get_redirect_path(path)
            if redirect_path:
                return RedirectResponse(
                    url=redirect_path,
                    status_code=status.HTTP_307_TEMPORARY_REDIRECT
                )
            
            # Se o endpoint está obsoleto e não tem redirecionamento, retorna erro
            return JSONResponse(
                status_code=status.HTTP_410_GONE,
                content={
                    "detail": f"Este endpoint está obsoleto e não está mais disponível na versão {api_version}",
                    "version": api_version,
                    "deprecated": True
                }
            )
        
        # Processa a requisição normalmente
        response = await call_next(request)
        
        # Adiciona cabeçalhos relacionados à versão
        response.headers["X-API-Version"] = api_version
        
        # Se a versão atual não for a mais recente, adiciona cabeçalho de aviso
        if api_version != APIVersion.latest():
            response.headers["X-API-Warning"] = f"Você está usando uma versão anterior da API. A versão mais recente é {APIVersion.latest()}"
        
        return response
    
    def _extract_version(self, request: Request) -> str:
        """
        Extrai a versão da API da requisição.
        
        A versão pode ser especificada de várias formas:
        1. Via caminho da URL: /api/v1/...
        2. Via cabeçalho: X-API-Version: v1
        3. Via parâmetro de consulta: ?api-version=v1
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            Versão da API.
        """
        # Verifica versão via caminho da URL
        path = request.url.path
        version_match = re.match(r"^/api/(v\d+)/", path)
        if version_match:
            version = version_match.group(1)
            return APIVersion.parse(version)
        
        # Verifica versão via cabeçalho
        header_version = request.headers.get("X-API-Version")
        if header_version:
            return APIVersion.parse(header_version)
        
        # Verifica versão via parâmetro de consulta
        query_version = request.query_params.get("api-version")
        if query_version:
            return APIVersion.parse(query_version)
        
        # Se não encontrou versão, usa a padrão
        return self.default_version
    
    def _check_deprecated_endpoint(self, path: str, version: str) -> Optional[VersionedEndpoint]:
        """
        Verifica se um caminho corresponde a um endpoint obsoleto para a versão especificada.
        
        Args:
            path: Caminho da URL.
            version: Versão da API.
            
        Returns:
            O endpoint obsoleto correspondente ou None se não for obsoleto.
        """
        # Verifica cada padrão de endpoint obsoleto
        for pattern, endpoint in self.deprecated_patterns:
            if pattern.match(path) and not endpoint.supports_version(version):
                return endpoint
        
        return None