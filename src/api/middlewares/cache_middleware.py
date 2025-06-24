"""
Neural Crypto Bot - Cache Middleware

Este módulo implementa o middleware de cache para a API.
Responsável por armazenar em cache respostas de endpoints para
melhorar a performance e reduzir a carga no servidor.
"""
import logging
import time
import json
import hashlib
from typing import Callable, Dict, List, Optional, Set, Tuple, Union
import re

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from common.infrastructure.cache.cache_manager import get_cache
from common.utils.config import get_settings

# Configure o logger
logger = logging.getLogger(__name__)


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Middleware para caching de respostas da API.
    
    Este middleware armazena em cache as respostas de endpoints GET específicos
    e retorna a versão em cache quando disponível, reduzindo a carga no servidor.
    """

    def __init__(
        self,
        app: ASGIApp,
        cache_ttl: int = 60,  # 60 segundos por padrão
        include_paths: Optional[List[str]] = None,
        exclude_paths: Optional[List[str]] = None,
        exclude_query_params: Optional[List[str]] = None,
        max_cache_size: int = 1024 * 1024 * 10,  # 10 MB
    ):
        """
        Inicializa o middleware de cache.
        
        Args:
            app: A aplicação FastAPI.
            cache_ttl: Tempo padrão de vida do cache em segundos.
            include_paths: Lista de caminhos (endpoints) que devem usar cache.
            exclude_paths: Lista de caminhos (endpoints) que não devem usar cache.
            exclude_query_params: Lista de parâmetros de consulta a excluir da chave de cache.
            max_cache_size: Tamanho máximo em bytes para entradas de cache.
        """
        super().__init__(app)
        self.settings = get_settings()
        self.cache_ttl = cache_ttl
        self.include_paths = include_paths or []
        self.exclude_paths = exclude_paths or [
            "/health",
            "/api/*/auth/*",
        ]
        self.exclude_query_params = exclude_query_params or [
            "t",  # Parâmetro timestamp usado para evitar cache
            "token",
            "access_token",
            "refresh_token",
        ]
        self.max_cache_size = max_cache_size
        
        # Definição de rotas com TTL específico
        self.ttl_map: Dict[str, int] = {
            # Rotas que mudam raramente
            "/api/*/strategies": 300,  # 5 minutos
            "/api/*/analytics/reports": 300,  # 5 minutos
            
            # Rotas que mudam com frequência moderada
            "/api/*/market/price": 30,  # 30 segundos
            "/api/*/market/orderbook": 10,  # 10 segundos
            
            # Rotas que mudam muito raramente
            "/api/*/exchanges": 3600,  # 1 hora
            "/api/*/settings/public": 3600,  # 1 hora
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa a requisição, armazenando ou recuperando do cache quando apropriado.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            call_next: Função para processar o próximo middleware/endpoint.
            
        Returns:
            Objeto Response com a resposta da API, do cache ou gerada.
        """
        # Só aplica cache para requisições GET
        if request.method != "GET":
            return await call_next(request)
        
        # Verifica se o caminho está nas exclusões ou inclusões
        path = request.url.path
        if not self._should_cache_path(path):
            return await call_next(request)
        
        # Verifica se a requisição está autenticada
        # Requisições autenticadas geralmente têm conteúdo personalizado
        if self._is_authenticated_request(request):
            # Só aplicamos cache para requisições autenticadas se for explicitamente incluído
            if not any(re.match(pattern, path) for pattern in self.include_paths):
                return await call_next(request)
        
        # Verifica cabeçalho Cache-Control
        if self._has_no_cache_header(request):
            return await call_next(request)
        
        # Obtém a chave de cache para esta requisição
        cache_key = self._generate_cache_key(request)
        
        # Verifica se há uma resposta em cache
        cache = get_cache()
        cached_response = await cache.get(cache_key)
        
        if cached_response:
            # Se encontrou no cache, retorna a resposta sem chamar o endpoint
            logger.debug(f"Cache hit: {request.method} {path}")
            return self._build_response_from_cache(cached_response)
        
        # Se não encontrou no cache, processa a requisição normalmente
        logger.debug(f"Cache miss: {request.method} {path}")
        response = await call_next(request)
        
        # Determina se a resposta deve ser armazenada em cache
        if self._should_cache_response(response):
            # Obtém o TTL adequado para esta rota
            ttl = self._get_ttl_for_path(path)
            
            # Armazena a resposta no cache
            await self._cache_response(cache_key, response, ttl)
        
        return response
    
    def _should_cache_path(self, path: str) -> bool:
        """
        Verifica se um caminho deve ser cacheado com base nas regras de inclusão/exclusão.
        
        Args:
            path: Caminho da URL a ser verificado.
            
        Returns:
            True se o caminho deve ser cacheado, False caso contrário.
        """
        # Verifica exclusões primeiro
        for pattern in self.exclude_paths:
            if self._match_path_pattern(pattern, path):
                return False
        
        # Se houver inclusões, verifica se o caminho está incluído
        if self.include_paths:
            return any(self._match_path_pattern(pattern, path) for pattern in self.include_paths)
        
        # Se não houver inclusões, todos os caminhos não excluídos são incluídos
        return True
    
    def _match_path_pattern(self, pattern: str, path: str) -> bool:
        """
        Verifica se um caminho corresponde a um padrão, com suporte a wildcards.
        
        Args:
            pattern: Padrão a ser verificado, pode conter '*' como wildcard.
            path: Caminho a ser verificado.
            
        Returns:
            True se o caminho corresponde ao padrão, False caso contrário.
        """
        # Converte o padrão com wildcard para regex
        regex_pattern = "^" + pattern.replace("*", ".*") + "$"
        return bool(re.match(regex_pattern, path))
    
    def _is_authenticated_request(self, request: Request) -> bool:
        """
        Verifica se a requisição está autenticada.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            True se a requisição está autenticada, False caso contrário.
        """
        # Verifica se há um token de autenticação no cabeçalho
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            return True
        
        # Verifica se há um token de acesso nos parâmetros de consulta
        token = request.query_params.get("access_token")
        if token:
            return True
        
        return False
    
    def _has_no_cache_header(self, request: Request) -> bool:
        """
        Verifica se a requisição tem cabeçalhos de controle de cache que desabilitam o cache.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            True se a requisição tem cabeçalhos para desabilitar cache, False caso contrário.
        """
        cache_control = request.headers.get("Cache-Control", "")
        return "no-cache" in cache_control or "no-store" in cache_control
    
    def _generate_cache_key(self, request: Request) -> str:
        """
        Gera uma chave de cache única para uma requisição.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            Uma chave de cache única para a requisição.
        """
        # Componentes principais da chave de cache
        method = request.method
        path = request.url.path
        
        # Filtra parâmetros de consulta, excluindo os especificados
        query_params = dict(request.query_params)
        for param in self.exclude_query_params:
            if param in query_params:
                del query_params[param]
        
        # Ordena os parâmetros para garantir consistência
        sorted_params = sorted(query_params.items())
        
        # Adiciona informações de autenticação se presente
        is_authenticated = self._is_authenticated_request(request)
        auth_info = "auth" if is_authenticated else "anon"
        
        # Constrói a chave base
        key_parts = [method, path, auth_info]
        
        # Adiciona parâmetros de consulta
        if sorted_params:
            params_str = "&".join(f"{k}={v}" for k, v in sorted_params)
            key_parts.append(params_str)
        
        # Junta todas as partes com um separador
        key = ":".join(key_parts)
        
        # Para chaves muito longas, usa um hash MD5
        if len(key) > 200:
            key_hash = hashlib.md5(key.encode()).hexdigest()
            return f"api:cache:{method}:{path}:{key_hash}"
        
        return f"api:cache:{key}"
    
    def _should_cache_response(self, response: Response) -> bool:
        """
        Verifica se uma resposta deve ser armazenada em cache.
        
        Args:
            response: Objeto Response a ser verificado.
            
        Returns:
            True se a resposta deve ser armazenada em cache, False caso contrário.
        """
        # Só armazena em cache respostas bem-sucedidas
        if response.status_code != 200:
            return False
        
        # Verifica cabeçalhos de controle de cache
        cache_control = response.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        # Verifica se a resposta é muito grande
        content_length = int(response.headers.get("Content-Length", "0"))
        if content_length > self.max_cache_size:
            return False
        
        return True
    
    def _get_ttl_for_path(self, path: str) -> int:
        """
        Obtém o TTL adequado para um caminho específico.
        
        Args:
            path: Caminho da URL.
            
        Returns:
            TTL em segundos para o caminho.
        """
        # Verifica se o caminho tem um TTL específico
        for pattern, ttl in self.ttl_map.items():
            if self._match_path_pattern(pattern, path):
                return ttl
        
        # Usa TTL padrão se não houver um específico
        return self.cache_ttl
    
    async def _cache_response(self, cache_key: str, response: Response, ttl: int) -> None:
        """
        Armazena uma resposta no cache.
        
        Args:
            cache_key: Chave de cache.
            response: Objeto Response a ser armazenado.
            ttl: Tempo de vida em segundos.
        """
        try:
            # Obtém o conteúdo da resposta
            content = b""
            if hasattr(response, "body"):
                content = response.body
            elif isinstance(response, JSONResponse):
                content = json.dumps(response.body).encode("utf-8")
            else:
                # Para outros tipos de resposta, serializa o conteúdo
                content = str(response.body).encode("utf-8")
            
            # Verifica o tamanho do conteúdo
            if len(content) > self.max_cache_size:
                return
            
            # Cria um objeto para armazenar no cache
            cache_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": content.decode("utf-8") if isinstance(content, bytes) else content,
                "content_type": response.headers.get("Content-Type", "application/json"),
            }
            
            # Armazena no cache
            cache = get_cache()
            await cache.set(cache_key, cache_data, ttl=ttl, tags=["api:response"])
        except Exception as e:
            logger.warning(f"Erro ao armazenar resposta no cache: {str(e)}")
    
    def _build_response_from_cache(self, cached_data: Dict) -> Response:
        """
        Constrói uma resposta a partir de dados em cache.
        
        Args:
            cached_data: Dados armazenados no cache.
            
        Returns:
            Um objeto Response reconstruído do cache.
        """
        status_code = cached_data.get("status_code", 200)
        headers = cached_data.get("headers", {})
        content = cached_data.get("content", "")
        content_type = cached_data.get("content_type", "application/json")
        
        # Se o conteúdo for uma string e o tipo for JSON, tenta converter para objeto JSON
        if isinstance(content, str) and "application/json" in content_type:
            try:
                content = json.loads(content)
                response = JSONResponse(
                    content=content,
                    status_code=status_code,
                    headers=headers
                )
            except json.JSONDecodeError:
                # Se falhar no parsing, trata como conteúdo de texto
                response = Response(
                    content=content,
                    status_code=status_code,
                    headers=headers,
                    media_type=content_type
                )
        else:
            # Para outros tipos de conteúdo
            response = Response(
                content=content,
                status_code=status_code,
                headers=headers,
                media_type=content_type
            )
        
        # Adiciona o cabeçalho de cache para indicar que a resposta veio do cache
        response.headers["X-Cache"] = "HIT"
        
        return response