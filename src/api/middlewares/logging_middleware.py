"""
Neural Crypto Bot - Logging Middleware

Este módulo implementa o middleware de logging para a API.
Responsável por registrar informações sobre requisições e respostas para monitoramento e debug.
"""
import json
import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Histogram
from starlette.middleware.base import BaseHTTPMiddleware

from common.utils.config import get_settings

# Configure o logger
logger = logging.getLogger(__name__)

# Métricas para Prometheus
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "API Request latency in seconds",
    ["method", "endpoint"]
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware para logging de requisições e respostas da API.
    
    Esta classe registra informações detalhadas sobre cada requisição,
    incluindo método, caminho, headers relevantes, tempo de processamento,
    e detalhes da resposta para fins de auditoria e diagnóstico.
    """

    def __init__(self, app):
        """
        Inicializa o middleware de logging.
        
        Args:
            app: A aplicação FastAPI.
        """
        super().__init__(app)
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa a requisição, registra logs e métricas.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            call_next: Função para processar o próximo middleware/endpoint.
            
        Returns:
            Objeto Response com a resposta da API.
        """
        # Gera um ID único para a requisição para rastreabilidade
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Registra o início da requisição
        start_time = time.time()
        
        # Prepara os dados de log iniciais
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        
        # Registra log de requisição recebida
        logger.info(
            f"Request started: id={request_id}, method={method}, path={path}, "
            f"client_ip={client_ip}, user_agent={user_agent}"
        )
        
        # Processa a requisição e captura a resposta
        try:
            response = await call_next(request)
            
            # Calcula o tempo de processamento
            process_time = time.time() - start_time
            
            # Atualiza métricas de latência
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(process_time)
            
            # Registra log de resposta
            status_code = response.status_code
            logger.info(
                f"Request completed: id={request_id}, method={method}, path={path}, "
                f"status_code={status_code}, duration={process_time:.4f}s"
            )
            
            # Adiciona cabeçalhos de rastreabilidade à resposta
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            # Calcula o tempo de processamento até o erro
            process_time = time.time() - start_time
            
            # Registra log de erro
            logger.error(
                f"Request failed: id={request_id}, method={method}, path={path}, "
                f"error={str(e)}, duration={process_time:.4f}s",
                exc_info=True
            )
            
            # Relança a exceção para ser tratada pelo handler global
            raise
            
    def _get_client_ip(self, request: Request) -> str:
        """
        Extrai o endereço IP real do cliente, considerando proxies.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            Endereço IP do cliente.
        """
        # Verifica headers comuns que podem conter o IP real em caso de proxies
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for:
            # O primeiro IP na lista é o do cliente original
            return x_forwarded_for.split(",")[0].strip()
            
        # Se não houver X-Forwarded-For, tenta outros headers
        for header in ["X-Real-IP", "CF-Connecting-IP"]:
            value = request.headers.get(header)
            if value:
                return value
                
        # Se não encontrar em nenhum header, usa o IP da conexão direta
        return request.client.host if request.client else "unknown"
    
    async def _log_request_body(self, request: Request) -> dict:
        """
        Extrai e formata o corpo da requisição para logging.
        Métodos seguros como GET e HEAD não têm o corpo registrado.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            Dicionário com dados do corpo ou indicação de que não há corpo.
        """
        if request.method in ["GET", "HEAD", "OPTIONS"]:
            return {"body": "[No body]"}
            
        try:
            # Tenta ler o corpo como JSON
            body = await request.json()
            
            # Censura campos sensíveis se o nível de logging não for DEBUG
            if self.settings.LOG_LEVEL != "DEBUG" and isinstance(body, dict):
                sensitive_fields = ["password", "token", "secret", "key", "api_key", "apikey"]
                for field in sensitive_fields:
                    if field in body:
                        body[field] = "***REDACTED***"
            
            return {"body": json.dumps(body)}
        except:
            # Se não conseguir ler como JSON, obtém o corpo como texto
            try:
                body = await request.body()
                return {"body": f"[Binary data, length: {len(body)} bytes]"}
            except:
                return {"body": "[Failed to read body]"}