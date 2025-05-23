"""
Neural Crypto Bot - Rate Limit Middleware

Este módulo implementa o middleware de rate limiting para proteger a API contra abusos.
Limita o número de requisições por cliente em determinados intervalos de tempo.
"""
import time
from typing import Dict, Tuple, Callable, Optional
import logging
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis
from prometheus_client import Counter

from src.common.utils.config import get_settings

# Configure o logger
logger = logging.getLogger(__name__)

# Métricas para Prometheus
RATE_LIMIT_HITS = Counter(
    "api_rate_limit_hits_total", 
    "Total number of rate limit hits",
    ["client_id", "endpoint"]
)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware para limitar a taxa de requisições por cliente.
    
    Implementa um algoritmo de "sliding window" para limitar o número de requisições
    que um cliente pode fazer em um determinado período de tempo. Suporta múltiplos
    limites (por segundo, minuto, hora) e pode ser configurado por rota.
    """

    def __init__(
        self,
        app,
        redis_client: Optional[redis.Redis] = None,
        rate_limits: Optional[Dict[str, Tuple[int, int]]] = None,
        exclude_paths: Optional[list] = None,
    ):
        """
        Inicializa o middleware de rate limiting.
        
        Args:
            app: A aplicação FastAPI.
            redis_client: Cliente Redis para armazenamento distribuído de contadores.
            rate_limits: Dicionário mapeando padrões de rota para tuplas (max_requests, window_seconds).
            exclude_paths: Lista de caminhos (endpoints) que não aplicam rate limiting.
        """
        super().__init__(app)
        self.settings = get_settings()
        self.redis_client = redis_client
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        
        # Define limites padrão se não fornecidos
        self.rate_limits = rate_limits or {
            # Limita todas as rotas a 60 requisições por minuto por padrão
            "*": (60, 60),
            # Limita rotas específicas que são mais pesadas
            "/api/v*/strategies/*/backtest": (10, 60),  # 10 backtests por minuto
            "/api/v*/analytics/performance": (20, 60),  # 20 análises de performance por minuto
            "/api/v*/analytics/reports": (5, 60),       # 5 relatórios por minuto
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Processa a requisição, verificando e aplicando limites de taxa.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            call_next: Função para processar o próximo middleware/endpoint.
            
        Returns:
            Objeto Response com a resposta da API ou erro 429 se o limite for excedido.
        """
        # Ignora rate limiting para caminhos excluídos
        if self._is_path_excluded(request.url.path):
            return await call_next(request)
        
        # Obtém o identificador do cliente (IP ou ID do usuário se autenticado)
        client_id = self._get_client_id(request)
        path = request.url.path
        
        # Determina o limite de taxa aplicável para esta rota
        max_requests, window_seconds = self._get_rate_limit_for_path(path)
        
        # Verifica se o cliente atingiu o limite
        if self._is_rate_limited(client_id, path, max_requests, window_seconds):
            # Incrementa métrica de Prometheus
            RATE_LIMIT_HITS.labels(client_id=client_id, endpoint=path).inc()
            
            # Log do evento
            logger.warning(
                f"Rate limit exceeded: client_id={client_id}, path={path}, "
                f"limit={max_requests}/{window_seconds}s"
            )
            
            # Retorna resposta 429 (Too Many Requests)
            retry_after = self._calculate_retry_after(client_id, path, window_seconds)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Muitas requisições. Por favor, tente novamente mais tarde.",
                    "limit": max_requests,
                    "window_seconds": window_seconds,
                },
                headers={"Retry-After": str(retry_after)},
            )
        
        # Incrementa o contador de requisições
        self._increment_request_count(client_id, path)
        
        # Processa a requisição normalmente
        response = await call_next(request)
        
        # Adiciona cabeçalhos relacionados ao rate limiting à resposta
        remaining = self._get_remaining_requests(client_id, path, max_requests)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(self._get_window_reset_time(window_seconds))
        
        return response
    
    def _is_path_excluded(self, path: str) -> bool:
        """
        Verifica se o caminho está excluído do rate limiting.
        
        Args:
            path: Caminho da URL a ser verificado.
            
        Returns:
            True se o caminho estiver excluído, False caso contrário.
        """
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    def _get_client_id(self, request: Request) -> str:
        """
        Obtém um identificador único para o cliente.
        
        Prioriza o ID do usuário autenticado, se disponível.
        Caso contrário, usa o endereço IP do cliente.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            Identificador único do cliente.
        """
        # Se o usuário estiver autenticado, usa o ID do usuário
        if hasattr(request.state, "user") and request.state.user:
            user_id = request.state.user.get("sub") or request.state.user.get("username")
            if user_id:
                return f"user:{user_id}"
        
        # Caso contrário, usa o IP do cliente
        client_ip = self._get_client_ip(request)
        return f"ip:{client_ip}"
    
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
    
    def _get_rate_limit_for_path(self, path: str) -> Tuple[int, int]:
        """
        Determina o limite de taxa aplicável para um caminho específico.
        
        Args:
            path: Caminho da URL a ser verificado.
            
        Returns:
            Tupla (max_requests, window_seconds) com o limite aplicável.
        """
        # Verifica as rotas específicas primeiro
        for route_pattern, limit in self.rate_limits.items():
            if route_pattern == "*":
                continue
                
            # Converte padrão de rota com curinga para regex simples
            pattern = route_pattern.replace("*", ".*")
            if re.match(pattern, path):
                return limit
        
        # Se nenhum padrão específico corresponder, usa o limite padrão
        return self.rate_limits.get("*", (60, 60))
    
    def _is_rate_limited(self, client_id: str, path: str, max_requests: int, window_seconds: int) -> bool:
        """
        Verifica se o cliente atingiu o limite de taxa para o caminho especificado.
        
        Args:
            client_id: Identificador do cliente.
            path: Caminho da requisição.
            max_requests: Número máximo de requisições permitidas.
            window_seconds: Janela de tempo em segundos.
            
        Returns:
            True se o cliente atingiu o limite, False caso contrário.
        """
        # Cria uma chave única para esta combinação de cliente e caminho
        key = f"ratelimit:{client_id}:{path}"
        
        # Se estiver usando Redis, verifica o contador distribuído
        if self.redis_client:
            current_count = self.redis_client.get(key)
            count = int(current_count or 0)
            return count >= max_requests
        
        # Implementação em memória simplificada para desenvolvimento
        # Em um ambiente de produção, use sempre o Redis para garantir consistência em múltiplas instâncias
        if not hasattr(self, "_rate_limit_counters"):
            self._rate_limit_counters = {}
        
        # Limpa contadores expirados
        self._clean_expired_counters()
        
        # Verifica se o contador existe e se está dentro do limite
        if key in self._rate_limit_counters:
            counters = self._rate_limit_counters[key]
            count = sum(1 for timestamp in counters if timestamp > time.time() - window_seconds)
            return count >= max_requests
        
        return False
    
    def _increment_request_count(self, client_id: str, path: str) -> None:
        """
        Incrementa o contador de requisições para o cliente e caminho especificados.
        
        Args:
            client_id: Identificador do cliente.
            path: Caminho da requisição.
        """
        # Cria uma chave única para esta combinação de cliente e caminho
        key = f"ratelimit:{client_id}:{path}"
        
        # Se estiver usando Redis, incrementa o contador distribuído
        if self.redis_client:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, 60 * 60)  # Define TTL de 1 hora para evitar vazamento de memória
            pipe.execute()
            return
        
        # Implementação em memória simplificada para desenvolvimento
        if not hasattr(self, "_rate_limit_counters"):
            self._rate_limit_counters = {}
        
        # Adiciona o timestamp atual ao contador
        if key not in self._rate_limit_counters:
            self._rate_limit_counters[key] = []
        
        self._rate_limit_counters[key].append(time.time())
    
    def _clean_expired_counters(self) -> None:
        """
        Limpa contadores expirados da implementação em memória.
        Chamado periodicamente para evitar vazamento de memória.
        """
        if not hasattr(self, "_rate_limit_counters"):
            return
        
        now = time.time()
        max_window = 60 * 60  # 1 hora, janela máxima que consideramos
        
        for key, counters in list(self._rate_limit_counters.items()):
            # Remove timestamps mais antigos que a janela máxima
            self._rate_limit_counters[key] = [ts for ts in counters if ts > now - max_window]
            
            # Remove a chave completamente se não houver mais contadores
            if not self._rate_limit_counters[key]:
                del self._rate_limit_counters[key]
    
    def _get_remaining_requests(self, client_id: str, path: str, max_requests: int) -> int:
        """
        Obtém o número de requisições restantes para o cliente e caminho especificados.
        
        Args:
            client_id: Identificador do cliente.
            path: Caminho da requisição.
            max_requests: Número máximo de requisições permitidas.
            
        Returns:
            Número de requisições restantes.
        """
        # Cria uma chave única para esta combinação de cliente e caminho
        key = f"ratelimit:{client_id}:{path}"
        
        # Se estiver usando Redis, verifica o contador distribuído
        if self.redis_client:
            current_count = self.redis_client.get(key)
            count = int(current_count or 0)
            return max(0, max_requests - count)
        
        # Implementação em memória
        if not hasattr(self, "_rate_limit_counters") or key not in self._rate_limit_counters:
            return max_requests
        
        window_seconds = 60  # Assume 60 segundos como janela padrão
        counters = self._rate_limit_counters[key]
        count = sum(1 for timestamp in counters if timestamp > time.time() - window_seconds)
        
        return max(0, max_requests - count)
    
    def _get_window_reset_time(self, window_seconds: int) -> int:
        """
        Calcula quando a janela atual será redefinida (em segundos).
        
        Args:
            window_seconds: Tamanho da janela em segundos.
            
        Returns:
            Tempo em segundos até a próxima redefinição da janela.
        """
        # Na implementação de sliding window, a janela desliza continuamente
        # Então o reset será aproximadamente daqui a window_seconds
        return int(time.time()) + window_seconds
    
    def _calculate_retry_after(self, client_id: str, path: str, window_seconds: int) -> int:
        """
        Calcula o tempo recomendado para o cliente esperar antes de tentar novamente.
        
        Args:
            client_id: Identificador do cliente.
            path: Caminho da requisição.
            window_seconds: Tamanho da janela em segundos.
            
        Returns:
            Tempo recomendado em segundos.
        """
        # Para uma implementação mais sofisticada, poderíamos calcular exatamente
        # quando as requisições mais antigas vão expirar da janela
        # Por enquanto, usamos uma abordagem simplificada
        
        # Recomenda esperar metade da janela antes de tentar novamente
        return max(1, window_seconds // 2)