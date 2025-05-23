"""
Gerenciador de cache para o Neural Crypto Bot.

Este módulo implementa funcionalidades de cache para melhorar a performance
e reduzir carga no banco de dados e serviços externos.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, Union, TypeVar, cast, List, Tuple
import json
import hashlib
import time
import inspect
import asyncio
import logging
from datetime import datetime, timedelta
import redis
import redis.asyncio as aioredis
from functools import wraps

from ...utils.config import get_settings

# Logger
logger = logging.getLogger(__name__)

# Tipo genérico para valor de retorno
T = TypeVar('T')

class CacheManager(ABC):
    """Interface base para implementações de cache."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Recupera um valor do cache.
        
        Args:
            key: A chave do valor a ser recuperado.
            
        Returns:
            O valor armazenado ou None se não encontrado.
        """
        pass
    
    @abstractmethod
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Armazena um valor no cache.
        
        Args:
            key: A chave para armazenar o valor.
            value: O valor a ser armazenado.
            ttl: Tempo de vida em segundos ou None para usar o padrão.
            tags: Tags associadas à entrada para invalidação agrupada.
            
        Returns:
            True se o valor foi armazenado com sucesso, False caso contrário.
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Remove um valor do cache.
        
        Args:
            key: A chave do valor a ser removido.
            
        Returns:
            True se o valor foi removido com sucesso, False caso contrário.
        """
        pass
    
    @abstractmethod
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalida todas as entradas de cache associadas a uma tag.
        
        Args:
            tag: A tag das entradas a serem invalidadas.
            
        Returns:
            Número de entradas invalidadas.
        """
        pass
    
    @abstractmethod
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalida todas as entradas de cache que correspondem a um padrão.
        
        Args:
            pattern: Padrão de correspondência para as chaves.
            
        Returns:
            Número de entradas invalidadas.
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Limpa todo o cache.
        
        Returns:
            True se o cache foi limpo com sucesso, False caso contrário.
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do cache.
        
        Returns:
            Dicionário com estatísticas como hits, misses, tamanho, etc.
        """
        pass


class RedisCache(CacheManager):
    """Implementação de cache usando Redis."""
    
    def __init__(
        self, 
        redis_client: Optional[aioredis.Redis] = None,
        namespace: str = "ncb",
        default_ttl: int = 3600,  # 1 hora
    ):
        """
        Inicializa o cache Redis.
        
        Args:
            redis_client: Cliente Redis assíncrono.
            namespace: Prefixo a ser aplicado a todas as chaves.
            default_ttl: Tempo de vida padrão para entradas em segundos.
        """
        self.settings = get_settings()
        
        if redis_client:
            self.redis = redis_client
        else:
            self.redis = aioredis.from_url(
                self.settings.REDIS_URL,
                decode_responses=True,
                username=self.settings.REDIS_USERNAME,
                password=self.settings.REDIS_PASSWORD,
            )
        
        # Prefixo para todas as chaves - útil para isolar diferentes ambientes
        self.namespace = namespace
        
        # TTL padrão para entradas
        self.default_ttl = default_ttl
        
        # Métricas de cache
        self._hits = 0
        self._misses = 0
        self._last_reset = time.time()
    
    def _get_namespaced_key(self, key: str) -> str:
        """
        Aplica o namespace à chave.
        
        Args:
            key: A chave original.
            
        Returns:
            A chave com o namespace aplicado.
        """
        return f"{self.namespace}:{key}"
    
    def _get_tag_key(self, tag: str) -> str:
        """
        Obtém a chave que armazena as chaves associadas a uma tag.
        
        Args:
            tag: A tag.
            
        Returns:
            A chave que armazena as chaves associadas à tag.
        """
        return f"{self.namespace}:tag:{tag}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Recupera um valor do cache.
        
        Args:
            key: A chave do valor a ser recuperado.
            
        Returns:
            O valor armazenado ou None se não encontrado.
        """
        namespaced_key = self._get_namespaced_key(key)
        
        try:
            value = await self.redis.get(namespaced_key)
            
            if value is None:
                self._misses += 1
                return None
            
            self._hits += 1
            return json.loads(value)
        except Exception as e:
            logger.warning(f"Erro ao recuperar do cache: {str(e)}")
            self._misses += 1
            return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Armazena um valor no cache.
        
        Args:
            key: A chave para armazenar o valor.
            value: O valor a ser armazenado.
            ttl: Tempo de vida em segundos ou None para usar o padrão.
            tags: Tags associadas à entrada para invalidação agrupada.
            
        Returns:
            True se o valor foi armazenado com sucesso, False caso contrário.
        """
        namespaced_key = self._get_namespaced_key(key)
        ttl_value = ttl if ttl is not None else self.default_ttl
        
        try:
            serialized_value = json.dumps(value)
            
            # Armazena o valor com TTL
            await self.redis.setex(
                name=namespaced_key,
                time=ttl_value,
                value=serialized_value
            )
            
            # Associa a chave às tags fornecidas
            if tags:
                async with self.redis.pipeline() as pipe:
                    for tag in tags:
                        tag_key = self._get_tag_key(tag)
                        await pipe.sadd(tag_key, namespaced_key)
                        # Define TTL para o conjunto de tags também
                        await pipe.expire(tag_key, ttl_value)
                    await pipe.execute()
            
            return True
        except Exception as e:
            logger.warning(f"Erro ao armazenar no cache: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Remove um valor do cache.
        
        Args:
            key: A chave do valor a ser removido.
            
        Returns:
            True se o valor foi removido com sucesso, False caso contrário.
        """
        namespaced_key = self._get_namespaced_key(key)
        
        try:
            await self.redis.delete(namespaced_key)
            return True
        except Exception as e:
            logger.warning(f"Erro ao remover do cache: {str(e)}")
            return False
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalida todas as entradas de cache associadas a uma tag.
        
        Args:
            tag: A tag das entradas a serem invalidadas.
            
        Returns:
            Número de entradas invalidadas.
        """
        tag_key = self._get_tag_key(tag)
        
        try:
            # Obtém todas as chaves associadas à tag
            keys = await self.redis.smembers(tag_key)
            
            if not keys:
                return 0
            
            # Remove todas as chaves
            count = await self.redis.delete(*keys)
            
            # Remove a própria chave da tag
            await self.redis.delete(tag_key)
            
            return count
        except Exception as e:
            logger.warning(f"Erro ao invalidar por tag: {str(e)}")
            return 0
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalida todas as entradas de cache que correspondem a um padrão.
        
        Args:
            pattern: Padrão de correspondência para as chaves.
            
        Returns:
            Número de entradas invalidadas.
        """
        namespaced_pattern = self._get_namespaced_key(pattern)
        
        try:
            # Usa o comando SCAN para encontrar chaves que correspondem ao padrão
            cursor = 0
            count = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=namespaced_pattern, count=100)
                
                if keys:
                    deleted = await self.redis.delete(*keys)
                    count += deleted
                
                if cursor == 0:
                    break
            
            return count
        except Exception as e:
            logger.warning(f"Erro ao invalidar por padrão: {str(e)}")
            return 0
    
    async def clear(self) -> bool:
        """
        Limpa todo o cache.
        
        Returns:
            True se o cache foi limpo com sucesso, False caso contrário.
        """
        try:
            # Obtém todas as chaves no namespace
            pattern = f"{self.namespace}:*"
            cursor = 0
            all_keys = []
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                all_keys.extend(keys)
                
                if cursor == 0:
                    break
            
            if all_keys:
                await self.redis.delete(*all_keys)
            
            # Reseta métricas
            self._hits = 0
            self._misses = 0
            self._last_reset = time.time()
            
            return True
        except Exception as e:
            logger.warning(f"Erro ao limpar cache: {str(e)}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do cache.
        
        Returns:
            Dicionário com estatísticas como hits, misses, tamanho, etc.
        """
        try:
            # Conta quantas chaves existem no namespace
            pattern = f"{self.namespace}:*"
            cursor = 0
            count = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                count += len(keys)
                
                if cursor == 0:
                    break
            
            # Calcula estatísticas de acerto/erro
            total_requests = self._hits + self._misses
            hit_ratio = self._hits / max(1, total_requests) * 100
            
            # Tempo desde o último reset
            uptime = time.time() - self._last_reset
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total_requests,
                "hit_ratio": round(hit_ratio, 2),
                "size": count,
                "uptime_seconds": round(uptime, 2),
            }
        except Exception as e:
            logger.warning(f"Erro ao obter estatísticas do cache: {str(e)}")
            return {
                "error": str(e),
                "hits": self._hits,
                "misses": self._misses,
            }


class InMemoryCache(CacheManager):
    """Implementação de cache em memória para desenvolvimento e testes."""
    
    def __init__(self, default_ttl: int = 3600):
        """
        Inicializa o cache em memória.
        
        Args:
            default_ttl: Tempo de vida padrão para entradas em segundos.
        """
        self.cache: Dict[str, Tuple[Any, float, List[str]]] = {}  # key -> (value, expire_at, tags)
        self.tag_index: Dict[str, List[str]] = {}  # tag -> [keys]
        self.default_ttl = default_ttl
        
        # Métricas de cache
        self._hits = 0
        self._misses = 0
        self._last_reset = time.time()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Recupera um valor do cache.
        
        Args:
            key: A chave do valor a ser recuperado.
            
        Returns:
            O valor armazenado ou None se não encontrado.
        """
        # Limpa entradas expiradas sempre que acessar o cache
        self._clean_expired()
        
        if key not in self.cache:
            self._misses += 1
            return None
        
        value, expire_at, _ = self.cache[key]
        
        # Verifica se a entrada expirou
        if expire_at < time.time():
            del self.cache[key]
            self._misses += 1
            return None
        
        self._hits += 1
        return value
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Armazena um valor no cache.
        
        Args:
            key: A chave para armazenar o valor.
            value: O valor a ser armazenado.
            ttl: Tempo de vida em segundos ou None para usar o padrão.
            tags: Tags associadas à entrada para invalidação agrupada.
            
        Returns:
            True se o valor foi armazenado com sucesso, False caso contrário.
        """
        ttl_value = ttl if ttl is not None else self.default_ttl
        expire_at = time.time() + ttl_value
        tag_list = tags or []
        
        # Armazena o valor
        self.cache[key] = (value, expire_at, tag_list)
        
        # Associa a chave às tags
        for tag in tag_list:
            if tag not in self.tag_index:
                self.tag_index[tag] = []
            
            if key not in self.tag_index[tag]:
                self.tag_index[tag].append(key)
        
        return True
    
    async def delete(self, key: str) -> bool:
        """
        Remove um valor do cache.
        
        Args:
            key: A chave do valor a ser removido.
            
        Returns:
            True se o valor foi removido com sucesso, False caso contrário.
        """
        if key not in self.cache:
            return False
        
        # Remove a chave das tags associadas
        _, _, tags = self.cache[key]
        for tag in tags:
            if tag in self.tag_index and key in self.tag_index[tag]:
                self.tag_index[tag].remove(key)
        
        # Remove a chave do cache
        del self.cache[key]
        return True
    
    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalida todas as entradas de cache associadas a uma tag.
        
        Args:
            tag: A tag das entradas a serem invalidadas.
            
        Returns:
            Número de entradas invalidadas.
        """
        if tag not in self.tag_index:
            return 0
        
        # Obtém todas as chaves associadas à tag
        keys = self.tag_index[tag].copy()
        count = 0
        
        # Remove cada chave
        for key in keys:
            if key in self.cache:
                del self.cache[key]
                count += 1
        
        # Limpa a tag
        del self.tag_index[tag]
        
        return count
    
    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Invalida todas as entradas de cache que correspondem a um padrão.
        
        Args:
            pattern: Padrão de correspondência para as chaves.
            
        Returns:
            Número de entradas invalidadas.
        """
        import re
        
        pattern_regex = re.compile(pattern.replace("*", ".*"))
        keys_to_delete = [key for key in self.cache.keys() if pattern_regex.match(key)]
        
        count = 0
        for key in keys_to_delete:
            await self.delete(key)
            count += 1
        
        return count
    
    async def clear(self) -> bool:
        """
        Limpa todo o cache.
        
        Returns:
            True se o cache foi limpo com sucesso, False caso contrário.
        """
        self.cache.clear()
        self.tag_index.clear()
        
        # Reseta métricas
        self._hits = 0
        self._misses = 0
        self._last_reset = time.time()
        
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Obtém estatísticas do cache.
        
        Returns:
            Dicionário com estatísticas como hits, misses, tamanho, etc.
        """
        # Limpa entradas expiradas para obter contagem precisa
        self._clean_expired()
        
        total_requests = self._hits + self._misses
        hit_ratio = self._hits / max(1, total_requests) * 100
        uptime = time.time() - self._last_reset
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_ratio": round(hit_ratio, 2),
            "size": len(self.cache),
            "uptime_seconds": round(uptime, 2),
        }
    
    def _clean_expired(self) -> None:
        """
        Remove entradas expiradas do cache.
        """
        now = time.time()
        expired_keys = [key for key, (_, expire_at, _) in self.cache.items() if expire_at < now]
        
        for key in expired_keys:
            _, _, tags = self.cache[key]
            
            # Remove a chave das tags associadas
            for tag in tags:
                if tag in self.tag_index and key in self.tag_index[tag]:
                    self.tag_index[tag].remove(key)
            
            del self.cache[key]


# Instância global do cache, inicializada sob demanda
_cache_instance: Optional[CacheManager] = None

def get_cache() -> CacheManager:
    """
    Obtém uma instância do cache.
    
    Returns:
        Uma instância de CacheManager.
    """
    global _cache_instance
    
    if _cache_instance is None:
        settings = get_settings()
        
        # Usa Redis em produção e cache em memória para desenvolvimento
        if settings.ENVIRONMENT in ("production", "staging"):
            _cache_instance = RedisCache(
                namespace=f"ncb:{settings.ENVIRONMENT}",
                default_ttl=int(settings.CACHE_DEFAULT_TTL or 3600)
            )
        else:
            _cache_instance = InMemoryCache(
                default_ttl=int(settings.CACHE_DEFAULT_TTL or 3600)
            )
    
    return _cache_instance


def generate_cache_key(
    prefix: str, 
    *args, 
    **kwargs
) -> str:
    """
    Gera uma chave de cache baseada nos argumentos.
    
    Args:
        prefix: Prefixo para a chave.
        *args: Argumentos posicionais para incorporar na chave.
        **kwargs: Argumentos nomeados para incorporar na chave.
        
    Returns:
        Uma chave de cache única.
    """
    # Gera uma representação consistente dos argumentos
    key_parts = [prefix]
    
    if args:
        for arg in args:
            # Trata tipos complexos usando hash do JSON
            if isinstance(arg, (dict, list, tuple, set)):
                key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
            else:
                key_parts.append(str(arg))
    
    if kwargs:
        # Ordena os argumentos nomeados para garantir consistência
        sorted_kwargs = sorted(kwargs.items())
        for k, v in sorted_kwargs:
            # Trata tipos complexos usando hash do JSON
            if isinstance(v, (dict, list, tuple, set)):
                key_parts.append(f"{k}:{hashlib.md5(json.dumps(v, sort_keys=True).encode()).hexdigest()}")
            else:
                key_parts.append(f"{k}:{v}")
    
    # Junta todas as partes com um separador
    key = ":".join(key_parts)
    
    # Se a chave for muito longa, usa um hash dela para evitar problemas
    if len(key) > 200:
        return f"{prefix}:{hashlib.md5(key.encode()).hexdigest()}"
    
    return key


def cached(
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    tags: Optional[List[str]] = None,
    exclude_args: Optional[List[str]] = None,
):
    """
    Decorador para cachear resultados de funções.
    
    Args:
        ttl: Tempo de vida em segundos ou None para usar o padrão.
        key_prefix: Prefixo para a chave de cache.
        tags: Tags a associar à entrada de cache.
        exclude_args: Lista de nomes de argumentos a excluir da chave de cache.
        
    Returns:
        Um decorador para funções.
    """
    def decorator(func: Callable[..., Any]):
        # Obtém os argumentos da função para extraí-los posteriormente
        sig = inspect.signature(func)
        
        # Determina o prefixo da chave
        prefix = key_prefix or f"{func.__module__}.{func.__qualname__}"
        
        # Define a lista de argumentos a excluir
        excluded_args = exclude_args or []
        
        # Constrói um wrapper assíncrono se a função for assíncrona, ou um wrapper normal caso contrário
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Obtém o cache
                cache = get_cache()
                
                # Mapeia argumentos posicionais para nomes
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Filtra argumentos excluídos
                filtered_kwargs = {k: v for k, v in bound_args.arguments.items() if k not in excluded_args}
                
                # Gera a chave de cache
                cache_key = generate_cache_key(prefix, **filtered_kwargs)
                
                # Tenta recuperar do cache
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Se não estiver no cache, executa a função
                result = await func(*args, **kwargs)
                
                # Armazena o resultado no cache
                await cache.set(cache_key, result, ttl=ttl, tags=tags)
                
                return result
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Obtém o cache
                cache = get_cache()
                
                # Mapeia argumentos posicionais para nomes
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                # Filtra argumentos excluídos
                filtered_kwargs = {k: v for k, v in bound_args.arguments.items() if k not in excluded_args}
                
                # Gera a chave de cache
                cache_key = generate_cache_key(prefix, **filtered_kwargs)
                
                # Tenta recuperar do cache
                # Como o cache é assíncrono, precisamos executar de forma síncrona
                cached_value = asyncio.run(cache.get(cache_key))
                if cached_value is not None:
                    return cached_value
                
                # Se não estiver no cache, executa a função
                result = func(*args, **kwargs)
                
                # Armazena o resultado no cache
                asyncio.run(cache.set(cache_key, result, ttl=ttl, tags=tags))
                
                return result
            
            return sync_wrapper
    
    return decorator