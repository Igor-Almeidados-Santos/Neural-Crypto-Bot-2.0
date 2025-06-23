"""
Balanceador de carga para exchanges de criptomoedas.

Este módulo implementa balanceamento de carga inteligente para distribuir
requisições entre múltiplas exchanges e instâncias, otimizando performance
e garantindo alta disponibilidade.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import random
from collections import defaultdict, deque

from src.data_collection.adapters.exchange_adapter_interface import ExchangeAdapterInterface

logger = logging.getLogger(__name__)


class BalancingStrategy(Enum):
    """Estratégias de balanceamento de carga."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    HEALTH_AWARE = "health_aware"
    ADAPTIVE = "adaptive"
    RANDOM = "random"


class InstanceState(Enum):
    """Estados possíveis de uma instância."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class InstanceMetrics:
    """Métricas de uma instância de exchange."""
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    last_response_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    error_rate: float = 0.0
    
    # Janelas deslizantes para métricas
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count_window: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def update_request_metrics(self, response_time: float, success: bool):
        """Atualiza métricas de requisição."""
        self.total_requests += 1
        self.last_request_time = datetime.utcnow()
        
        if success:
            self.successful_requests += 1
            self.last_response_time = datetime.utcnow()
            self.response_times.append(response_time)
            
            # Calcula média de tempo de resposta
            if self.response_times:
                self.avg_response_time = statistics.mean(self.response_times)
        else:
            self.failed_requests += 1
            self.last_error_time = datetime.utcnow()
            self.error_count_window.append(datetime.utcnow())
        
        # Calcula taxa de erro
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
    
    def get_recent_error_rate(self, window_minutes: int = 5) -> float:
        """Calcula taxa de erro recente."""
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_errors = sum(1 for error_time in self.error_count_window if error_time > cutoff)
        
        # Estima requisições recentes baseado na taxa atual
        if self.total_requests > 0:
            time_since_start = (datetime.utcnow() - (self.last_request_time or datetime.utcnow())).total_seconds() / 60
            if time_since_start > 0:
                recent_requests = max(1, int(self.total_requests * (window_minutes / max(1, time_since_start))))
                return recent_errors / recent_requests
        
        return 0.0


@dataclass
class ExchangeInstance:
    """Instância de uma exchange com suas configurações e métricas."""
    id: str
    exchange: ExchangeAdapterInterface
    weight: float = 1.0
    priority: int = 1
    state: InstanceState = InstanceState.HEALTHY
    metrics: InstanceMetrics = field(default_factory=InstanceMetrics)
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Configurações de saúde
    max_connections: int = 100
    max_error_rate: float = 0.1
    max_response_time: float = 5.0
    health_check_interval: int = 30
    
    # Status de saúde
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3
    
    @property
    def is_available(self) -> bool:
        """Verifica se a instância está disponível para receber requisições."""
        return self.state in [InstanceState.HEALTHY, InstanceState.DEGRADED]
    
    @property
    def load_score(self) -> float:
        """Calcula score de carga (menor é melhor)."""
        if not self.is_available:
            return float('inf')
        
        # Combina múltiples fatores
        connection_factor = self.metrics.active_connections / max(1, self.max_connections)
        response_time_factor = self.metrics.avg_response_time / max(0.1, self.max_response_time)
        error_rate_factor = self.metrics.error_rate / max(0.01, self.max_error_rate)
        
        return (connection_factor + response_time_factor + error_rate_factor) / self.weight
    
    async def health_check(self) -> bool:
        """Executa verificação de saúde da instância."""
        try:
            start_time = time.time()
            
            # Verifica se a exchange está ativa
            if hasattr(self.exchange, 'ping'):
                await self.exchange.ping()
            else:
                # Fallback: tenta buscar trading pairs
                await self.exchange.fetch_trading_pairs()
            
            response_time = time.time() - start_time
            
            # Atualiza métricas
            self.metrics.update_request_metrics(response_time, True)
            self.last_health_check = datetime.utcnow()
            self.consecutive_failures = 0
            
            # Determina estado baseado nas métricas
            if (self.metrics.error_rate > self.max_error_rate or 
                self.metrics.avg_response_time > self.max_response_time):
                self.state = InstanceState.DEGRADED
            else:
                self.state = InstanceState.HEALTHY
            
            return True
            
        except Exception as e:
            logger.warning(f"Health check falhou para {self.id}: {str(e)}")
            
            self.metrics.update_request_metrics(0, False)
            self.consecutive_failures += 1
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.state = InstanceState.UNHEALTHY
            else:
                self.state = InstanceState.DEGRADED
            
            return False


class CircuitBreaker:
    """
    Circuit breaker para proteção contra faltas de instâncias.
    
    Implementa padrão Circuit Breaker para evitar sobrecarga
    de instâncias com problemas.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Executa função protegida pelo circuit breaker."""
        if self.state == 'OPEN':
            if (datetime.utcnow() - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Chamado quando operação é bem-sucedida."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Chamado quando operação falha."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class ExchangeLoadBalancer:
    """
    Balanceador de carga principal para exchanges.
    
    Gerencia múltiplas instâncias de exchanges, distribui requisições
    de forma inteligente e monitora saúde das instâncias.
    """
    
    def __init__(
        self,
        strategy: BalancingStrategy = BalancingStrategy.ADAPTIVE,
        health_check_interval: int = 30,
        enable_circuit_breaker: bool = True
    ):
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Instâncias gerenciadas
        self._instances: Dict[str, ExchangeInstance] = {}
        self._instance_groups: Dict[str, List[str]] = defaultdict(list)
        
        # Circuit breakers por instância
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Estado do balanceador
        self._current_index = 0  # Para round robin
        self._last_health_check = datetime.utcnow()
        
        # Tasks de controle
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Estatísticas
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'instance_selections': defaultdict(int)
        }
    
    async def start(self) -> None:
        """Inicia o balanceador de carga."""
        if self._running:
            return
        
        self._running = True
        
        # Inicia health checks periódicos
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Load balancer iniciado")
    
    async def stop(self) -> None:
        """Para o balanceador de carga."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancela health checks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Load balancer parado")
    
    def add_instance(
        self,
        instance_id: str,
        exchange: ExchangeAdapterInterface,
        weight: float = 1.0,
        priority: int = 1,
        group: Optional[str] = None,
        **config
    ) -> None:
        """
        Adiciona uma instância ao balanceador.
        
        Args:
            instance_id: ID único da instância
            exchange: Adaptador da exchange
            weight: Peso da instância para balanceamento
            priority: Prioridade da instância
            group: Grupo da instância (opcional)
            **config: Configurações adicionais
        """
        instance = ExchangeInstance(
            id=instance_id,
            exchange=exchange,
            weight=weight,
            priority=priority,
            config=config
        )
        
        self._instances[instance_id] = instance
        
        if group:
            self._instance_groups[group].append(instance_id)
        
        # Cria circuit breaker se habilitado
        if self.enable_circuit_breaker:
            self._circuit_breakers[instance_id] = CircuitBreaker()
        
        logger.info(f"Instância adicionada: {instance_id} (peso: {weight}, prioridade: {priority})")
    
    def remove_instance(self, instance_id: str) -> None:
        """Remove uma instância do balanceador."""
        if instance_id in self._instances:
            instance = self._instances[instance_id]
            instance.state = InstanceState.OFFLINE
            
            del self._instances[instance_id]
            
            # Remove dos grupos
            for group_instances in self._instance_groups.values():
                if instance_id in group_instances:
                    group_instances.remove(instance_id)
            
            # Remove circuit breaker
            if instance_id in self._circuit_breakers:
                del self._circuit_breakers[instance_id]
            
            logger.info(f"Instância removida: {instance_id}")
    
    async def get_instance(
        self,
        group: Optional[str] = None,
        exclude: Optional[Set[str]] = None
    ) -> Optional[ExchangeInstance]:
        """
        Obtém uma instância baseada na estratégia de balanceamento.
        
        Args:
            group: Grupo específico de instâncias (opcional)
            exclude: IDs de instâncias a excluir (opcional)
            
        Returns:
            Instância selecionada ou None se nenhuma disponível
        """
        available_instances = self._get_available_instances(group, exclude)
        
        if not available_instances:
            logger.warning("Nenhuma instância disponível")
            return None
        
        # Seleciona instância baseado na estratégia
        if self.strategy == BalancingStrategy.ROUND_ROBIN:
            instance = self._select_round_robin(available_instances)
        elif self.strategy == BalancingStrategy.WEIGHTED_ROUND_ROBIN:
            instance = self._select_weighted_round_robin(available_instances)
        elif self.strategy == BalancingStrategy.LEAST_CONNECTIONS:
            instance = self._select_least_connections(available_instances)
        elif self.strategy == BalancingStrategy.LEAST_RESPONSE_TIME:
            instance = self._select_least_response_time(available_instances)
        elif self.strategy == BalancingStrategy.HEALTH_AWARE:
            instance = self._select_health_aware(available_instances)
        elif self.strategy == BalancingStrategy.ADAPTIVE:
            instance = self._select_adaptive(available_instances)
        elif self.strategy == BalancingStrategy.RANDOM:
            instance = self._select_random(available_instances)
        else:
            instance = available_instances[0]  # Fallback
        
        if instance:
            instance.metrics.active_connections += 1
            self._stats['instance_selections'][instance.id] += 1
        
        return instance
    
    async def execute_request(
        self,
        request_func: Callable[[ExchangeAdapterInterface], Awaitable[Any]],
        group: Optional[str] = None,
        max_retries: int = 3,
        exclude_failed: bool = True
    ) -> Any:
        """
        Executa uma requisição usando balanceamento de carga.
        
        Args:
            request_func: Função que executa a requisição
            group: Grupo de instâncias a usar (opcional)
            max_retries: Número máximo de tentativas
            exclude_failed: Se deve excluir instâncias que falharam
            
        Returns:
            Resultado da requisição
        """
        excluded_instances = set()
        last_exception = None
        
        for attempt in range(max_retries + 1):
            instance = await self.get_instance(
                group=group,
                exclude=excluded_instances if exclude_failed else None
            )
            
            if not instance:
                break
            
            start_time = time.time()
            success = False
            
            try:
                self._stats['total_requests'] += 1
                
                # Executa com circuit breaker se habilitado
                if self.enable_circuit_breaker and instance.id in self._circuit_breakers:
                    cb = self._circuit_breakers[instance.id]
                    result = await cb.call(request_func, instance.exchange)
                else:
                    result = await request_func(instance.exchange)
                
                success = True
                response_time = time.time() - start_time
                
                # Atualiza métricas
                instance.metrics.update_request_metrics(response_time, True)
                self._stats['successful_requests'] += 1
                
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                instance.metrics.update_request_metrics(response_time, False)
                self._stats['failed_requests'] += 1
                
                if exclude_failed:
                    excluded_instances.add(instance.id)
                
                last_exception = e
                logger.warning(f"Requisição falhou na instância {instance.id}: {str(e)}")
                
            finally:
                instance.metrics.active_connections = max(0, instance.metrics.active_connections - 1)
        
        # Se chegou aqui, todas as tentativas falharam
        if last_exception:
            raise last_exception
        else:
            raise Exception("Nenhuma instância disponível para processar a requisição")
    
    def _get_available_instances(
        self,
        group: Optional[str] = None,
        exclude: Optional[Set[str]] = None
    ) -> List[ExchangeInstance]:
        """Obtém lista de instâncias disponíveis."""
        instances = []
        
        # Filtra por grupo se especificado
        if group and group in self._instance_groups:
            candidate_ids = self._instance_groups[group]
        else:
            candidate_ids = list(self._instances.keys())
        
        # Filtra instâncias disponíveis
        for instance_id in candidate_ids:
            if instance_id in self._instances:
                instance = self._instances[instance_id]
                
                if (instance.is_available and 
                    (not exclude or instance_id not in exclude)):
                    instances.append(instance)
        
        return sorted(instances, key=lambda x: x.priority)
    
    def _select_round_robin(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção round robin."""
        if not instances:
            return None
        
        instance = instances[self._current_index % len(instances)]
        self._current_index += 1
        return instance
    
    def _select_weighted_round_robin(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção round robin ponderada."""
        if not instances:
            return None
        
        # Cria lista expandida baseada nos pesos
        weighted_instances = []
        for instance in instances:
            weight_count = max(1, int(instance.weight * 10))
            weighted_instances.extend([instance] * weight_count)
        
        instance = weighted_instances[self._current_index % len(weighted_instances)]
        self._current_index += 1
        return instance
    
    def _select_least_connections(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção por menor número de conexões."""
        if not instances:
            return None
        
        return min(instances, key=lambda x: x.metrics.active_connections)
    
    def _select_least_response_time(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção por menor tempo de resposta."""
        if not instances:
            return None
        
        return min(instances, key=lambda x: x.metrics.avg_response_time)
    
    def _select_health_aware(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção baseada na saúde das instâncias."""
        if not instances:
            return None
        
        # Prioriza instâncias saudáveis
        healthy_instances = [i for i in instances if i.state == InstanceState.HEALTHY]
        if healthy_instances:
            return min(healthy_instances, key=lambda x: x.load_score)
        
        # Fallback para instâncias degradadas
        degraded_instances = [i for i in instances if i.state == InstanceState.DEGRADED]
        if degraded_instances:
            return min(degraded_instances, key=lambda x: x.load_score)
        
        return instances[0]
    
    def _select_adaptive(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção adaptativa combinando múltiplos fatores."""
        if not instances:
            return None
        
        # Combina vários fatores para seleção inteligente
        return min(instances, key=lambda x: x.load_score)
    
    def _select_random(self, instances: List[ExchangeInstance]) -> ExchangeInstance:
        """Seleção aleatória."""
        if not instances:
            return None
        
        return random.choice(instances)
    
    async def _health_check_loop(self) -> None:
        """Loop de verificação de saúde das instâncias."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no health check loop: {str(e)}")
                await asyncio.sleep(5)  # Espera antes de tentar novamente
    
    async def _perform_health_checks(self) -> None:
        """Executa verificação de saúde em todas as instâncias."""
        if not self._instances:
            return
        
        logger.debug(f"Executando health check em {len(self._instances)} instâncias")
        
        # Executa health checks em paralelo
        tasks = []
        for instance in self._instances.values():
            task = asyncio.create_task(instance.health_check())
            tasks.append((instance.id, task))
        
        # Aguarda resultados
        for instance_id, task in tasks:
            try:
                result = await task
                if not result:
                    logger.warning(f"Health check falhou para instância {instance_id}")
            except Exception as e:
                logger.error(f"Erro no health check da instância {instance_id}: {str(e)}")
        
        self._last_health_check = datetime.utcnow()
    
    def get_instance_by_id(self, instance_id: str) -> Optional[ExchangeInstance]:
        """Obtém instância pelo ID."""
        return self._instances.get(instance_id)
    
    def get_all_instances(self) -> List[ExchangeInstance]:
        """Obtém todas as instâncias."""
        return list(self._instances.values())
    
    def get_available_instances(self) -> List[ExchangeInstance]:
        """Obtém apenas instâncias disponíveis."""
        return [instance for instance in self._instances.values() if instance.is_available]
    
    def get_instance_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas das instâncias."""
        stats = {
            'total_instances': len(self._instances),
            'available_instances': len(self.get_available_instances()),
            'instances': {}
        }
        
        for instance_id, instance in self._instances.items():
            stats['instances'][instance_id] = {
                'state': instance.state.value,
                'weight': instance.weight,
                'priority': instance.priority,
                'metrics': {
                    'active_connections': instance.metrics.active_connections,
                    'total_requests': instance.metrics.total_requests,
                    'successful_requests': instance.metrics.successful_requests,
                    'failed_requests': instance.metrics.failed_requests,
                    'error_rate': instance.metrics.error_rate,
                    'avg_response_time': instance.metrics.avg_response_time,
                    'load_score': instance.load_score
                }
            }
        
        return stats
    
    def get_balancer_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do balanceador."""
        return {
            'strategy': self.strategy.value,
            'running': self._running,
            'last_health_check': self._last_health_check.isoformat() if self._last_health_check else None,
            'requests': self._stats,
            'instances': self.get_instance_stats()
        }
    
    def reset_stats(self) -> None:
        """Reseta estatísticas do balanceador."""
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_response_time': 0.0,
            'instance_selections': defaultdict(int)
        }
        
        # Reseta métricas das instâncias
        for instance in self._instances.values():
            instance.metrics = InstanceMetrics()


# Função de conveniência para criar balanceador com configuração padrão
def create_exchange_load_balancer(
    strategy: BalancingStrategy = BalancingStrategy.ADAPTIVE,
    health_check_interval: int = 30
) -> ExchangeLoadBalancer:
    """Cria um balanceador de carga com configuração padrão."""
    return ExchangeLoadBalancer(
        strategy=strategy,
        health_check_interval=health_check_interval,
        enable_circuit_breaker=True
    )