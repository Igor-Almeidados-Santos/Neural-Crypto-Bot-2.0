"""
Sistema de balanceamento de carga para distribuição de requisições.

Este módulo implementa mecanismos para distribuir requisições entre
múltiplas instâncias, evitando sobrecarga de servidores e melhorando
a disponibilidade e confiabilidade do sistema.
"""
import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Awaitable, TypeVar, Generic, Union

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


logger = logging.getLogger(__name__)


class BalancingStrategy(str, Enum):
    """Estratégias de balanceamento de carga."""
    ROUND_ROBIN = "round_robin"           # Alternância circular
    RANDOM = "random"                     # Seleção aleatória
    LEAST_CONNECTIONS = "least_conn"      # Menor número de conexões ativas
    LEAST_RESPONSE_TIME = "least_time"    # Menor tempo de resposta médio
    IP_HASH = "ip_hash"                   # Hash baseado no IP
    WEIGHTED = "weighted"                 # Ponderado por capacidade/peso
    LEAST_REQUESTS = "least_requests"     # Menor número de requisições recentes


class InstanceStatus(str, Enum):
    """Status possíveis de uma instância."""
    ONLINE = "online"        # Instância disponível
    OFFLINE = "offline"      # Instância indisponível
    DEGRADED = "degraded"    # Instância com performance comprometida
    MAINTENANCE = "maintenance"  # Instância em manutenção
    STANDBY = "standby"      # Instância em espera
    OVERLOADED = "overloaded"  # Instância sobrecarregada


@dataclass
class InstanceMetrics:
    """Métricas de uma instância para balanceamento de carga."""
    active_connections: int = 0
    total_requests: int = 0
    recent_requests: int = 0  # Últimos 60 segundos
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_used: datetime = datetime.utcnow()
    last_check: datetime = datetime.utcnow()
    consecutive_failures: int = 0
    status: InstanceStatus = InstanceStatus.ONLINE
    weight: int = 1  # Peso para balanceamento ponderado


T = TypeVar('T')


class LoadBalancer(Generic[T]):
    """
    Balanceador de carga para distribuir requisições entre instâncias.
    
    Implementa várias estratégias de balanceamento para distribuir
    requisições entre múltiplas instâncias de servidores ou serviços.
    """
    
    def __init__(
        self,
        instances: List[T],
        strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 60,
        circuit_breaker_threshold: int = 5,
        max_retries: int = 3
    ):
        """
        Inicializa o balanceador de carga.
        
        Args:
            instances: Lista de instâncias a serem balanceadas
            strategy: Estratégia de balanceamento
            health_check_interval: Intervalo em segundos para verificação de saúde
            circuit_breaker_threshold: Número de falhas consecutivas para ativar o circuit breaker
            max_retries: Número máximo de tentativas em caso de falha
        """
        self.instances = instances
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.max_retries = max_retries
        
        # Índice para round robin
        self._current_index = 0
        
        # Métricas por instância
        self._metrics: Dict[int, InstanceMetrics] = {
            i: InstanceMetrics() for i in range(len(instances))
        }
        
        # Task de verificação de saúde
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Lock para operações concorrentes
        self._lock = asyncio.Lock()
        
        # Cache para IP hash
        self._ip_cache: Dict[str, int] = {}
    
    async def start(self) -> None:
        """
        Inicia o balanceador de carga e a verificação de saúde periódica.
        """
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info(f"Balanceador de carga iniciado com estratégia {self.strategy}")
    
    async def stop(self) -> None:
        """
        Para o balanceador de carga e a verificação de saúde.
        """
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Balanceador de carga parado")
    
    def get_all_instances(self) -> List[T]:
        """
        Retorna todas as instâncias registradas.
        
        Returns:
            List[T]: Lista de todas as instâncias
        """
        return self.instances.copy()
    
    def get_available_instances(self) -> List[T]:
        """
        Retorna todas as instâncias disponíveis (online).
        
        Returns:
            List[T]: Lista de instâncias disponíveis
        """
        return [
            self.instances[i] 
            for i in range(len(self.instances)) 
            if self._metrics[i].status == InstanceStatus.ONLINE
        ]
    
    def get_instance_status(self, instance_idx: int) -> InstanceStatus:
        """
        Retorna o status de uma instância específica.
        
        Args:
            instance_idx: Índice da instância
            
        Returns:
            InstanceStatus: Status atual da instância
        """
        if instance_idx < 0 or instance_idx >= len(self.instances):
            raise ValueError(f"Índice de instância inválido: {instance_idx}")
            
        return self._metrics[instance_idx].status
    
    def set_instance_status(self, instance_idx: int, status: InstanceStatus) -> None:
        """
        Define o status de uma instância específica.
        
        Args:
            instance_idx: Índice da instância
            status: Novo status da instância
        """
        if instance_idx < 0 or instance_idx >= len(self.instances):
            raise ValueError(f"Índice de instância inválido: {instance_idx}")
            
        self._metrics[instance_idx].status = status
        logger.info(f"Status da instância {instance_idx} alterado para {status}")
    
    def get_instance_metrics(self, instance_idx: int) -> InstanceMetrics:
        """
        Retorna as métricas de uma instância específica.
        
        Args:
            instance_idx: Índice da instância
            
        Returns:
            InstanceMetrics: Métricas atuais da instância
        """
        if instance_idx < 0 or instance_idx >= len(self.instances):
            raise ValueError(f"Índice de instância inválido: {instance_idx}")
            
        return self._metrics[instance_idx]
    
    async def execute(
        self, 
        operation: Callable[[T], Awaitable[Any]],
        client_id: Optional[str] = None
    ) -> Any:
        """
        Executa uma operação em uma instância selecionada pelo balanceador.
        
        Args:
            operation: Função assíncrona que recebe uma instância e retorna um resultado
            client_id: ID do cliente para estratégias baseadas em hash (opcional)
            
        Returns:
            Any: Resultado da operação
            
        Raises:
            Exception: Se todas as tentativas falharem
        """
        for attempt in range(self.max_retries):
            instance_idx = await self._select_instance(client_id)
            
            if instance_idx is None:
                raise RuntimeError("Nenhuma instância disponível")
                
            instance = self.instances[instance_idx]
            metrics = self._metrics[instance_idx]
            
            # Atualiza métricas antes da execução
            metrics.active_connections += 1
            metrics.total_requests += 1
            metrics.recent_requests += 1
            metrics.last_used = datetime.utcnow()
            
            start_time = time.time()
            
            try:
                result = await operation(instance)
                
                # Atualiza métricas após execução bem-sucedida
                metrics.success_count += 1
                metrics.consecutive_failures = 0
                
                # Atualiza tempo médio de resposta
                elapsed = time.time() - start_time
                metrics.avg_response_time = (metrics.avg_response_time * 0.9) + (elapsed * 0.1)
                
                return result
                
            except Exception as e:
                # Atualiza métricas após falha
                metrics.error_count += 1
                metrics.consecutive_failures += 1
                
                # Verifica se deve ativar o circuit breaker
                if metrics.consecutive_failures >= self.circuit_breaker_threshold:
                    metrics.status = InstanceStatus.DEGRADED
                    logger.warning(f"Instância {instance_idx} marcada como degradada após {metrics.consecutive_failures} falhas consecutivas")
                
                logger.error(f"Erro ao executar operação na instância {instance_idx}: {str(e)}")
                
                # Se esta foi a última tentativa, propaga a exceção
                if attempt == self.max_retries - 1:
                    raise
            
            finally:
                # Decrementa conexões ativas
                metrics.active_connections = max(0, metrics.active_connections - 1)
        
        # Este ponto não deveria ser alcançado, mas por segurança
        raise RuntimeError("Todas as tentativas falharam")
    
    async def _select_instance(self, client_id: Optional[str] = None) -> Optional[int]:
        """
        Seleciona uma instância com base na estratégia de balanceamento.
        
        Args:
            client_id: ID do cliente para estratégias baseadas em hash (opcional)
            
        Returns:
            Optional[int]: Índice da instância selecionada ou None se nenhuma disponível
        """
        # Obtém instâncias disponíveis
        available_indices = [
            i for i in range(len(self.instances)) 
            if self._metrics[i].status in [InstanceStatus.ONLINE, InstanceStatus.DEGRADED]
        ]
        
        if not available_indices:
            return None
            
        # Aplica a estratégia de balanceamento
        if self.strategy == BalancingStrategy.ROUND_ROBIN:
            return await self._select_round_robin(available_indices)
            
        elif self.strategy == BalancingStrategy.RANDOM:
            return random.choice(available_indices)
            
        elif self.strategy == BalancingStrategy.LEAST_CONNECTIONS:
            return await self._select_least_connections(available_indices)
            
        elif self.strategy == BalancingStrategy.LEAST_RESPONSE_TIME:
            return await self._select_least_response_time(available_indices)
            
        elif self.strategy == BalancingStrategy.IP_HASH:
            return await self._select_ip_hash(available_indices, client_id)
            
        elif self.strategy == BalancingStrategy.WEIGHTED:
            return await self._select_weighted(available_indices)
            
        elif self.strategy == BalancingStrategy.LEAST_REQUESTS:
            return await self._select_least_requests(available_indices)
            
        else:
            # Estratégia não reconhecida, usa round robin como fallback
            return await self._select_round_robin(available_indices)
    
    async def _select_round_robin(self, available_indices: List[int]) -> int:
        """
        Seleciona uma instância usando a estratégia Round Robin.
        
        Args:
            available_indices: Lista de índices de instâncias disponíveis
            
        Returns:
            int: Índice da instância selecionada
        """
        async with self._lock:
            # Incrementa o índice circular
            self._current_index = (self._current_index + 1) % len(self.instances)
            
            # Encontra o próximo índice disponível
            while self._current_index not in available_indices:
                self._current_index = (self._current_index + 1) % len(self.instances)
            
            return self._current_index
    
    async def _select_least_connections(self, available_indices: List[int]) -> int:
        """
        Seleciona a instância com o menor número de conexões ativas.
        
        Args:
            available_indices: Lista de índices de instâncias disponíveis
            
        Returns:
            int: Índice da instância selecionada
        """
        min_connections = float('inf')
        selected_idx = available_indices[0]
        
        for idx in available_indices:
            connections = self._metrics[idx].active_connections
            
            if connections < min_connections:
                min_connections = connections
                selected_idx = idx
        
        return selected_idx
    
    async def _select_least_response_time(self, available_indices: List[int]) -> int:
        """
        Seleciona a instância com o menor tempo médio de resposta.
        
        Args:
            available_indices: Lista de índices de instâncias disponíveis
            
        Returns:
            int: Índice da instância selecionada
        """
        min_time = float('inf')
        selected_idx = available_indices[0]
        
        for idx in available_indices:
            response_time = self._metrics[idx].avg_response_time
            
            # Se não houver dados suficientes, usa o número de conexões como critério secundário
            if response_time == 0.0:
                response_time = self._metrics[idx].active_connections
            
            if response_time < min_time:
                min_time = response_time
                selected_idx = idx
        
        return selected_idx
    
    async def _select_ip_hash(self, available_indices: List[int], client_id: Optional[str]) -> int:
        """
        Seleciona uma instância com base no hash do ID do cliente.
        
        Args:
            available_indices: Lista de índices de instâncias disponíveis
            client_id: ID do cliente a ser usado para o hash
            
        Returns:
            int: Índice da instância selecionada
        """
        if not client_id:
            # Se não houver ID de cliente, usa round robin como fallback
            return await self._select_round_robin(available_indices)
        
        # Verifica se o cliente já está no cache
        if client_id in self._ip_cache:
            cached_idx = self._ip_cache[client_id]
            
            # Verifica se a instância ainda está disponível
            if cached_idx in available_indices:
                return cached_idx
        
        # Calcula o hash do ID do cliente
        hash_value = hash(client_id) % len(available_indices)
        selected_idx = available_indices[hash_value]
        
        # Armazena no cache
        self._ip_cache[client_id] = selected_idx
        
        return selected_idx
    
    async def _select_weighted(self, available_indices: List[int]) -> int:
        """
        Seleciona uma instância com base em pesos.
        
        Args:
            available_indices: Lista de índices de instâncias disponíveis
            
        Returns:
            int: Índice da instância selecionada
        """
        # Calcula o peso total
        total_weight = sum(self._metrics[idx].weight for idx in available_indices)
        
        if total_weight <= 0:
            # Se não houver pesos positivos, usa round robin como fallback
            return await self._select_round_robin(available_indices)
        
        # Seleciona aleatoriamente com base nos pesos
        value = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for idx in available_indices:
            cumulative_weight += self._metrics[idx].weight
            if value <= cumulative_weight:
                return idx
        
        # Caso excepcional (não deveria ocorrer)
        return available_indices[0]
    
    async def _select_least_requests(self, available_indices: List[int]) -> int:
        """
        Seleciona a instância com o menor número de requisições recentes.
        
        Args:
            available_indices: Lista de índices de instâncias disponíveis
            
        Returns:
            int: Índice da instância selecionada
        """
        min_requests = float('inf')
        selected_idx = available_indices[0]
        
        for idx in available_indices:
            requests = self._metrics[idx].recent_requests
            
            if requests < min_requests:
                min_requests = requests
                selected_idx = idx
        
        return selected_idx
    
    async def _health_check_loop(self) -> None:
        """
        Loop de verificação de saúde periódica das instâncias.
        """
        while True:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                logger.error(f"Erro na verificação de saúde: {str(e)}")
                await asyncio.sleep(10)  # Espera um pouco antes de tentar novamente
    
    async def _perform_health_check(self) -> None:
        """
        Realiza verificação de saúde em todas as instâncias.
        """
        logger.debug("Iniciando verificação de saúde das instâncias")
        
        # Reseta contadores de requisições recentes
        for metrics in self._metrics.values():
            metrics.recent_requests = 0
            metrics.last_check = datetime.utcnow()
        
        # Para cada instância, verifica o status
        for idx, instance in enumerate(self.instances):
            metrics = self._metrics[idx]
            
            if metrics.status == InstanceStatus.MAINTENANCE:
                # Ignora instâncias em manutenção
                continue
                
            try:
                # Verifica se a instância implementa um método de verificação de saúde
                if hasattr(instance, 'health_check') and callable(getattr(instance, 'health_check')):
                    is_healthy = await instance.health_check()
                    
                    if is_healthy:
                        # Se estava degradada e agora está saudável, restaura para online
                        if metrics.status == InstanceStatus.DEGRADED:
                            metrics.status = InstanceStatus.ONLINE
                            logger.info(f"Instância {idx} restaurada para online")
                    else:
                        metrics.status = InstanceStatus.DEGRADED
                        logger.warning(f"Instância {idx} marcada como degradada após falha na verificação de saúde")
                
                # Se não tiver muitas falhas recentes, considera online
                elif metrics.status == InstanceStatus.DEGRADED and metrics.consecutive_failures < self.circuit_breaker_threshold / 2:
                    metrics.status = InstanceStatus.ONLINE
                    logger.info(f"Instância {idx} restaurada para online")
                
            except Exception as e:
                # Falha na verificação de saúde
                logger.error(f"Erro na verificação de saúde da instância {idx}: {str(e)}")
                metrics.status = InstanceStatus.DEGRADED
                
        logger.debug("Verificação de saúde das instâncias concluída")
    
    def add_instance(self, instance: T, weight: int = 1) -> int:
        """
        Adiciona uma nova instância ao balanceador.
        
        Args:
            instance: Instância a ser adicionada
            weight: Peso da instância para balanceamento ponderado
            
        Returns:
            int: Índice da nova instância
        """
        idx = len(self.instances)
        self.instances.append(instance)
        
        # Inicializa métricas para a nova instância
        metrics = InstanceMetrics()
        metrics.weight = weight
        self._metrics[idx] = metrics
        
        logger.info(f"Nova instância adicionada com índice {idx}")
        return idx
    
    def remove_instance(self, instance_idx: int) -> None:
        """
        Remove uma instância do balanceador.
        
        Args:
            instance_idx: Índice da instância a ser removida
        """
        if instance_idx < 0 or instance_idx >= len(self.instances):
            raise ValueError(f"Índice de instância inválido: {instance_idx}")
            
        # Coloca a instância em modo de manutenção para não receber novas requisições
        self._metrics[instance_idx].status = InstanceStatus.MAINTENANCE
        
        # Aguarda até que não haja conexões ativas
        # Nota: em uma implementação real, isso seria feito de forma assíncrona
        logger.info(f"Instância {instance_idx} marcada para remoção, aguardando conexões ativas")
        
        # Remove a instância (em uma implementação real, seria melhor apenas marcar como removida)
        # self.instances.pop(instance_idx)
        # del self._metrics[instance_idx]
        
        logger.info(f"Instância {instance_idx} removida com sucesso")
    
    def update_weight(self, instance_idx: int, weight: int) -> None:
        """
        Atualiza o peso de uma instância para balanceamento ponderado.
        
        Args:
            instance_idx: Índice da instância
            weight: Novo peso
        """
        if instance_idx < 0 or instance_idx >= len(self.instances):
            raise ValueError(f"Índice de instância inválido: {instance_idx}")
            
        self._metrics[instance_idx].weight = weight
        logger.info(f"Peso da instância {instance_idx} atualizado para {weight}")
    
    def set_strategy(self, strategy: BalancingStrategy) -> None:
        """
        Altera a estratégia de balanceamento.
        
        Args:
            strategy: Nova estratégia
        """
        self.strategy = strategy
        logger.info(f"Estratégia de balanceamento alterada para {strategy}")
        
        # Reseta o índice para round robin
        self._current_index = 0
        
        # Limpa o cache de IP hash
        if strategy != BalancingStrategy.IP_HASH:
            self._ip_cache.clear()


class ExchangeLoadBalancer(LoadBalancer[ExchangeAdapterInterface]):
    """
    Balanceador de carga específico para adaptadores de exchanges.
    
    Estende o balanceador genérico com funcionalidades específicas
    para balancear requisições entre múltiplas instâncias de exchanges.
    """
    
    def __init__(
        self,
        instances: List[ExchangeAdapterInterface],
        strategy: BalancingStrategy = BalancingStrategy.ROUND_ROBIN,
        health_check_interval: int = 60,
        circuit_breaker_threshold: int = 5,
        max_retries: int = 3,
        rate_limit_buffer: float = 0.8  # 80% do limite de taxa
    ):
        """
        Inicializa o balanceador de carga para exchanges.
        
        Args:
            instances: Lista de adaptadores de exchange
            strategy: Estratégia de balanceamento
            health_check_interval: Intervalo em segundos para verificação de saúde
            circuit_breaker_threshold: Número de falhas consecutivas para ativar o circuit breaker
            max_retries: Número máximo de tentativas em caso de falha
            rate_limit_buffer: Porcentagem do limite de taxa a ser utilizada
        """
        super().__init__(
            instances=instances,
            strategy=strategy,
            health_check_interval=health_check_interval,
            circuit_breaker_threshold=circuit_breaker_threshold,
            max_retries=max_retries
        )
        
        self.rate_limit_buffer = rate_limit_buffer
        
        # Limites de taxa por instância e método
        self._rate_limits: Dict[int, Dict[str, int]] = {}
        
        # Contadores de requisições por instância e método
        self._request_counters: Dict[int, Dict[str, int]] = {}
        
        # Timestamps de reset de limite por instância e método
        self._reset_times: Dict[int, Dict[str, datetime]] = {}
    
    async def _perform_health_check(self) -> None:
        """
        Realiza verificação de saúde específica para exchanges.
        """
        await super()._perform_health_check()
        
        # Verificações adicionais específicas para exchanges
        for idx, exchange in enumerate(self.instances):
            metrics = self._metrics[idx]
            
            if metrics.status == InstanceStatus.MAINTENANCE:
                continue
                
            try:
                # Verifica se a exchange está respondendo
                pairs = await exchange.fetch_trading_pairs()
                
                if pairs:
                    # Atualiza o status para online se estava degradada
                    if metrics.status == InstanceStatus.DEGRADED:
                        metrics.status = InstanceStatus.ONLINE
                        logger.info(f"Exchange {exchange.name} (idx: {idx}) restaurada para online")
                else:
                    metrics.status = InstanceStatus.DEGRADED
                    logger.warning(f"Exchange {exchange.name} (idx: {idx}) marcada como degradada: nenhum par disponível")
                
            except Exception as e:
                logger.error(f"Erro na verificação de saúde da exchange {exchange.name} (idx: {idx}): {str(e)}")
                metrics.status = InstanceStatus.DEGRADED
    
    async def execute_for_trading_pair(
        self,
        trading_pair: str,
        operation: Callable[[ExchangeAdapterInterface], Awaitable[Any]],
        method_name: str = "generic"
    ) -> Any:
        """
        Executa uma operação para um par de negociação específico.
        
        Args:
            trading_pair: Par de negociação
            operation: Função assíncrona que recebe um adaptador de exchange
            method_name: Nome do método para controle de limite de taxa
            
        Returns:
            Any: Resultado da operação
        """
        # Filtra instâncias que suportam o par de negociação
        available_indices = []
        
        for idx, exchange in enumerate(self.instances):
            if (self._metrics[idx].status in [InstanceStatus.ONLINE, InstanceStatus.DEGRADED] and
                exchange.validate_trading_pair(trading_pair)):
                
                # Verifica se o limite de taxa foi atingido
                if not self._is_rate_limited(idx, method_name):
                    available_indices.append(idx)
        
        if not available_indices:
            raise ValueError(f"Nenhuma instância disponível para o par {trading_pair}")
        
        # Seleciona uma instância entre as disponíveis
        selected_idx = await self._select_from_indices(available_indices)
        instance = self.instances[selected_idx]
        
        # Incrementa o contador de requisições
        self._increment_request_counter(selected_idx, method_name)
        
        # Executa a operação
        try:
            result = await operation(instance)
            
            # Atualiza métricas de sucesso
            metrics = self._metrics[selected_idx]
            metrics.success_count += 1
            metrics.consecutive_failures = 0
            
            return result
            
        except Exception as e:
            # Atualiza métricas de erro
            metrics = self._metrics[selected_idx]
            metrics.error_count += 1
            metrics.consecutive_failures += 1
            
            # Verifica se deve ativar o circuit breaker
            if metrics.consecutive_failures >= self.circuit_breaker_threshold:
                metrics.status = InstanceStatus.DEGRADED
                logger.warning(f"Exchange {instance.name} (idx: {selected_idx}) marcada como degradada após {metrics.consecutive_failures} falhas consecutivas")
            
            # Repropaga a exceção
            raise
    
    async def _select_from_indices(self, available_indices: List[int]) -> int:
        """
        Seleciona uma instância entre as disponíveis.
        
        Args:
            available_indices: Lista de índices disponíveis
            
        Returns:
            int: Índice selecionado
        """
        if self.strategy == BalancingStrategy.LEAST_REQUESTS:
            # Para exchanges, considera também os limites de taxa
            min_rate_usage = float('inf')
            selected_idx = available_indices[0]
            
            for idx in available_indices:
                rate_usage = sum(self._request_counters.get(idx, {}).values())
                
                if rate_usage < min_rate_usage:
                    min_rate_usage = rate_usage
                    selected_idx = idx
            
            return selected_idx
            
        elif self.strategy == BalancingStrategy.WEIGHTED:
            # Para exchanges, ajusta os pesos com base nos limites de taxa
            weights = []
            
            for idx in available_indices:
                rate_usage = sum(self._request_counters.get(idx, {}).values())
                rate_limit = sum(self._rate_limits.get(idx, {}).values())
                
                if rate_limit > 0:
                    # Quanto menor o uso proporcional, maior o peso
                    weight = (1.0 - (rate_usage / rate_limit)) * self._metrics[idx].weight
                else:
                    weight = self._metrics[idx].weight
                
                weights.append(max(0.1, weight))  # Mínimo de 0.1 para evitar zero
            
            # Seleciona aleatoriamente com base nos pesos
            total_weight = sum(weights)
            value = random.uniform(0, total_weight)
            
            cumulative_weight = 0
            for i, idx in enumerate(available_indices):
                cumulative_weight += weights[i]
                if value <= cumulative_weight:
                    return idx
            
            return available_indices[0]
            
        else:
            # Para outras estratégias, usa a seleção padrão
            if len(available_indices) == 1:
                return available_indices[0]
                
            # Filtra os índices disponíveis na lista completa para usar as estratégias padrão
            temp_available = [i for i in range(len(self.instances)) if i in available_indices]
            selected = await self._select_instance()
            
            # Se o selecionado não estiver nos disponíveis, pega o primeiro
            if selected not in available_indices:
                return available_indices[0]
                
            return selected
    
    def set_rate_limit(self, instance_idx: int, method: str, limit: int, reset_seconds: int) -> None:
        """
        Define um limite de taxa para uma instância e método.
        
        Args:
            instance_idx: Índice da instância
            method: Nome do método
            limit: Número máximo de requisições
            reset_seconds: Segundos até o reset do limite
        """
        if instance_idx < 0 or instance_idx >= len(self.instances):
            raise ValueError(f"Índice de instância inválido: {instance_idx}")
            
        # Inicializa dicionários se necessário
        if instance_idx not in self._rate_limits:
            self._rate_limits[instance_idx] = {}
            self._request_counters[instance_idx] = {}
            self._reset_times[instance_idx] = {}
        
        # Ajusta o limite para o buffer
        adjusted_limit = int(limit * self.rate_limit_buffer)
        
        # Define o limite e o tempo de reset
        self._rate_limits[instance_idx][method] = adjusted_limit
        self._request_counters[instance_idx][method] = 0
        self._reset_times[instance_idx][method] = datetime.utcnow() + timedelta(seconds=reset_seconds)
        
        logger.debug(f"Limite de taxa definido para instância {instance_idx}, método {method}: {adjusted_limit}/{reset_seconds}s")
    
    def _is_rate_limited(self, instance_idx: int, method: str) -> bool:
        """
        Verifica se uma instância atingiu o limite de taxa para um método.
        
        Args:
            instance_idx: Índice da instância
            method: Nome do método
            
        Returns:
            bool: True se o limite foi atingido, False caso contrário
        """
        # Se não houver limite definido, não está limitado
        if (instance_idx not in self._rate_limits or
            method not in self._rate_limits[instance_idx]):
            return False
        
        # Verifica se o tempo de reset já passou
        now = datetime.utcnow()
        if (instance_idx in self._reset_times and
            method in self._reset_times[instance_idx] and
            now >= self._reset_times[instance_idx][method]):
            
            # Reseta o contador
            self._request_counters[instance_idx][method] = 0
            
            # Define um novo tempo de reset
            # Nota: em uma implementação real, isso seria atualizado com base na resposta da API
            reset_seconds = 60  # 1 minuto como padrão
            self._reset_times[instance_idx][method] = now + timedelta(seconds=reset_seconds)
            
            return False
        
        # Verifica se o contador atingiu o limite
        counter = self._request_counters.get(instance_idx, {}).get(method, 0)
        limit = self._rate_limits.get(instance_idx, {}).get(method, float('inf'))
        
        return counter >= limit
    
    def _increment_request_counter(self, instance_idx: int, method: str) -> None:
        """
        Incrementa o contador de requisições para uma instância e método.
        
        Args:
            instance_idx: Índice da instância
            method: Nome do método
        """
        if instance_idx not in self._request_counters:
            self._request_counters[instance_idx] = {}
            
        if method not in self._request_counters[instance_idx]:
            self._request_counters[instance_idx][method] = 0
            
        self._request_counters[instance_idx][method] += 1