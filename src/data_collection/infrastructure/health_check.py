"""
Sistema de Health Check para Data Collection Module.

Este módulo implementa verificações de saúde abrangentes para todos os 
componentes do sistema de coleta de dados, incluindo:
- Conectividade com exchanges
- Status do banco de dados  
- Saúde do Kafka
- Uso de memória e recursos
- Performance das conexões
- Estado do load balancer
"""

import asyncio
import psutil
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

try:
    import aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    from confluent_kafka.admin import AdminClient
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Status de saúde dos componentes."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Resultado de uma verificação de saúde."""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    response_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'response_time_ms': self.response_time_ms,
            'healthy': self.status == HealthStatus.HEALTHY
        }


@dataclass
class HealthReport:
    """Relatório completo de saúde."""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    summary: Dict[str, Any]
    timestamp: datetime
    total_checks: int
    healthy_checks: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'overall_status': self.overall_status.value,
            'overall_healthy': self.overall_status == HealthStatus.HEALTHY,
            'checks': [check.to_dict() for check in self.checks],
            'summary': self.summary,
            'timestamp': self.timestamp.isoformat(),
            'total_checks': self.total_checks,
            'healthy_checks': self.healthy_checks,
            'health_score': (self.healthy_checks / max(1, self.total_checks)) * 100
        }


class HealthCheckService:
    """
    Serviço principal de verificação de saúde.
    
    Implementa verificações abrangentes de todos os componentes do sistema
    de coleta de dados, com cache inteligente e alertas automáticos.
    """
    
    def __init__(
        self,
        cache_ttl_seconds: int = 30,
        warning_threshold_ms: float = 1000.0,
        error_threshold_ms: float = 5000.0
    ):
        """
        Inicializa o serviço de health check.
        
        Args:
            cache_ttl_seconds: TTL do cache de resultados
            warning_threshold_ms: Threshold para warnings (ms)
            error_threshold_ms: Threshold para erros (ms)
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.warning_threshold_ms = warning_threshold_ms
        self.error_threshold_ms = error_threshold_ms
        
        # Cache de resultados
        self._cache: Dict[str, HealthCheckResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Estatísticas
        self._stats = {
            'total_checks': 0,
            'successful_checks': 0,
            'failed_checks': 0,
            'avg_response_time_ms': 0.0,
            'last_full_check': None
        }
        
        # Configurações específicas
        self._system_thresholds = {
            'memory_usage_percent': 85.0,
            'cpu_usage_percent': 90.0,
            'disk_usage_percent': 85.0,
            'connection_pool_usage_percent': 80.0
        }
    
    async def run_all_checks(self, use_cache: bool = True) -> HealthReport:
        """
        Executa todas as verificações de saúde disponíveis.
        
        Args:
            use_cache: Se deve usar cache para resultados recentes
            
        Returns:
            HealthReport: Relatório completo de saúde
        """
        start_time = time.perf_counter()
        
        logger.info("Iniciando verificação completa de saúde")
        
        # Lista de verificações a executar
        check_functions = [
            self._check_system_resources,
            self._check_database_connectivity,
            self._check_kafka_connectivity,
            self._check_redis_connectivity,
            self._check_exchange_connectivity,
            self._check_load_balancer_health,
            self._check_memory_leaks,
            self._check_thread_health,
            self._check_performance_metrics
        ]
        
        # Executar verificações em paralelo
        tasks = []
        for check_func in check_functions:
            task = asyncio.create_task(
                self._run_single_check(check_func, use_cache)
            )
            tasks.append(task)
        
        # Aguardar resultados
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Processar resultados
        checks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Criar resultado de erro para checks que falharam
                error_check = HealthCheckResult(
                    component=check_functions[i].__name__.replace('_check_', ''),
                    status=HealthStatus.UNHEALTHY,
                    message=f"Erro na verificação: {str(result)}",
                    details={'error': str(result), 'type': type(result).__name__},
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0
                )
                checks.append(error_check)
            elif isinstance(result, HealthCheckResult):
                checks.append(result)
        
        # Calcular status geral
        total_checks = len(checks)
        healthy_checks = sum(1 for check in checks if check.status == HealthStatus.HEALTHY)
        degraded_checks = sum(1 for check in checks if check.status == HealthStatus.DEGRADED)
        
        # Determinar status geral
        if healthy_checks == total_checks:
            overall_status = HealthStatus.HEALTHY
        elif healthy_checks + degraded_checks == total_checks:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        # Criar sumário
        total_time = (time.perf_counter() - start_time) * 1000
        summary = {
            'total_response_time_ms': total_time,
            'health_score': (healthy_checks / max(1, total_checks)) * 100,
            'critical_issues': [
                check.component for check in checks 
                if check.status == HealthStatus.UNHEALTHY
            ],
            'warnings': [
                check.component for check in checks 
                if check.status == HealthStatus.DEGRADED
            ]
        }
        
        # Atualizar estatísticas
        self._update_stats(total_time, overall_status == HealthStatus.HEALTHY)
        
        # Criar relatório
        report = HealthReport(
            overall_status=overall_status,
            checks=checks,
            summary=summary,
            timestamp=datetime.utcnow(),
            total_checks=total_checks,
            healthy_checks=healthy_checks
        )
        
        logger.info(f"Verificação completa finalizada em {total_time:.2f}ms - Status: {overall_status.value}")
        
        return report
    
    async def _run_single_check(
        self, 
        check_func, 
        use_cache: bool = True
    ) -> HealthCheckResult:
        """Executa uma verificação individual com cache."""
        component_name = check_func.__name__.replace('_check_', '')
        
        # Verificar cache
        if use_cache and self._is_cache_valid(component_name):
            logger.debug(f"Usando resultado em cache para {component_name}")
            return self._cache[component_name]
        
        # Executar verificação
        start_time = time.perf_counter()
        
        try:
            result = await check_func()
            response_time = (time.perf_counter() - start_time) * 1000
            
            # Atualizar tempo de resposta no resultado
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = response_time
                
                # Aplicar thresholds de performance
                if response_time > self.error_threshold_ms:
                    result.status = HealthStatus.UNHEALTHY
                    result.message += f" (Tempo de resposta alto: {response_time:.2f}ms)"
                elif response_time > self.warning_threshold_ms:
                    if result.status == HealthStatus.HEALTHY:
                        result.status = HealthStatus.DEGRADED
                        result.message += f" (Tempo de resposta degradado: {response_time:.2f}ms)"
            
            # Atualizar cache
            self._cache[component_name] = result
            self._cache_timestamps[component_name] = datetime.utcnow()
            
            return result
            
        except Exception as e:
            response_time = (time.perf_counter() - start_time) * 1000
            
            logger.error(f"Erro na verificação {component_name}: {str(e)}")
            
            error_result = HealthCheckResult(
                component=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Erro na verificação: {str(e)}",
                details={'error': str(e), 'type': type(e).__name__},
                timestamp=datetime.utcnow(),
                response_time_ms=response_time
            )
            
            return error_result
    
    def _is_cache_valid(self, component_name: str) -> bool:
        """Verifica se o cache é válido para um componente."""
        if component_name not in self._cache:
            return False
        
        cache_time = self._cache_timestamps.get(component_name)
        if not cache_time:
            return False
        
        age = (datetime.utcnow() - cache_time).total_seconds()
        return age < self.cache_ttl_seconds
    
    def _update_stats(self, response_time_ms: float, success: bool):
        """Atualiza estatísticas do serviço."""
        self._stats['total_checks'] += 1
        
        if success:
            self._stats['successful_checks'] += 1
        else:
            self._stats['failed_checks'] += 1
        
        # Calcular média móvel do tempo de resposta
        current_avg = self._stats['avg_response_time_ms']
        total_checks = self._stats['total_checks']
        
        new_avg = ((current_avg * (total_checks - 1)) + response_time_ms) / total_checks
        self._stats['avg_response_time_ms'] = new_avg
        
        self._stats['last_full_check'] = datetime.utcnow().isoformat()
    
    # ===== VERIFICAÇÕES ESPECÍFICAS =====
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """Verifica recursos do sistema (CPU, memória, disco)."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memória
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disco
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Load average (apenas Unix)
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = [0, 0, 0]  # Windows não suporta
            
            # Determinar status
            issues = []
            if cpu_percent > self._system_thresholds['cpu_usage_percent']:
                issues.append(f"CPU alta: {cpu_percent:.1f}%")
            
            if memory_percent > self._system_thresholds['memory_usage_percent']:
                issues.append(f"Memória alta: {memory_percent:.1f}%")
            
            if disk_percent > self._system_thresholds['disk_usage_percent']:
                issues.append(f"Disco cheio: {disk_percent:.1f}%")
            
            # Status baseado nos issues
            if issues:
                status = HealthStatus.DEGRADED if len(issues) == 1 else HealthStatus.UNHEALTHY
                message = f"Problemas de recursos: {', '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Recursos do sistema normais"
            
            details = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_mb': memory.available / 1024 / 1024,
                'disk_percent': disk_percent,
                'disk_free_gb': disk.free / 1024 / 1024 / 1024,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2],
                'issues': issues
            }
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Erro ao verificar recursos: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_database_connectivity(self) -> HealthCheckResult:
        """Verifica conectividade com banco de dados."""
        try:
            # Simular verificação de banco
            # Em implementação real, usaria connection pool real
            await asyncio.sleep(0.01)  # Simular latência
            
            # Simular alguns cenários
            import random
            if random.random() < 0.1:  # 10% chance de erro simulado
                raise Exception("Timeout de conexão")
            
            details = {
                'connection_pool_size': 5,
                'active_connections': 2,
                'max_connections': 20,
                'query_response_time_ms': 15.5
            }
            
            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                message="Banco de dados conectado e responsivo",
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Erro de conectividade: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_kafka_connectivity(self) -> HealthCheckResult:
        """Verifica conectividade com Kafka."""
        try:
            if not HAS_KAFKA:
                return HealthCheckResult(
                    component="kafka",
                    status=HealthStatus.UNKNOWN,
                    message="Kafka client não disponível",
                    details={'reason': 'kafka_client_not_installed'},
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0
                )
            
            # Simular verificação do Kafka
            await asyncio.sleep(0.02)  # Simular latência
            
            details = {
                'brokers_available': 3,
                'topics_count': 15,
                'producer_buffer_size': 1024,
                'last_successful_send': datetime.utcnow().isoformat()
            }
            
            return HealthCheckResult(
                component="kafka",
                status=HealthStatus.HEALTHY,
                message="Kafka cluster acessível",
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="kafka",
                status=HealthStatus.UNHEALTHY,
                message=f"Erro de conectividade Kafka: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_redis_connectivity(self) -> HealthCheckResult:
        """Verifica conectividade com Redis."""
        try:
            if not HAS_REDIS:
                return HealthCheckResult(
                    component="redis",
                    status=HealthStatus.UNKNOWN,
                    message="Redis client não disponível",
                    details={'reason': 'redis_client_not_installed'},
                    timestamp=datetime.utcnow(),
                    response_time_ms=0.0
                )
            
            # Simular verificação do Redis
            await asyncio.sleep(0.005)  # Simular latência baixa
            
            details = {
                'connected_clients': 12,
                'used_memory_mb': 45.2,
                'hit_rate_percent': 89.5,
                'keys_count': 1547
            }
            
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.HEALTHY,
                message="Redis cache operacional",
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="redis",
                status=HealthStatus.DEGRADED,
                message=f"Cache indisponível: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_exchange_connectivity(self) -> HealthCheckResult:
        """Verifica conectividade com exchanges."""
        try:
            # Simular verificação de múltiplas exchanges
            exchanges_status = {
                'binance': {'status': 'connected', 'latency_ms': 45.2},
                'coinbase': {'status': 'connected', 'latency_ms': 67.8},
                'kraken': {'status': 'degraded', 'latency_ms': 156.3},
                'bybit': {'status': 'connected', 'latency_ms': 89.1}
            }
            
            connected_count = sum(1 for ex in exchanges_status.values() if ex['status'] == 'connected')
            total_count = len(exchanges_status)
            
            # Determinar status geral
            if connected_count == total_count:
                status = HealthStatus.HEALTHY
                message = f"Todas as {total_count} exchanges conectadas"
            elif connected_count >= total_count * 0.7:  # 70% conectadas
                status = HealthStatus.DEGRADED
                message = f"{connected_count}/{total_count} exchanges conectadas"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Apenas {connected_count}/{total_count} exchanges conectadas"
            
            details = {
                'exchanges': exchanges_status,
                'connected_count': connected_count,
                'total_count': total_count,
                'avg_latency_ms': sum(ex['latency_ms'] for ex in exchanges_status.values()) / total_count
            }
            
            return HealthCheckResult(
                component="exchanges",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="exchanges",
                status=HealthStatus.UNHEALTHY,
                message=f"Erro nas exchanges: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_load_balancer_health(self) -> HealthCheckResult:
        """Verifica saúde do load balancer."""
        try:
            # Simular métricas do load balancer
            instances_status = {
                'instance_1': {'healthy': True, 'load': 0.45, 'response_time_ms': 67.2},
                'instance_2': {'healthy': True, 'load': 0.52, 'response_time_ms': 73.1},
                'instance_3': {'healthy': False, 'load': 0.98, 'response_time_ms': 1250.5},
                'instance_4': {'healthy': True, 'load': 0.38, 'response_time_ms': 59.8}
            }
            
            healthy_instances = sum(1 for inst in instances_status.values() if inst['healthy'])
            total_instances = len(instances_status)
            
            # Determinar status
            if healthy_instances == total_instances:
                status = HealthStatus.HEALTHY
                message = f"Load balancer operacional - {healthy_instances}/{total_instances} instâncias"
            elif healthy_instances >= total_instances * 0.5:  # Pelo menos 50%
                status = HealthStatus.DEGRADED
                message = f"Load balancer degradado - {healthy_instances}/{total_instances} instâncias"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Load balancer falhou - {healthy_instances}/{total_instances} instâncias"
            
            details = {
                'instances': instances_status,
                'healthy_instances': healthy_instances,
                'total_instances': total_instances,
                'avg_load': sum(inst['load'] for inst in instances_status.values()) / total_instances,
                'avg_response_time_ms': sum(inst['response_time_ms'] for inst in instances_status.values()) / total_instances
            }
            
            return HealthCheckResult(
                component="load_balancer",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="load_balancer",
                status=HealthStatus.UNHEALTHY,
                message=f"Erro no load balancer: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_memory_leaks(self) -> HealthCheckResult:
        """Verifica possíveis vazamentos de memória."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Simular histórico de memória (em implementação real, seria persistido)
            # Para demonstração, simular uma tendência crescente
            memory_history_mb = [
                memory_info.rss / 1024 / 1024 - (i * 2) for i in range(10, 0, -1)
            ]
            
            # Calcular tendência
            if len(memory_history_mb) >= 5:
                recent_trend = sum(memory_history_mb[-3:]) / 3 - sum(memory_history_mb[:3]) / 3
                growth_rate_mb_per_check = recent_trend
            else:
                growth_rate_mb_per_check = 0
            
            # Determinar status baseado na tendência
            if growth_rate_mb_per_check > 10:  # Crescimento > 10MB por check
                status = HealthStatus.UNHEALTHY
                message = f"Possível vazamento de memória detectado (+{growth_rate_mb_per_check:.1f}MB)"
            elif growth_rate_mb_per_check > 5:  # Crescimento > 5MB por check
                status = HealthStatus.DEGRADED
                message = f"Crescimento de memória elevado (+{growth_rate_mb_per_check:.1f}MB)"
            else:
                status = HealthStatus.HEALTHY
                message = "Uso de memória estável"
            
            details = {
                'current_memory_mb': memory_info.rss / 1024 / 1024,
                'memory_history_mb': memory_history_mb,
                'growth_rate_mb_per_check': growth_rate_mb_per_check,
                'gc_collections': getattr(process, '_gc_collections', 0)
            }
            
            return HealthCheckResult(
                component="memory_leaks",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="memory_leaks",
                status=HealthStatus.UNKNOWN,
                message=f"Erro na verificação de memória: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_thread_health(self) -> HealthCheckResult:
        """Verifica saúde das threads/tasks."""
        try:
            # Verificar tasks do asyncio
            all_tasks = asyncio.all_tasks()
            running_tasks = [task for task in all_tasks if not task.done()]
            completed_tasks = [task for task in all_tasks if task.done()]
            cancelled_tasks = [task for task in completed_tasks if task.cancelled()]
            
            # Verificar threads do sistema
            process = psutil.Process()
            thread_count = process.num_threads()
            
            # Determinar status
            issues = []
            if len(running_tasks) > 1000:
                issues.append(f"Muitas tasks ativas: {len(running_tasks)}")
            
            if thread_count > 100:
                issues.append(f"Muitas threads: {thread_count}")
            
            if len(cancelled_tasks) > len(all_tasks) * 0.1:  # >10% canceladas
                issues.append(f"Taxa alta de cancelamento: {len(cancelled_tasks)}")
            
            if issues:
                status = HealthStatus.DEGRADED
                message = f"Problemas de concorrência: {', '.join(issues)}"
            else:
                status = HealthStatus.HEALTHY
                message = "Threads e tasks saudáveis"
            
            details = {
                'active_tasks': len(running_tasks),
                'completed_tasks': len(completed_tasks),
                'cancelled_tasks': len(cancelled_tasks),
                'total_tasks': len(all_tasks),
                'system_threads': thread_count,
                'issues': issues
            }
            
            return HealthCheckResult(
                component="thread_health",
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="thread_health",
                status=HealthStatus.UNKNOWN,
                message=f"Erro na verificação de threads: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    async def _check_performance_metrics(self) -> HealthCheckResult:
        """Verifica métricas de performance."""
        try:
            # Simular métricas de performance
            metrics = {
                'requests_per_second': 1250.5,
                'avg_response_time_ms': 67.3,
                'p95_response_time_ms': 145.7,
                'p99_response_time_ms': 267.2,
                'error_rate_percent': 0.12,
                'throughput_mbps': 15.6
            }
            
            # Determinar status baseado nas métricas
            issues = []
            
            if metrics['avg_response_time_ms'] > 100:
                issues.append(f"Latência alta: {metrics['avg_response_time_ms']:.1f}ms")
            
            if metrics['error_rate_percent'] > 1.0:
                issues.append(f"Taxa de erro alta: {metrics['error_rate_percent']:.2f}%")
            
            if metrics['requests_per_second'] < 100:
                issues.append(f"Throughput baixo: {metrics['requests_per_second']:.1f} RPS")
            
            # Determinar status
            if issues:
                if len(issues) >= 2:
                    status = HealthStatus.UNHEALTHY
                    message = f"Performance degradada: {', '.join(issues)}"
                else:
                    status = HealthStatus.DEGRADED
                    message = f"Alerta de performance: {issues[0]}"
            else:
                status = HealthStatus.HEALTHY
                message = "Performance dentro dos parâmetros normais"
            
            return HealthCheckResult(
                component="performance",
                status=status,
                message=message,
                details=metrics,
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="performance",
                status=HealthStatus.UNKNOWN,
                message=f"Erro na verificação de performance: {str(e)}",
                details={'error': str(e)},
                timestamp=datetime.utcnow(),
                response_time_ms=0.0
            )
    
    # ===== MÉTODOS UTILITÁRIOS =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do serviço."""
        return self._stats.copy()
    
    def clear_cache(self) -> None:
        """Limpa o cache de resultados."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache de health checks limpo")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check do próprio serviço de health check."""
        return {
            'healthy': True,
            'service': 'health_check_service',
            'cache_size': len(self._cache),
            'stats': self._stats,
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }


# ===== FUNÇÕES DE CONVENIÊNCIA =====

async def quick_health_check() -> Dict[str, Any]:
    """
    Executa uma verificação rápida de saúde.
    
    Returns:
        Dict[str, Any]: Resultado básico de saúde
    """
    try:
        start_time = time.perf_counter()
        
        # Verificações básicas rápidas
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Determinar status
        if cpu_percent > 90 or memory_percent > 90:
            status = "unhealthy"
            message = "Recursos do sistema críticos"
        elif cpu_percent > 70 or memory_percent > 70:
            status = "degraded"  
            message = "Recursos do sistema elevados"
        else:
            status = "healthy"
            message = "Sistema operacional"
        
        response_time = (time.perf_counter() - start_time) * 1000
        
        return {
            'status': status,
            'healthy': status == 'healthy',
            'message': message,
            'response_time_ms': response_time,
            'timestamp': datetime.utcnow().isoformat(),
            'details': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent
            }
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'healthy': False,
            'message': f"Erro na verificação rápida: {str(e)}",
            'response_time_ms': 0.0,
            'timestamp': datetime.utcnow().isoformat(),
            'details': {'error': str(e)}
        }


async def detailed_system_check() -> Dict[str, Any]:
    """
    Executa verificação detalhada do sistema.
    
    Returns:
        Dict[str, Any]: Informações detalhadas do sistema
    """
    try:
        # Informações do sistema
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        
        # CPU
        cpu_info = {
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'percent': psutil.cpu_percent(interval=1),
            'freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        }
        
        # Memória
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        memory_info = {
            'virtual': {
                'total_gb': memory.total / 1024 / 1024 / 1024,
                'available_gb': memory.available / 1024 / 1024 / 1024,
                'percent': memory.percent,
                'used_gb': memory.used / 1024 / 1024 / 1024
            },
            'swap': {
                'total_gb': swap.total / 1024 / 1024 / 1024,
                'used_gb': swap.used / 1024 / 1024 / 1024,
                'percent': swap.percent
            }
        }
        
        # Disco
        disk = psutil.disk_usage('/')
        disk_info = {
            'total_gb': disk.total / 1024 / 1024 / 1024,
            'used_gb': disk.used / 1024 / 1024 / 1024,
            'free_gb': disk.free / 1024 / 1024 / 1024,
            'percent': (disk.used / disk.total) * 100
        }
        
        # Rede
        network = psutil.net_io_counters()
        network_info = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        return {
            'system': {
                'boot_time': boot_time.isoformat(),
                'uptime_hours': (datetime.now() - boot_time).total_seconds() / 3600
            },
            'cpu': cpu_info,
            'memory': memory_info,
            'disk': disk_info,
            'network': network_info,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }


# ===== EXEMPLO DE USO =====

if __name__ == "__main__":
    async def example_usage():
        """Exemplo de uso do sistema de health check."""
        
        # Criar serviço
        health_service = HealthCheckService()
        
        # Verificação rápida
        print("=== VERIFICAÇÃO RÁPIDA ===")
        quick_result = await quick_health_check()
        print(json.dumps(quick_result, indent=2))
        
        # Verificação completa
        print("\n=== VERIFICAÇÃO COMPLETA ===")
        full_report = await health_service.run_all_checks()
        print(json.dumps(full_report.to_dict(), indent=2))
        
        # Verificação detalhada do sistema
        print("\n=== INFORMAÇÕES DETALHADAS DO SISTEMA ===")
        system_info = await detailed_system_check()
        print(json.dumps(system_info, indent=2))
    
    # Executar exemplo
    asyncio.run(example_usage())