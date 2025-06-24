import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from risk_management.domain.entities.risk_metric import MetricType
from risk_management.domain.entities.risk_profile import RiskLevel, RiskProfile
from risk_management.domain.events.risk_threshold_breached_event import RiskThresholdBreachedEvent
from risk_management.infrastructure.alert_notifier import AlertNotifier


class CircuitBreakerState(Enum):
    """Estados possíveis do circuit breaker."""
    CLOSED = "closed"  # Funcionamento normal
    OPEN = "open"      # Interrompido
    HALF_OPEN = "half_open"  # Modo de teste após período de cooldown


@dataclass
class CircuitBreakerConfig:
    """Configuração de um circuit breaker."""
    name: str
    cooldown_period_minutes: int = 60
    reset_threshold: int = 3
    consecutive_success_to_close: int = 5
    failure_threshold: int = 3
    recovery_timeout_minutes: int = 120
    auto_reset: bool = False
    requires_manual_reset_if_critical: bool = True
    metrics_to_monitor: Set[MetricType] = None


class CircuitBreakerService:
    """
    Serviço responsável por gerenciar circuit breakers que podem interromper 
    negociações quando métricas de risco excedem limites críticos.
    """
    
    def __init__(self, alert_notifier: AlertNotifier, logger: Optional[logging.Logger] = None):
        """
        Inicializa o serviço de circuit breaker.
        
        Args:
            alert_notifier: Serviço para envio de notificações de alerta
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._alert_notifier = alert_notifier
        self._circuit_breakers: Dict[str, Tuple[CircuitBreakerConfig, CircuitBreakerState, datetime]] = {}
        self._breaker_failure_counts: Dict[str, int] = {}
        self._breaker_success_counts: Dict[str, int] = {}
        self._last_events: Dict[str, RiskThresholdBreachedEvent] = {}
    
    def register_circuit_breaker(self, entity_id: str, config: CircuitBreakerConfig) -> None:
        """
        Registra um novo circuit breaker para uma entidade (portfolio, estratégia ou ativo).
        
        Args:
            entity_id: Identificador da entidade
            config: Configuração do circuit breaker
        """
        if entity_id in self._circuit_breakers:
            self._logger.warning(f"Circuit breaker para {entity_id} já existe e será substituído")
        
        self._circuit_breakers[entity_id] = (config, CircuitBreakerState.CLOSED, datetime.utcnow())
        self._breaker_failure_counts[entity_id] = 0
        self._breaker_success_counts[entity_id] = 0
        self._logger.info(f"Circuit breaker '{config.name}' registrado para {entity_id}")
    
    def get_state(self, entity_id: str) -> Optional[CircuitBreakerState]:
        """
        Obtém o estado atual de um circuit breaker.
        
        Args:
            entity_id: Identificador da entidade
            
        Returns:
            Estado atual do circuit breaker ou None se não existir
        """
        if entity_id not in self._circuit_breakers:
            return None
        
        return self._circuit_breakers[entity_id][1]
    
    def is_open(self, entity_id: str) -> bool:
        """
        Verifica se um circuit breaker está aberto (negociações interrompidas).
        
        Args:
            entity_id: Identificador da entidade
            
        Returns:
            True se o circuit breaker estiver aberto, False caso contrário
        """
        state = self.get_state(entity_id)
        return state == CircuitBreakerState.OPEN
    
    def process_threshold_event(self, event: RiskThresholdBreachedEvent) -> None:
        """
        Processa um evento de violação de limite e atualiza o estado do circuit breaker.
        
        Args:
            event: Evento de violação de limite de risco
        """
        entity_id = event.portfolio_id or event.strategy_id or event.asset_id
        if not entity_id or entity_id not in self._circuit_breakers:
            self._logger.warning(f"Nenhum circuit breaker registrado para entidade {entity_id}")
            return
        
        config, current_state, last_updated = self._circuit_breakers[entity_id]
        
        # Verificar se a métrica está sendo monitorada por este circuit breaker
        if config.metrics_to_monitor and event.metric.type not in config.metrics_to_monitor:
            return
        
        # Atualizar contadores de falha apenas se o circuit breaker estiver fechado ou meio-aberto
        if current_state in (CircuitBreakerState.CLOSED, CircuitBreakerState.HALF_OPEN):
            if event.is_critical:
                # Evento crítico abre o circuit breaker imediatamente
                self._open_circuit_breaker(entity_id, event)
            else:
                # Incrementar contador de falhas e verificar se atingiu o limite
                self._breaker_failure_counts[entity_id] += 1
                self._logger.info(
                    f"Circuit breaker '{config.name}' para {entity_id} registrou falha "
                    f"{self._breaker_failure_counts[entity_id]}/{config.failure_threshold}"
                )
                
                if self._breaker_failure_counts[entity_id] >= config.failure_threshold:
                    self._open_circuit_breaker(entity_id, event)
    
    def _open_circuit_breaker(self, entity_id: str, event: RiskThresholdBreachedEvent) -> None:
        """
        Abre um circuit breaker, interrompendo negociações para a entidade.
        
        Args:
            entity_id: Identificador da entidade
            event: Evento de violação que causou a abertura
        """
        config, _, _ = self._circuit_breakers[entity_id]
        self._circuit_breakers[entity_id] = (config, CircuitBreakerState.OPEN, datetime.utcnow())
        self._breaker_failure_counts[entity_id] = 0
        self._breaker_success_counts[entity_id] = 0
        self._last_events[entity_id] = event
        
        self._logger.warning(
            f"Circuit breaker '{config.name}' ABERTO para {entity_id} devido a violação de limite: "
            f"{event.get_message()}"
        )
        
        # Enviar notificação de alerta
        self._alert_notifier.send_circuit_breaker_alert(
            entity_id=entity_id,
            breaker_name=config.name,
            event=event,
            state=CircuitBreakerState.OPEN,
        )
    
    def record_success(self, entity_id: str) -> None:
        """
        Registra uma operação bem-sucedida para um circuit breaker no estado meio-aberto.
        
        Args:
            entity_id: Identificador da entidade
        """
        if entity_id not in self._circuit_breakers:
            return
        
        config, current_state, _ = self._circuit_breakers[entity_id]
        
        if current_state == CircuitBreakerState.HALF_OPEN:
            self._breaker_success_counts[entity_id] += 1
            self._logger.info(
                f"Circuit breaker '{config.name}' para {entity_id} registrou sucesso "
                f"{self._breaker_success_counts[entity_id]}/{config.consecutive_success_to_close}"
            )
            
            if self._breaker_success_counts[entity_id] >= config.consecutive_success_to_close:
                self._close_circuit_breaker(entity_id)
    
    def _close_circuit_breaker(self, entity_id: str) -> None:
        """
        Fecha um circuit breaker, permitindo negociações para a entidade.
        
        Args:
            entity_id: Identificador da entidade
        """
        if entity_id not in self._circuit_breakers:
            return
        
        config, old_state, _ = self._circuit_breakers[entity_id]
        self._circuit_breakers[entity_id] = (config, CircuitBreakerState.CLOSED, datetime.utcnow())
        self._breaker_failure_counts[entity_id] = 0
        self._breaker_success_counts[entity_id] = 0
        
        self._logger.info(f"Circuit breaker '{config.name}' FECHADO para {entity_id}")
        
        # Enviar notificação de resolução
        if old_state != CircuitBreakerState.CLOSED and entity_id in self._last_events:
            self._alert_notifier.send_circuit_breaker_alert(
                entity_id=entity_id,
                breaker_name=config.name,
                event=self._last_events[entity_id],
                state=CircuitBreakerState.CLOSED,
                resolution_message="Circuit breaker restaurado após sucesso em operações de teste",
            )
    
    def try_transition_to_half_open(self, entity_id: str) -> bool:
        """
        Tenta transicionar um circuit breaker do estado aberto para meio-aberto.
        
        Args:
            entity_id: Identificador da entidade
            
        Returns:
            True se a transição foi bem-sucedida, False caso contrário
        """
        if entity_id not in self._circuit_breakers:
            return False
        
        config, current_state, last_updated = self._circuit_breakers[entity_id]
        
        if current_state != CircuitBreakerState.OPEN:
            return False
        
        # Verificar se passou tempo suficiente desde a abertura
        elapsed = datetime.utcnow() - last_updated
        if elapsed.total_seconds() < (config.cooldown_period_minutes * 60):
            return False
        
        # Transicionar para meio-aberto
        self._circuit_breakers[entity_id] = (config, CircuitBreakerState.HALF_OPEN, datetime.utcnow())
        self._logger.info(
            f"Circuit breaker '{config.name}' para {entity_id} transitou para MEIO-ABERTO "
            f"após período de cooldown de {config.cooldown_period_minutes} minutos"
        )
        return True
    
    def reset(self, entity_id: str) -> bool:
        """
        Realiza reset manual em um circuit breaker, colocando-o no estado meio-aberto.
        
        Args:
            entity_id: Identificador da entidade
            
        Returns:
            True se o reset foi bem-sucedido, False caso contrário
        """
        if entity_id not in self._circuit_breakers:
            return False
        
        config, current_state, _ = self._circuit_breakers[entity_id]
        
        if current_state != CircuitBreakerState.OPEN:
            return False
        
        # Eventos críticos podem exigir reset manual
        last_event = self._last_events.get(entity_id)
        if (last_event and last_event.is_critical and config.requires_manual_reset_if_critical):
            self._circuit_breakers[entity_id] = (config, CircuitBreakerState.HALF_OPEN, datetime.utcnow())
            self._breaker_failure_counts[entity_id] = 0
            self._breaker_success_counts[entity_id] = 0
            
            self._logger.info(
                f"Circuit breaker '{config.name}' para {entity_id} MANUALMENTE transitou para MEIO-ABERTO"
            )
            return True
        
        return False
    
    def check_auto_reset(self) -> List[str]:
        """
        Verifica todos os circuit breakers abertos e aplica auto-reset conforme configuração.
        Este método deve ser chamado periodicamente.
        
        Returns:
            Lista de IDs de entidades que foram automaticamente resetadas
        """
        reset_ids = []
        now = datetime.utcnow()
        
        for entity_id, (config, state, last_updated) in list(self._circuit_breakers.items()):
            if state == CircuitBreakerState.OPEN and config.auto_reset:
                elapsed = now - last_updated
                
                # Verificar se passou o tempo de recuperação automática
                if elapsed.total_seconds() >= (config.recovery_timeout_minutes * 60):
                    # Eventos críticos podem exigir reset manual mesmo com auto_reset habilitado
                    last_event = self._last_events.get(entity_id)
                    if not (last_event and last_event.is_critical and config.requires_manual_reset_if_critical):
                        self._circuit_breakers[entity_id] = (config, CircuitBreakerState.HALF_OPEN, now)
                        self._logger.info(
                            f"Circuit breaker '{config.name}' para {entity_id} AUTO-RESETADO para MEIO-ABERTO "
                            f"após {config.recovery_timeout_minutes} minutos"
                        )
                        reset_ids.append(entity_id)
        
        return reset_ids
    
    def get_all_circuit_breakers(self) -> Dict[str, Dict]:
        """
        Retorna informações sobre todos os circuit breakers registrados.
        
        Returns:
            Dicionário com informações de todos os circuit breakers
        """
        result = {}
        for entity_id, (config, state, last_updated) in self._circuit_breakers.items():
            result[entity_id] = {
                "name": config.name,
                "state": state.value,
                "last_updated": last_updated.isoformat(),
                "failure_count": self._breaker_failure_counts.get(entity_id, 0),
                "success_count": self._breaker_success_counts.get(entity_id, 0),
                "last_event": self._last_events.get(entity_id).get_message() 
                            if entity_id in self._last_events else None,
            }
        return result