import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import httpx

from src.risk_management.application.services.circuit_breaker_service import CircuitBreakerState
from src.risk_management.domain.entities.risk_profile import RiskLevel
from src.risk_management.domain.events.risk_threshold_breached_event import RiskThresholdBreachedEvent


class AlertChannel(Enum):
    """Canais disponíveis para envio de alertas."""
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    SMS = "sms"
    WEBHOOK = "webhook"


class AlertPriority(Enum):
    """Níveis de prioridade para alertas."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertNotifier:
    """
    Serviço responsável por enviar notificações e alertas quando limites de risco são violados.
    """
    
    def __init__(
        self,
        channels: Dict[AlertChannel, Dict[str, str]],
        min_priority: AlertPriority = AlertPriority.LOW,
        rate_limit_minutes: int = 5,
        include_metrics: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o serviço de notificação.
        
        Args:
            channels: Configuração dos canais de notificação
            min_priority: Prioridade mínima para envio de alertas
            rate_limit_minutes: Intervalo mínimo entre alertas (rate limiting)
            include_metrics: Se deve incluir métricas detalhadas
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._channels = channels
        self._min_priority = min_priority
        self._rate_limit_minutes = rate_limit_minutes
        self._include_metrics = include_metrics
        self._last_alerts: Dict[str, datetime] = {}  # Para rate limiting
    
    def send_threshold_breach_alert(
        self,
        event: RiskThresholdBreachedEvent,
        priority: AlertPriority = None,
        additional_info: Dict[str, str] = None
    ) -> bool:
        """
        Envia alerta quando um limite de risco é violado.
        
        Args:
            event: Evento de violação de limite
            priority: Prioridade do alerta (derivada do evento se None)
            additional_info: Informações adicionais para incluir no alerta
            
        Returns:
            True se o alerta foi enviado com sucesso
        """
        # Determinar prioridade com base no evento se não fornecida
        if priority is None:
            if event.is_critical:
                priority = AlertPriority.CRITICAL
            elif event.risk_level == RiskLevel.HIGH:
                priority = AlertPriority.HIGH
            elif event.risk_level == RiskLevel.MEDIUM:
                priority = AlertPriority.MEDIUM
            else:
                priority = AlertPriority.LOW
        
        # Verificar se a prioridade atende ao mínimo configurado
        if self._get_priority_value(priority) < self._get_priority_value(self._min_priority):
            return False
        
        # Aplicar rate limiting por entidade
        entity_id = event.portfolio_id or event.strategy_id or event.asset_id or "unknown"
        rate_limit_key = f"{entity_id}_{event.metric.type.name}"
        
        if rate_limit_key in self._last_alerts:
            minutes_since_last = (datetime.utcnow() - self._last_alerts[rate_limit_key]).total_seconds() / 60
            if minutes_since_last < self._rate_limit_minutes:
                self._logger.debug(
                    f"Rate limiting aplicado para alerta {rate_limit_key}: "
                    f"{minutes_since_last:.1f} min < {self._rate_limit_minutes} min"
                )
                return False
        
        # Atualizar timestamp do último alerta
        self._last_alerts[rate_limit_key] = datetime.utcnow()
        
        # Preparar conteúdo do alerta
        alert_content = {
            "title": f"Alerta de Risco - {event.metric.type.name}",
            "message": event.get_message(),
            "priority": priority.value,
            "timestamp": event.timestamp.isoformat(),
            "entity_type": "portfolio" if event.portfolio_id else "strategy" if event.strategy_id else "asset",
            "entity_id": event.portfolio_id or event.strategy_id or event.asset_id or "unknown",
            "risk_level": event.risk_level.value,
            "is_critical": event.is_critical,
            "threshold_value": event.threshold_value,
            "actual_value": event.actual_value,
        }
        
        # Adicionar métricas detalhadas se configurado
        if self._include_metrics:
            alert_content["metric"] = {
                "type": event.metric.type.name,
                "value": event.metric.value,
                "timestamp": event.metric.timestamp.isoformat(),
                "confidence_level": event.metric.confidence_level,
                "metadata": event.metric.metadata or {},
            }
        
        # Adicionar informações adicionais se fornecidas
        if additional_info:
            alert_content["additional_info"] = additional_info
        
        # Enviar para todos os canais configurados
        success = False
        for channel, config in self._channels.items():
            try:
                sent = self._send_to_channel(channel, config, alert_content)
                success = success or sent
            except Exception as e:
                self._logger.error(f"Erro ao enviar alerta para {channel.value}: {e}")
        
        return success
    
    def send_circuit_breaker_alert(
        self,
        entity_id: str,
        breaker_name: str,
        event: RiskThresholdBreachedEvent,
        state: CircuitBreakerState,
        resolution_message: Optional[str] = None,
        priority: Optional[AlertPriority] = None
    ) -> bool:
        """
        Envia alerta quando um circuit breaker muda de estado.
        
        Args:
            entity_id: Identificador da entidade
            breaker_name: Nome do circuit breaker
            event: Evento que causou a mudança de estado
            state: Novo estado do circuit breaker
            resolution_message: Mensagem de resolução (para fechamento)
            priority: Prioridade do alerta
            
        Returns:
            True se o alerta foi enviado com sucesso
        """
        # Determinar prioridade com base no estado se não fornecida
        if priority is None:
            if state == CircuitBreakerState.OPEN:
                priority = AlertPriority.CRITICAL
            elif state == CircuitBreakerState.HALF_OPEN:
                priority = AlertPriority.HIGH
            else:
                priority = AlertPriority.MEDIUM
        
        # Verificar se a prioridade atende ao mínimo configurado
        if self._get_priority_value(priority) < self._get_priority_value(self._min_priority):
            return False
        
        # Aplicar rate limiting
        rate_limit_key = f"cb_{entity_id}_{state.value}"
        
        if rate_limit_key in self._last_alerts:
            minutes_since_last = (datetime.utcnow() - self._last_alerts[rate_limit_key]).total_seconds() / 60
            if minutes_since_last < self._rate_limit_minutes:
                return False
        
        # Atualizar timestamp do último alerta
        self._last_alerts[rate_limit_key] = datetime.utcnow()
        
        # Preparar título e mensagem com base no estado
        if state == CircuitBreakerState.OPEN:
            title = f"ALERTA: Circuit Breaker ATIVADO - {breaker_name}"
            message = (
                f"Circuit breaker '{breaker_name}' para {entity_id} foi ABERTO devido a: "
                f"{event.get_message()}"
            )
        elif state == CircuitBreakerState.HALF_OPEN:
            title = f"Circuit Breaker em Teste - {breaker_name}"
            message = (
                f"Circuit breaker '{breaker_name}' para {entity_id} está em modo de teste. "
                f"Operações serão permitidas em volume limitado."
            )
        else:  # CLOSED
            title = f"Circuit Breaker Desativado - {breaker_name}"
            message = (
                f"Circuit breaker '{breaker_name}' para {entity_id} foi FECHADO. "
                f"Operações normais foram restauradas."
            )
            
            if resolution_message:
                message += f" {resolution_message}"
        
        # Preparar conteúdo do alerta
        alert_content = {
            "title": title,
            "message": message,
            "priority": priority.value,
            "timestamp": datetime.utcnow().isoformat(),
            "entity_id": entity_id,
            "circuit_breaker": {
                "name": breaker_name,
                "state": state.value,
                "triggered_by": {
                    "metric_type": event.metric.type.name,
                    "threshold_value": event.threshold_value,
                    "actual_value": event.actual_value,
                    "timestamp": event.timestamp.isoformat(),
                }
            }
        }
        
        # Adicionar mensagem de resolução se disponível
        if resolution_message:
            alert_content["resolution_message"] = resolution_message
        
        # Enviar para todos os canais configurados
        success = False
        for channel, config in self._channels.items():
            try:
                sent = self._send_to_channel(channel, config, alert_content)
                success = success or sent
            except Exception as e:
                self._logger.error(f"Erro ao enviar alerta de circuit breaker para {channel.value}: {e}")
        
        return success
    
    def send_exposure_alert(
        self,
        entity_id: str,
        entity_type: str,
        current_value: float,
        max_allowed: float,
        percentage_used: float,
        is_critical: bool,
        additional_info: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Envia alerta quando limites de exposição são violados.
        
        Args:
            entity_id: Identificador da entidade (ativo ou categoria)
            entity_type: Tipo da entidade ("asset" ou "category")
            current_value: Valor atual da exposição
            max_allowed: Valor máximo permitido
            percentage_used: Percentual do limite utilizado
            is_critical: Se é uma violação crítica
            additional_info: Informações adicionais
            
        Returns:
            True se o alerta foi enviado com sucesso
        """
        # Determinar prioridade
        priority = AlertPriority.CRITICAL if is_critical else AlertPriority.HIGH
        
        # Verificar se a prioridade atende ao mínimo configurado
        if self._get_priority_value(priority) < self._get_priority_value(self._min_priority):
            return False
        
        # Aplicar rate limiting
        rate_limit_key = f"exposure_{entity_type}_{entity_id}"
        
        if rate_limit_key in self._last_alerts:
            minutes_since_last = (datetime.utcnow() - self._last_alerts[rate_limit_key]).total_seconds() / 60
            if minutes_since_last < self._rate_limit_minutes:
                return False
        
        # Atualizar timestamp do último alerta
        self._last_alerts[rate_limit_key] = datetime.utcnow()
        
        # Preparar título e mensagem
        severity = "CRÍTICO" if is_critical else "ALERTA"
        title = f"{severity}: Limite de Exposição Excedido - {entity_type.title()} {entity_id}"
        message = (
            f"Exposição para {entity_type} '{entity_id}' excedeu limite: "
            f"Atual: {current_value:.2f}, Máximo: {max_allowed:.2f} "
            f"({percentage_used:.1f}% do limite)"
        )
        
        # Preparar conteúdo do alerta
        alert_content = {
            "title": title,
            "message": message,
            "priority": priority.value,
            "timestamp": datetime.utcnow().isoformat(),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "exposure": {
                "current_value": current_value,
                "max_allowed": max_allowed,
                "percentage_used": percentage_used,
                "is_critical": is_critical,
            }
        }
        
        # Adicionar informações adicionais se fornecidas
        if additional_info:
            alert_content["additional_info"] = additional_info
        
        # Enviar para todos os canais configurados
        success = False
        for channel, config in self._channels.items():
            try:
                sent = self._send_to_channel(channel, config, alert_content)
                success = success or sent
            except Exception as e:
                self._logger.error(f"Erro ao enviar alerta de exposição para {channel.value}: {e}")
        
        return success
    
    def _send_to_channel(
        self,
        channel: AlertChannel,
        config: Dict[str, str],
        content: Dict[str, Union[str, float, bool, Dict]]
    ) -> bool:
        """
        Envia alerta para um canal específico.
        
        Args:
            channel: Canal de notificação
            config: Configuração do canal
            content: Conteúdo do alerta
            
        Returns:
            True se o envio foi bem-sucedido
        """
        if channel == AlertChannel.EMAIL:
            return self._send_email(config, content)
        elif channel == AlertChannel.SLACK:
            return self._send_slack(config, content)
        elif channel == AlertChannel.TELEGRAM:
            return self._send_telegram(config, content)
        elif channel == AlertChannel.SMS:
            return self._send_sms(config, content)
        elif channel == AlertChannel.WEBHOOK:
            return self._send_webhook(config, content)
        else:
            self._logger.warning(f"Canal de notificação não suportado: {channel}")
            return False
    
    def _send_email(self, config: Dict[str, str], content: Dict[str, Union[str, float, bool, Dict]]) -> bool:
        """
        Envia alerta por email.
        Implementação simplificada - em produção, usar biblioteca SMTP adequada.
        
        Args:
            config: Configuração do email
            content: Conteúdo do alerta
            
        Returns:
            True se o envio foi bem-sucedido
        """
        # Implementação simplificada - em produção, usar biblioteca SMTP
        # como smtplib ou serviço como SendGrid, Mailgun, etc.
        self._logger.info(f"[EMAIL] Enviando alerta para {config.get('recipients')}: {content['title']}")
        return True
    
    def _send_slack(self, config: Dict[str, str], content: Dict[str, Union[str, float, bool, Dict]]) -> bool:
        """
        Envia alerta para Slack.
        
        Args:
            config: Configuração do Slack
            content: Conteúdo do alerta
            
        Returns:
            True se o envio foi bem-sucedido
        """
        webhook_url = config.get("webhook_url")
        if not webhook_url:
            self._logger.error("URL do webhook do Slack não configurada")
            return False
        
        # Formatar mensagem para Slack
        priority_emoji = {
            "low": ":information_source:",
            "medium": ":warning:",
            "high": ":rotating_light:",
            "critical": ":fire:"
        }
        
        emoji = priority_emoji.get(content["priority"], ":warning:")
        
        slack_message = {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} {content['title']}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": content["message"]
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Prioridade:* {content['priority'].upper()} | *Timestamp:* {content['timestamp']}"
                        }
                    ]
                }
            ]
        }
        
        # Adicionar detalhes adicionais se disponíveis
        details = {}
        for key, value in content.items():
            if key not in ["title", "message", "priority", "timestamp"] and isinstance(value, (str, int, float, bool)):
                details[key] = value
        
        if details:
            details_text = "\n".join([f"*{k.replace('_', ' ').title()}:* {v}" for k, v in details.items()])
            slack_message["blocks"].append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Detalhes:*\n{details_text}"
                }
            })
        
        try:
            response = httpx.post(
                webhook_url,
                json=slack_message,
                timeout=5.0
            )
            if response.status_code == 200:
                return True
            else:
                self._logger.error(f"Erro ao enviar para Slack: {response.status_code} {response.text}")
                return False
        except Exception as e:
            self._logger.error(f"Exceção ao enviar para Slack: {e}")
            return False
    
    def _send_telegram(self, config: Dict[str, str], content: Dict[str, Union[str, float, bool, Dict]]) -> bool:
        """
        Envia alerta para Telegram.
        
        Args:
            config: Configuração do Telegram
            content: Conteúdo do alerta
            
        Returns:
            True se o envio foi bem-sucedido
        """
        bot_token = config.get("bot_token")
        chat_id = config.get("chat_id")
        
        if not bot_token or not chat_id:
            self._logger.error("Token do bot ou ID do chat do Telegram não configurados")
            return False
        
        # Formatar mensagem para Telegram
        priority_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🚨",
            "critical": "🔥"
        }
        
        emoji = priority_emoji.get(content["priority"], "⚠️")
        
        message_text = (
            f"{emoji} <b>{content['title']}</b>\n\n"
            f"{content['message']}\n\n"
            f"<b>Prioridade:</b> {content['priority'].upper()}\n"
            f"<b>Timestamp:</b> {content['timestamp']}"
        )
        
        # Adicionar detalhes adicionais
        for key, value in content.items():
            if key not in ["title", "message", "priority", "timestamp"] and isinstance(value, (str, int, float, bool)):
                message_text += f"\n<b>{key.replace('_', ' ').title()}:</b> {value}"
        
        params = {
            "chat_id": chat_id,
            "text": message_text,
            "parse_mode": "HTML"
        }
        
        try:
            response = httpx.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                params=params,
                timeout=5.0
            )
            if response.status_code == 200:
                return True
            else:
                self._logger.error(f"Erro ao enviar para Telegram: {response.status_code} {response.text}")
                return False
        except Exception as e:
            self._logger.error(f"Exceção ao enviar para Telegram: {e}")
            return False
    
    def _send_sms(self, config: Dict[str, str], content: Dict[str, Union[str, float, bool, Dict]]) -> bool:
        """
        Envia alerta por SMS.
        Implementação simplificada - em produção, usar provedor adequado.
        
        Args:
            config: Configuração do SMS
            content: Conteúdo do alerta
            
        Returns:
            True se o envio foi bem-sucedido
        """
        # Implementação simplificada - em produção, usar provedor como Twilio, Vonage, etc.
        self._logger.info(f"[SMS] Enviando alerta para {config.get('phone_numbers')}: {content['title']}")
        return True
    
    def _send_webhook(self, config: Dict[str, str], content: Dict[str, Union[str, float, bool, Dict]]) -> bool:
        """
        Envia alerta para webhook genérico.
        
        Args:
            config: Configuração do webhook
            content: Conteúdo do alerta
            
        Returns:
            True se o envio foi bem-sucedido
        """
        webhook_url = config.get("url")
        if not webhook_url:
            self._logger.error("URL do webhook não configurada")
            return False
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Adicionar cabeçalhos adicionais se configurados
        if "headers" in config:
            try:
                additional_headers = json.loads(config["headers"])
                headers.update(additional_headers)
            except json.JSONDecodeError:
                self._logger.error("Cabeçalhos adicionais do webhook inválidos")
        
        try:
            response = httpx.post(
                webhook_url,
                json=content,
                headers=headers,
                timeout=5.0
            )
            if response.status_code >= 200 and response.status_code < 300:
                return True
            else:
                self._logger.error(f"Erro ao enviar para webhook: {response.status_code} {response.text}")
                return False
        except Exception as e:
            self._logger.error(f"Exceção ao enviar para webhook: {e}")
            return False
    
    def _get_priority_value(self, priority: AlertPriority) -> int:
        """
        Converte prioridade em valor numérico para comparação.
        
        Args:
            priority: Prioridade do alerta
            
        Returns:
            Valor numérico da prioridade
        """
        if priority == AlertPriority.CRITICAL:
            return 4
        elif priority == AlertPriority.HIGH:
            return 3
        elif priority == AlertPriority.MEDIUM:
            return 2
        else:  # LOW
            return 1