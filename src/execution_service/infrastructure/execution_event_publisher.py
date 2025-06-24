"""
Publisher para eventos de execução.

Este módulo implementa o publisher para eventos de execução de ordens,
permitindo que outros componentes do sistema reajam a esses eventos.
"""
import json
import logging
import uuid
from typing import Optional

import confluent_kafka
from confluent_kafka import Producer

from execution_service.domain.entities.execution import Execution
from execution_service.domain.entities.order import Order
from execution_service.domain.events.order_created_event import OrderCreatedEvent
from execution_service.domain.events.order_executed_event import OrderExecutedEvent

logger = logging.getLogger(__name__)


class ExecutionEventPublisher:
    """
    Publisher para eventos de execução.
    
    Esta classe é responsável por publicar eventos relacionados à execução
    de ordens em tópicos Kafka, permitindo a comunicação assíncrona com
    outros componentes do sistema.
    """
    
    def __init__(self, kafka_config: dict):
        """
        Inicializa o publisher.
        
        Args:
            kafka_config: Configuração do Kafka.
        """
        self.producer = Producer(kafka_config)
        self.order_created_topic = kafka_config.get('order_created_topic', 'order-created')
        self.order_executed_topic = kafka_config.get('order_executed_topic', 'order-executed')
    
    def publish_order_created(self, order: Order, user_id: Optional[str] = None) -> bool:
        """
        Publica um evento de criação de ordem.
        
        Args:
            order: A ordem criada.
            user_id: ID do usuário que criou a ordem (opcional).
            
        Returns:
            bool: True se o evento foi publicado com sucesso, False caso contrário.
        """
        try:
            # Criar evento
            event_id = str(uuid.uuid4())
            event = OrderCreatedEvent.from_order(order, event_id, user_id)
            
            # Serializar para JSON
            event_json = event.to_json()
            
            # Publicar no tópico
            self.producer.produce(
                self.order_created_topic,
                key=order.id,
                value=event_json,
                callback=self._delivery_report
            )
            
            # Garantir que a mensagem seja enviada
            self.producer.flush(timeout=10)
            
            logger.info(f"Evento de criação de ordem publicado: {order.id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao publicar evento de criação de ordem: {str(e)}")
            return False
    
    def publish_order_executed(
        self, execution: Execution, order: Order, user_id: Optional[str] = None
    ) -> bool:
        """
        Publica um evento de execução de ordem.
        
        Args:
            execution: A execução da ordem.
            order: A ordem executada.
            user_id: ID do usuário que criou a ordem (opcional).
            
        Returns:
            bool: True se o evento foi publicado com sucesso, False caso contrário.
        """
        try:
            # Criar evento
            event_id = str(uuid.uuid4())
            event = OrderExecutedEvent.from_execution(execution, order, event_id, user_id)
            
            # Serializar para JSON
            event_json = event.to_json()
            
            # Publicar no tópico
            self.producer.produce(
                self.order_executed_topic,
                key=order.id,
                value=event_json,
                callback=self._delivery_report
            )
            
            # Garantir que a mensagem seja enviada
            self.producer.flush(timeout=10)
            
            logger.info(f"Evento de execução de ordem publicado: {order.id}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao publicar evento de execução de ordem: {str(e)}")
            return False
    
    def _delivery_report(self, err, msg):
        """
        Callback para relatar o status de entrega da mensagem.
        
        Args:
            err: Erro, se houver.
            msg: Mensagem enviada.
        """
        if err is not None:
            logger.error(f"Falha na entrega da mensagem: {str(err)}")
        else:
            logger.debug(f"Mensagem entregue ao tópico {msg.topic()} [{msg.partition()}]")
    
    def close(self):
        """
        Fecha o producer Kafka.
        """
        self.producer.flush(timeout=10)
        logger.info("Producer Kafka fechado")


class InMemoryEventPublisher(ExecutionEventPublisher):
    """
    Implementação em memória do publisher de eventos para uso em testes.
    """
    
    def __init__(self):
        """
        Inicializa o publisher em memória.
        """
        self.order_created_events = []
        self.order_executed_events = []
    
    def publish_order_created(self, order: Order, user_id: Optional[str] = None) -> bool:
        """
        Publica um evento de criação de ordem em memória.
        
        Args:
            order: A ordem criada.
            user_id: ID do usuário que criou a ordem (opcional).
            
        Returns:
            bool: True, sempre.
        """
        event_id = str(uuid.uuid4())
        event = OrderCreatedEvent.from_order(order, event_id, user_id)
        self.order_created_events.append(event)
        logger.info(f"Evento de criação de ordem armazenado em memória: {order.id}")
        return True
    
    def publish_order_executed(
        self, execution: Execution, order: Order, user_id: Optional[str] = None
    ) -> bool:
        """
        Publica um evento de execução de ordem em memória.
        
        Args:
            execution: A execução da ordem.
            order: A ordem executada.
            user_id: ID do usuário que criou a ordem (opcional).
            
        Returns:
            bool: True, sempre.
        """
        event_id = str(uuid.uuid4())
        event = OrderExecutedEvent.from_execution(execution, order, event_id, user_id)
        self.order_executed_events.append(event)
        logger.info(f"Evento de execução de ordem armazenado em memória: {order.id}")
        return True
    
    def clear(self):
        """
        Limpa todos os eventos armazenados.
        """
        self.order_created_events = []
        self.order_executed_events = []
    
    def close(self):
        """
        Método vazio para compatibilidade com a interface.
        """
        pass