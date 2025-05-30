"""
Evento OrderCreatedEvent para o domínio de execução.

Este módulo define o evento OrderCreatedEvent, que é disparado
quando uma nova ordem é criada no sistema.
"""
import json
from datetime import datetime
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.execution_service.domain.entities.order import Order


class OrderCreatedEvent(BaseModel):
    """
    Evento que representa a criação de uma nova ordem.
    
    Este evento é disparado quando uma nova ordem é criada no sistema,
    permitindo que outros componentes reajam a esta ação.
    """
    
    event_id: str = Field(...)
    event_type: str = "order_created"
    event_timestamp: datetime = Field(default_factory=datetime.utcnow)
    order_id: str
    trading_pair: str
    side: str
    order_type: str
    quantity: float
    price: Optional[float]
    exchange: str
    strategy_id: Optional[str]
    user_id: Optional[str]
    additional_data: Dict = Field(default_factory=dict)
    
    @classmethod
    def from_order(cls, order: Order, event_id: str, user_id: Optional[str] = None) -> 'OrderCreatedEvent':
        """
        Cria um evento a partir de uma ordem.
        
        Args:
            order: A ordem que foi criada.
            event_id: ID único do evento.
            user_id: ID do usuário que criou a ordem (opcional).
            
        Returns:
            OrderCreatedEvent: Evento de criação de ordem.
        """
        additional_data = {}
        if order.metadata:
            additional_data.update(order.metadata)
        
        return cls(
            event_id=event_id,
            order_id=order.id,
            trading_pair=order.trading_pair,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            exchange=order.exchange,
            strategy_id=order.strategy_id,
            user_id=user_id,
            additional_data=additional_data,
        )
    
    def to_json(self) -> str:
        """
        Converte o evento para uma string JSON.
        
        Returns:
            str: Representação JSON do evento.
        """
        return json.dumps(self.dict(), default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'OrderCreatedEvent':
        """
        Cria um evento a partir de uma string JSON.
        
        Args:
            json_str: String JSON representando o evento.
            
        Returns:
            OrderCreatedEvent: Evento de criação de ordem.
        """
        data = json.loads(json_str)
        # Converter timestamp de string para datetime
        if 'event_timestamp' in data and isinstance(data['event_timestamp'], str):
            data['event_timestamp'] = datetime.fromisoformat(data['event_timestamp'].replace('Z', '+00:00'))
        return cls(**data)