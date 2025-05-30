"""
Evento OrderExecutedEvent para o domínio de execução.

Este módulo define o evento OrderExecutedEvent, que é disparado
quando uma ordem é executada no sistema.
"""
import json
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.execution_service.domain.entities.execution import Execution
from src.execution_service.domain.entities.order import Order


class OrderExecutedEvent(BaseModel):
    """
    Evento que representa a execução de uma ordem.
    
    Este evento é disparado quando uma ordem é executada no sistema,
    permitindo que outros componentes reajam a esta ação.
    """
    
    event_id: str = Field(...)
    event_type: str = "order_executed"
    event_timestamp: datetime = Field(default_factory=datetime.utcnow)
    order_id: str
    execution_id: str
    trading_pair: str
    side: str
    order_type: str
    original_quantity: float
    filled_quantity: float
    average_price: Optional[float]
    fees: float
    status: str  # 'filled', 'partial', 'failed'
    exchange: str
    execution_algorithm: str
    strategy_id: Optional[str]
    user_id: Optional[str]
    child_orders_count: int = 0
    execution_time_seconds: float = 0
    metrics: Dict = Field(default_factory=dict)
    error_message: Optional[str] = None
    additional_data: Dict = Field(default_factory=dict)
    
    @classmethod
    def from_execution(
        cls, execution: Execution, order: Order, event_id: str, user_id: Optional[str] = None
    ) -> 'OrderExecutedEvent':
        """
        Cria um evento a partir de uma execução e uma ordem.
        
        Args:
            execution: A execução da ordem.
            order: A ordem que foi executada.
            event_id: ID único do evento.
            user_id: ID do usuário que criou a ordem (opcional).
            
        Returns:
            OrderExecutedEvent: Evento de execução de ordem.
        """
        execution_time = 0
        if execution.completed_at and execution.started_at:
            execution_time = (execution.completed_at - execution.started_at).total_seconds()
        
        additional_data = {}
        if order.metadata:
            additional_data.update(order.metadata)
        
        return cls(
            event_id=event_id,
            order_id=order.id,
            execution_id=execution.id,
            trading_pair=order.trading_pair,
            side=order.side,
            order_type=order.order_type,
            original_quantity=order.quantity,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
            fees=order.fees,
            status=order.status,
            exchange=order.exchange,
            execution_algorithm=execution.algorithm,
            strategy_id=order.strategy_id,
            user_id=user_id,
            child_orders_count=len(execution.sub_orders),
            execution_time_seconds=execution_time,
            metrics=execution.metrics,
            error_message=execution.message if not execution.success else None,
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
    def from_json(cls, json_str: str) -> 'OrderExecutedEvent':
        """
        Cria um evento a partir de uma string JSON.
        
        Args:
            json_str: String JSON representando o evento.
            
        Returns:
            OrderExecutedEvent: Evento de execução de ordem.
        """
        data = json.loads(json_str)
        # Converter timestamp de string para datetime
        if 'event_timestamp' in data and isinstance(data['event_timestamp'], str):
            data['event_timestamp'] = datetime.fromisoformat(data['event_timestamp'].replace('Z', '+00:00'))
        return cls(**data)