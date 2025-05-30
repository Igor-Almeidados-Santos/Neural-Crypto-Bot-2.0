"""
Entidade Order para o domínio de execução.

Este módulo define a entidade Order, que representa uma ordem de trading
no domínio de execução.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class Order(BaseModel):
    """
    Entidade que representa uma ordem de trading.
    
    Uma ordem contém todas as informações necessárias para executar
    uma operação de compra ou venda em uma exchange.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trading_pair: str
    side: str  # 'buy' ou 'sell'
    order_type: str  # 'market', 'limit', etc.
    quantity: float
    price: Optional[float] = None
    status: str = "pending"  # 'pending', 'filled', 'partial', 'cancelled', 'failed'
    filled_quantity: float = 0
    average_price: Optional[float] = None
    fees: float = 0
    exchange: str
    exchange_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    child_orders: List["Order"] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('side')
    def validate_side(cls, v):
        """Valida que o lado da ordem é 'buy' ou 'sell'."""
        if v not in ['buy', 'sell']:
            raise ValueError(f"Side deve ser 'buy' ou 'sell', recebido: {v}")
        return v
    
    @validator('order_type')
    def validate_order_type(cls, v):
        """Valida que o tipo da ordem é um dos tipos suportados."""
        valid_types = ['market', 'limit', 'stop_limit', 'stop_market', 'trailing_stop']
        if v not in valid_types:
            raise ValueError(f"Order type deve ser um dos: {valid_types}, recebido: {v}")
        return v
    
    @validator('status')
    def validate_status(cls, v):
        """Valida que o status da ordem é um dos status suportados."""
        valid_statuses = ['pending', 'filled', 'partial', 'cancelled', 'failed']
        if v not in valid_statuses:
            raise ValueError(f"Status deve ser um dos: {valid_statuses}, recebido: {v}")
        return v
    
    @validator('price', always=True)
    def validate_price(cls, v, values):
        """Valida que o preço está presente para ordens do tipo limit."""
        if values.get('order_type') == 'limit' and v is None:
            raise ValueError("Preço é obrigatório para ordens do tipo limit")
        return v
    
    def update_status(self, new_status: str, filled_quantity: Optional[float] = None, 
                     average_price: Optional[float] = None, fees: Optional[float] = None,
                     exchange_order_id: Optional[str] = None, error_message: Optional[str] = None):
        """
        Atualiza o status da ordem e outros campos relacionados.
        
        Args:
            new_status: Novo status da ordem.
            filled_quantity: Quantidade preenchida (opcional).
            average_price: Preço médio de execução (opcional).
            fees: Taxas pagas (opcional).
            exchange_order_id: ID da ordem na exchange (opcional).
            error_message: Mensagem de erro, se houver (opcional).
        """
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        if filled_quantity is not None:
            self.filled_quantity = filled_quantity
            
        if average_price is not None:
            self.average_price = average_price
            
        if fees is not None:
            self.fees = fees
            
        if exchange_order_id is not None:
            self.exchange_order_id = exchange_order_id
            
        if error_message is not None:
            self.error_message = error_message
            
        if new_status in ['filled', 'partial']:
            self.executed_at = datetime.utcnow()
    
    def get_notional_value(self) -> float:
        """
        Calcula o valor nocional da ordem.
        
        Returns:
            float: O valor nocional (quantidade * preço).
        """
        price = self.average_price if self.status in ['filled', 'partial'] else self.price
        if price is None:
            return 0
        return self.quantity * price
    
    def get_filled_notional_value(self) -> float:
        """
        Calcula o valor nocional preenchido da ordem.
        
        Returns:
            float: O valor nocional preenchido (quantidade preenchida * preço médio).
        """
        if self.average_price is None:
            return 0
        return self.filled_quantity * self.average_price
    
    def get_execution_progress(self) -> float:
        """
        Calcula o progresso da execução da ordem.
        
        Returns:
            float: Percentual de execução (0-1).
        """
        if self.quantity == 0:
            return 0
        return self.filled_quantity / self.quantity
    
    def is_complete(self) -> bool:
        """
        Verifica se a ordem está completa (preenchida ou cancelada/falha).
        
        Returns:
            bool: True se a ordem estiver completa, False caso contrário.
        """
        return self.status in ['filled', 'cancelled', 'failed']
    
    def is_active(self) -> bool:
        """
        Verifica se a ordem está ativa (pendente ou parcial).
        
        Returns:
            bool: True se a ordem estiver ativa, False caso contrário.
        """
        return self.status in ['pending', 'partial']
    
    def add_child_order(self, child_order: "Order"):
        """
        Adiciona uma ordem filha a esta ordem.
        
        Args:
            child_order: A ordem filha a ser adicionada.
        """
        self.child_orders.append(child_order)
        
    def update_from_child_orders(self):
        """
        Atualiza os campos da ordem baseado nas suas ordens filhas.
        
        Útil para ordens que foram divididas em várias sub-ordens.
        """
        if not self.child_orders:
            return
            
        total_filled = sum(order.filled_quantity for order in self.child_orders)
        self.filled_quantity = total_filled
        
        # Calcular preço médio ponderado
        filled_value = sum(order.filled_quantity * (order.average_price or 0) 
                           for order in self.child_orders if order.average_price is not None)
        
        if total_filled > 0:
            self.average_price = filled_value / total_filled
            
        # Somar taxas
        self.fees = sum(order.fees for order in self.child_orders)
        
        # Atualizar status
        if total_filled >= self.quantity:
            self.status = "filled"
            self.executed_at = datetime.utcnow()
        elif total_filled > 0:
            self.status = "partial"
            self.executed_at = datetime.utcnow()
        elif any(order.status == "failed" for order in self.child_orders):
            self.status = "failed"
        
        self.updated_at = datetime.utcnow()
        
    def to_dict(self) -> Dict:
        """
        Converte a ordem para um dicionário.
        
        Returns:
            Dict: Representação da ordem como dicionário.
        """
        return {
            "id": self.id,
            "trading_pair": self.trading_pair,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "fees": self.fees,
            "exchange": self.exchange,
            "exchange_order_id": self.exchange_order_id,
            "strategy_id": self.strategy_id,
            "parent_order_id": self.parent_order_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "child_orders_count": len(self.child_orders),
        }