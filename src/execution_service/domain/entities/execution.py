"""
Entidade Execution para o domínio de execução.

Este módulo define a entidade Execution, que representa uma execução de ordem
no domínio de execução.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from src.execution_service.domain.entities.order import Order
from src.execution_service.domain.value_objects.execution_parameters import ExecutionParameters


class Execution(BaseModel):
    """
    Entidade que representa a execução de uma ordem.
    
    Uma execução contém todos os detalhes sobre como uma ordem foi executada,
    incluindo o algoritmo utilizado, parâmetros, resultado e estatísticas.
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    order_id: str
    algorithm: str  # 'twap', 'iceberg', 'smart_routing', etc.
    parameters: ExecutionParameters
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    status: str = "in_progress"  # 'in_progress', 'completed', 'failed', 'cancelled'
    success: bool = False
    message: Optional[str] = None
    sub_orders: List[Order] = Field(default_factory=list)
    metrics: Dict = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('algorithm')
    def validate_algorithm(cls, v):
        """Valida que o algoritmo é um dos algoritmos suportados."""
        valid_algorithms = ['twap', 'iceberg', 'smart_routing', 'vwap', 'direct']
        if v not in valid_algorithms:
            raise ValueError(f"Algorithm deve ser um dos: {valid_algorithms}, recebido: {v}")
        return v
    
    @validator('status')
    def validate_status(cls, v):
        """Valida que o status é um dos status suportados."""
        valid_statuses = ['in_progress', 'completed', 'failed', 'cancelled']
        if v not in valid_statuses:
            raise ValueError(f"Status deve ser um dos: {valid_statuses}, recebido: {v}")
        return v
    
    def complete(self, success: bool, message: Optional[str] = None, sub_orders: Optional[List[Order]] = None):
        """
        Marca a execução como completa.
        
        Args:
            success: Indica se a execução foi bem-sucedida.
            message: Mensagem de resultado (opcional).
            sub_orders: Lista de sub-ordens criadas durante a execução (opcional).
        """
        self.status = "completed"
        self.success = success
        self.message = message
        self.completed_at = datetime.utcnow()
        
        if sub_orders:
            self.sub_orders = sub_orders
    
    def fail(self, message: str):
        """
        Marca a execução como falha.
        
        Args:
            message: Mensagem de erro.
        """
        self.status = "failed"
        self.success = False
        self.message = message
        self.completed_at = datetime.utcnow()
    
    def cancel(self, message: Optional[str] = None):
        """
        Marca a execução como cancelada.
        
        Args:
            message: Mensagem de cancelamento (opcional).
        """
        self.status = "cancelled"
        self.success = False
        self.message = message or "Execução cancelada"
        self.completed_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        """
        Verifica se a execução está ativa.
        
        Returns:
            bool: True se a execução estiver em progresso, False caso contrário.
        """
        return self.status == "in_progress"
    
    def add_sub_order(self, order: Order):
        """
        Adiciona uma sub-ordem à execução.
        
        Args:
            order: A sub-ordem a ser adicionada.
        """
        self.sub_orders.append(order)
    
    def add_metric(self, key: str, value: Union[float, str, dict]):
        """
        Adiciona uma métrica à execução.
        
        Args:
            key: Nome da métrica.
            value: Valor da métrica.
        """
        self.metrics[key] = value
    
    def update_metrics(self, metrics: Dict):
        """
        Atualiza as métricas da execução.
        
        Args:
            metrics: Dicionário com métricas a serem atualizadas.
        """
        self.metrics.update(metrics)
    
    def calculate_execution_metrics(self, parent_order: Optional[Order] = None):
        """
        Calcula métricas de execução com base nas sub-ordens.
        
        Args:
            parent_order: Ordem pai, se disponível, para cálculo de métricas adicionais.
        """
        if not self.sub_orders:
            return
        
        # Calcular métricas básicas
        total_quantity = sum(order.quantity for order in self.sub_orders)
        filled_quantity = sum(order.filled_quantity for order in self.sub_orders)
        
        # Calcular preço médio ponderado
        filled_value = sum(
            order.filled_quantity * (order.average_price or 0)
            for order in self.sub_orders
            if order.average_price is not None
        )
        
        avg_price = filled_value / filled_quantity if filled_quantity > 0 else 0
        
        # Calcular taxas totais
        total_fees = sum(order.fees for order in self.sub_orders)
        
        # Calcular tempos
        if self.completed_at and self.started_at:
            execution_time_seconds = (self.completed_at - self.started_at).total_seconds()
        else:
            execution_time_seconds = 0
        
        # Calcular slippage se tivermos a ordem pai
        slippage = 0
        if parent_order and parent_order.price and avg_price > 0:
            if parent_order.side == 'buy':
                slippage = (avg_price - parent_order.price) / parent_order.price
            else:  # sell
                slippage = (parent_order.price - avg_price) / parent_order.price
            
            # Converter para percentual e limitar a precisão
            slippage = round(slippage * 100, 4)
        
        # Adicionar métricas
        metrics = {
            "total_quantity": total_quantity,
            "filled_quantity": filled_quantity,
            "fill_percentage": (filled_quantity / total_quantity * 100) if total_quantity > 0 else 0,
            "average_price": avg_price,
            "total_fees": total_fees,
            "execution_time_seconds": execution_time_seconds,
            "success_rate": len([o for o in self.sub_orders if o.status == "filled"]) / len(self.sub_orders) * 100 if self.sub_orders else 0,
            "slippage_percent": slippage,
            "num_sub_orders": len(self.sub_orders),
        }
        
        # Estatísticas por exchange (para smart routing)
        if self.algorithm == "smart_routing":
            exchange_stats = {}
            for order in self.sub_orders:
                if order.exchange not in exchange_stats:
                    exchange_stats[order.exchange] = {
                        "total_quantity": 0,
                        "filled_quantity": 0,
                        "total_value": 0,
                        "success_count": 0,
                        "fail_count": 0,
                    }
                
                exchange_stats[order.exchange]["total_quantity"] += order.quantity
                exchange_stats[order.exchange]["filled_quantity"] += order.filled_quantity
                exchange_stats[order.exchange]["total_value"] += order.filled_quantity * (order.average_price or 0)
                
                if order.status == "filled":
                    exchange_stats[order.exchange]["success_count"] += 1
                elif order.status == "failed":
                    exchange_stats[order.exchange]["fail_count"] += 1
            
            metrics["exchange_stats"] = exchange_stats
        
        self.update_metrics(metrics)
    
    def to_dict(self) -> Dict:
        """
        Converte a execução para um dicionário.
        
        Returns:
            Dict: Representação da execução como dicionário.
        """
        return {
            "id": self.id,
            "order_id": self.order_id,
            "algorithm": self.algorithm,
            "parameters": self.parameters.dict() if self.parameters else {},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "success": self.success,
            "message": self.message,
            "sub_orders_count": len(self.sub_orders),
            "metrics": self.metrics,
        }