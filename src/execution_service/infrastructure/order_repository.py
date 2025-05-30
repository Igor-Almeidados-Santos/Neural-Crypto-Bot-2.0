"""
Repositório para persistência de ordens.

Este módulo implementa o repositório para persistência de ordens
utilizando SQLAlchemy.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    and_,
    desc,
    func,
    or_,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.execution_service.domain.entities.order import Order

logger = logging.getLogger(__name__)

Base = declarative_base()


class OrderModel(Base):
    """Modelo SQLAlchemy para a tabela de ordens."""
    
    __tablename__ = "orders"
    
    id = Column(String, primary_key=True)
    trading_pair = Column(String, nullable=False, index=True)
    side = Column(String, nullable=False)
    order_type = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    status = Column(String, nullable=False, index=True)
    filled_quantity = Column(Float, nullable=False, default=0)
    average_price = Column(Float, nullable=True)
    fees = Column(Float, nullable=False, default=0)
    exchange = Column(String, nullable=False, index=True)
    exchange_order_id = Column(String, nullable=True)
    strategy_id = Column(String, nullable=True, index=True)
    parent_order_id = Column(String, ForeignKey("orders.id"), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)
    error_message = Column(String, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Relacionamento para ordens filhas
    child_orders = relationship("OrderModel", 
                               backref="parent_order",
                               remote_side=[id],
                               cascade="all, delete-orphan")
    
    def to_entity(self) -> Order:
        """
        Converte o modelo para uma entidade Order.
        
        Returns:
            Order: Entidade Order.
        """
        # Converter atributos básicos
        order = Order(
            id=self.id,
            trading_pair=self.trading_pair,
            side=self.side,
            order_type=self.order_type,
            quantity=self.quantity,
            price=self.price,
            status=self.status,
            filled_quantity=self.filled_quantity,
            average_price=self.average_price,
            fees=self.fees,
            exchange=self.exchange,
            exchange_order_id=self.exchange_order_id,
            strategy_id=self.strategy_id,
            parent_order_id=self.parent_order_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            executed_at=self.executed_at,
            error_message=self.error_message,
            metadata=self.metadata or {},
        )
        
        # Converter ordens filhas recursivamente
        order.child_orders = [child.to_entity() for child in self.child_orders]
        
        return order
    
    @classmethod
    def from_entity(cls, order: Order) -> "OrderModel":
        """
        Cria um modelo a partir de uma entidade Order.
        
        Args:
            order: Entidade Order.
            
        Returns:
            OrderModel: Modelo SQLAlchemy.
        """
        return cls(
            id=order.id,
            trading_pair=order.trading_pair,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            price=order.price,
            status=order.status,
            filled_quantity=order.filled_quantity,
            average_price=order.average_price,
            fees=order.fees,
            exchange=order.exchange,
            exchange_order_id=order.exchange_order_id,
            strategy_id=order.strategy_id,
            parent_order_id=order.parent_order_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
            executed_at=order.executed_at,
            error_message=order.error_message,
            metadata=order.metadata,
            child_orders=[cls.from_entity(child) for child in order.child_orders],
        )


class OrderRepository:
    """
    Repositório para persistência de ordens.
    
    Esta classe encapsula o acesso ao banco de dados para operações
    com ordens, como salvar, buscar, atualizar e deletar.
    """
    
    def __init__(self, db_url: str):
        """
        Inicializa o repositório.
        
        Args:
            db_url: URL de conexão ao banco de dados.
        """
        self.engine = create_async_engine(db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def initialize(self):
        """
        Inicializa o repositório, criando as tabelas se necessário.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Tabelas do repositório de ordens criadas/verificadas")
    
    async def save(self, order: Order) -> Order:
        """
        Salva uma ordem no banco de dados.
        
        Args:
            order: Ordem a ser salva.
            
        Returns:
            Order: Ordem salva.
        """
        async with self.async_session() as session:
            async with session.begin():
                # Verificar se a ordem já existe
                existing = await session.get(OrderModel, order.id)
                
                if existing:
                    # Atualizar campos da ordem existente
                    existing.trading_pair = order.trading_pair
                    existing.side = order.side
                    existing.order_type = order.order_type
                    existing.quantity = order.quantity
                    existing.price = order.price
                    existing.status = order.status
                    existing.filled_quantity = order.filled_quantity
                    existing.average_price = order.average_price
                    existing.fees = order.fees
                    existing.exchange = order.exchange
                    existing.exchange_order_id = order.exchange_order_id
                    existing.strategy_id = order.strategy_id
                    existing.parent_order_id = order.parent_order_id
                    existing.updated_at = datetime.utcnow()
                    existing.executed_at = order.executed_at
                    existing.error_message = order.error_message
                    existing.metadata = order.metadata
                    
                    # Não atualizamos child_orders diretamente para evitar problemas de CASCADE
                    order_model = existing
                else:
                    # Criar novo modelo a partir da entidade
                    order_model = OrderModel.from_entity(order)
                    session.add(order_model)
            
            # Commit é feito automaticamente pelo session.begin()
            return order
    
    async def get_by_id(self, order_id: str) -> Optional[Order]:
        """
        Busca uma ordem pelo ID.
        
        Args:
            order_id: ID da ordem.
            
        Returns:
            Optional[Order]: Ordem encontrada ou None.
        """
        async with self.async_session() as session:
            order_model = await session.get(OrderModel, order_id)
            if order_model:
                return order_model.to_entity()
            return None
    
    async def get_by_exchange_order_id(
        self, exchange: str, exchange_order_id: str
    ) -> Optional[Order]:
        """
        Busca uma ordem pelo ID na exchange.
        
        Args:
            exchange: Nome da exchange.
            exchange_order_id: ID da ordem na exchange.
            
        Returns:
            Optional[Order]: Ordem encontrada ou None.
        """
        async with self.async_session() as session:
            query = select(OrderModel).where(
                and_(
                    OrderModel.exchange == exchange,
                    OrderModel.exchange_order_id == exchange_order_id,
                )
            )
            result = await session.execute(query)
            order_model = result.scalars().first()
            
            if order_model:
                return order_model.to_entity()
            return None
    
    async def list_with_pagination(
        self,
        page: int = 1,
        size: int = 10,
        filters: Optional[Dict] = None,
        sort_by: str = "created_at",
        sort_dir: str = "desc",
    ) -> Tuple[List[Order], int]:
        """
        Lista ordens com paginação e filtros.
        
        Args:
            page: Número da página (começando em 1).
            size: Tamanho da página.
            filters: Filtros a serem aplicados.
            sort_by: Campo para ordenação.
            sort_dir: Direção da ordenação ('asc' ou 'desc').
            
        Returns:
            Tuple[List[Order], int]: Lista de ordens e contagem total.
        """
        if page < 1:
            page = 1
        if size < 1:
            size = 10
        
        async with self.async_session() as session:
            # Construir query base
            query = select(OrderModel)
            count_query = select(func.count()).select_from(OrderModel)
            
            # Aplicar filtros
            if filters:
                filter_conditions = []
                
                if "trading_pair" in filters:
                    filter_conditions.append(OrderModel.trading_pair == filters["trading_pair"])
                
                if "side" in filters:
                    filter_conditions.append(OrderModel.side == filters["side"])
                
                if "status" in filters:
                    if isinstance(filters["status"], list):
                        filter_conditions.append(OrderModel.status.in_(filters["status"]))
                    else:
                        filter_conditions.append(OrderModel.status == filters["status"])
                
                if "exchange" in filters:
                    if isinstance(filters["exchange"], list):
                        filter_conditions.append(OrderModel.exchange.in_(filters["exchange"]))
                    else:
                        filter_conditions.append(OrderModel.exchange == filters["exchange"])
                
                if "strategy_id" in filters:
                    filter_conditions.append(OrderModel.strategy_id == filters["strategy_id"])
                
                if "parent_order_id" in filters:
                    filter_conditions.append(OrderModel.parent_order_id == filters["parent_order_id"])
                
                if "start_date" in filters:
                    filter_conditions.append(OrderModel.created_at >= filters["start_date"])
                
                if "end_date" in filters:
                    filter_conditions.append(OrderModel.created_at <= filters["end_date"])
                
                if "min_quantity" in filters:
                    filter_conditions.append(OrderModel.quantity >= filters["min_quantity"])
                
                if "max_quantity" in filters:
                    filter_conditions.append(OrderModel.quantity <= filters["max_quantity"])
                
                if "search" in filters:
                    search = f"%{filters['search']}%"
                    filter_conditions.append(
                        or_(
                            OrderModel.id.like(search),
                            OrderModel.trading_pair.like(search),
                            OrderModel.exchange.like(search),
                            OrderModel.exchange_order_id.like(search),
                        )
                    )
                
                # Excluir ordens filhas se especificado
                if filters.get("exclude_child_orders", False):
                    filter_conditions.append(OrderModel.parent_order_id.is_(None))
                
                # Aplicar condições à query
                if filter_conditions:
                    query = query.where(and_(*filter_conditions))
                    count_query = count_query.where(and_(*filter_conditions))
            
            # Aplicar ordenação
            if sort_dir.lower() == "asc":
                query = query.order_by(getattr(OrderModel, sort_by))
            else:
                query = query.order_by(desc(getattr(OrderModel, sort_by)))
            
            # Executar query de contagem
            count_result = await session.execute(count_query)
            total = count_result.scalar()
            
            # Aplicar paginação
            query = query.offset((page - 1) * size).limit(size)
            
            # Executar query principal
            result = await session.execute(query)
            order_models = result.scalars().all()
            
            # Converter para entidades
            orders = [model.to_entity() for model in order_models]
            
            return orders, total
    
    async def delete(self, order_id: str) -> bool:
        """
        Deleta uma ordem pelo ID.
        
        Args:
            order_id: ID da ordem.
            
        Returns:
            bool: True se a ordem foi deletada, False caso contrário.
        """
        async with self.async_session() as session:
            async with session.begin():
                order_model = await session.get(OrderModel, order_id)
                if order_model:
                    await session.delete(order_model)
                    return True
                return False
    
    async def get_order_stats(
        self, filters: Optional[Dict] = None
    ) -> Dict[str, Union[int, float]]:
        """
        Obtém estatísticas de ordens com base em filtros.
        
        Args:
            filters: Filtros a serem aplicados.
            
        Returns:
            Dict[str, Union[int, float]]: Estatísticas das ordens.
        """
        async with self.async_session() as session:
            # Construir queries
            total_query = select(func.count()).select_from(OrderModel)
            filled_query = select(func.count()).select_from(OrderModel).where(OrderModel.status == "filled")
            partial_query = select(func.count()).select_from(OrderModel).where(OrderModel.status == "partial")
            pending_query = select(func.count()).select_from(OrderModel).where(OrderModel.status == "pending")
            failed_query = select(func.count()).select_from(OrderModel).where(OrderModel.status == "failed")
            cancelled_query = select(func.count()).select_from(OrderModel).where(OrderModel.status == "cancelled")
            
            volume_query = select(func.sum(OrderModel.quantity)).select_from(OrderModel)
            filled_volume_query = select(func.sum(OrderModel.filled_quantity)).select_from(OrderModel)
            
            buy_query = select(func.count()).select_from(OrderModel).where(OrderModel.side == "buy")
            sell_query = select(func.count()).select_from(OrderModel).where(OrderModel.side == "sell")
            
            buy_volume_query = select(func.sum(OrderModel.quantity)).select_from(OrderModel).where(OrderModel.side == "buy")
            sell_volume_query = select(func.sum(OrderModel.quantity)).select_from(OrderModel).where(OrderModel.side == "sell")
            
            fees_query = select(func.sum(OrderModel.fees)).select_from(OrderModel)
            
            # Aplicar filtros
            if filters:
                filter_conditions = []
                
                if "trading_pair" in filters:
                    filter_conditions.append(OrderModel.trading_pair == filters["trading_pair"])
                
                if "exchange" in filters:
                    if isinstance(filters["exchange"], list):
                        filter_conditions.append(OrderModel.exchange.in_(filters["exchange"]))
                    else:
                        filter_conditions.append(OrderModel.exchange == filters["exchange"])
                
                if "strategy_id" in filters:
                    filter_conditions.append(OrderModel.strategy_id == filters["strategy_id"])
                
                if "start_date" in filters:
                    filter_conditions.append(OrderModel.created_at >= filters["start_date"])
                
                if "end_date" in filters:
                    filter_conditions.append(OrderModel.created_at <= filters["end_date"])
                
                # Aplicar condições às queries
                if filter_conditions:
                    total_query = total_query.where(and_(*filter_conditions))
                    filled_query = filled_query.where(and_(*filter_conditions))
                    partial_query = partial_query.where(and_(*filter_conditions))
                    pending_query = pending_query.where(and_(*filter_conditions))
                    failed_query = failed_query.where(and_(*filter_conditions))
                    cancelled_query = cancelled_query.where(and_(*filter_conditions))
                    volume_query = volume_query.where(and_(*filter_conditions))
                    filled_volume_query = filled_volume_query.where(and_(*filter_conditions))
                    buy_query = buy_query.where(and_(*filter_conditions))
                    sell_query = sell_query.where(and_(*filter_conditions))
                    buy_volume_query = buy_volume_query.where(and_(*filter_conditions))
                    sell_volume_query = sell_volume_query.where(and_(*filter_conditions))
                    fees_query = fees_query.where(and_(*filter_conditions))
            
            # Executar queries
            total_result = await session.execute(total_query)
            filled_result = await session.execute(filled_query)
            partial_result = await session.execute(partial_query)
            pending_result = await session.execute(pending_query)
            failed_result = await session.execute(failed_query)
            cancelled_result = await session.execute(cancelled_query)
            volume_result = await session.execute(volume_query)
            filled_volume_result = await session.execute(filled_volume_query)
            buy_result = await session.execute(buy_query)
            sell_result = await session.execute(sell_query)
            buy_volume_result = await session.execute(buy_volume_query)
            sell_volume_result = await session.execute(sell_volume_query)
            fees_result = await session.execute(fees_query)
            
            # Extrair resultados
            total = total_result.scalar() or 0
            filled = filled_result.scalar() or 0
            partial = partial_result.scalar() or 0
            pending = pending_result.scalar() or 0
            failed = failed_result.scalar() or 0
            cancelled = cancelled_result.scalar() or 0
            volume = volume_result.scalar() or 0
            filled_volume = filled_volume_result.scalar() or 0
            buy_count = buy_result.scalar() or 0
            sell_count = sell_result.scalar() or 0
            buy_volume = buy_volume_result.scalar() or 0
            sell_volume = sell_volume_result.scalar() or 0
            total_fees = fees_result.scalar() or 0
            
            # Calcular percentuais
            fill_rate = (filled / total) * 100 if total > 0 else 0
            success_rate = ((filled + partial) / total) * 100 if total > 0 else 0
            buy_percentage = (buy_count / total) * 100 if total > 0 else 0
            
            return {
                "total_orders": total,
                "filled_orders": filled,
                "partial_orders": partial,
                "pending_orders": pending,
                "failed_orders": failed,
                "cancelled_orders": cancelled,
                "total_volume": volume,
                "filled_volume": filled_volume,
                "fill_percentage": filled_volume / volume * 100 if volume > 0 else 0,
                "buy_orders": buy_count,
                "sell_orders": sell_count,
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "buy_percentage": buy_percentage,
                "sell_percentage": 100 - buy_percentage,
                "fill_rate": fill_rate,
                "success_rate": success_rate,
                "total_fees": total_fees,
            }
    
    async def update_order_status(
        self,
        order_id: str,
        status: str,
        filled_quantity: Optional[float] = None,
        average_price: Optional[float] = None,
        fees: Optional[float] = None,
        exchange_order_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Order]:
        """
        Atualiza o status de uma ordem.
        
        Args:
            order_id: ID da ordem.
            status: Novo status.
            filled_quantity: Quantidade preenchida (opcional).
            average_price: Preço médio (opcional).
            fees: Taxas (opcional).
            exchange_order_id: ID da ordem na exchange (opcional).
            error_message: Mensagem de erro (opcional).
            
        Returns:
            Optional[Order]: Ordem atualizada ou None se não encontrada.
        """
        async with self.async_session() as session:
            async with session.begin():
                order_model = await session.get(OrderModel, order_id)
                if not order_model:
                    return None
                
                # Atualizar campos
                order_model.status = status
                order_model.updated_at = datetime.utcnow()
                
                if filled_quantity is not None:
                    order_model.filled_quantity = filled_quantity
                
                if average_price is not None:
                    order_model.average_price = average_price
                
                if fees is not None:
                    order_model.fees = fees
                
                if exchange_order_id is not None:
                    order_model.exchange_order_id = exchange_order_id
                
                if error_message is not None:
                    order_model.error_message = error_message
                
                if status in ["filled", "partial"]:
                    order_model.executed_at = datetime.utcnow()
                
                return order_model.to_entity()
    
    async def get_active_orders(
        self, exchange: Optional[str] = None, strategy_id: Optional[str] = None
    ) -> List[Order]:
        """
        Obtém as ordens ativas (pending ou partial).
        
        Args:
            exchange: Filtrar por exchange (opcional).
            strategy_id: Filtrar por estratégia (opcional).
            
        Returns:
            List[Order]: Lista de ordens ativas.
        """
        async with self.async_session() as session:
            # Construir query base
            query = select(OrderModel).where(
                OrderModel.status.in_(["pending", "partial"])
            )
            
            # Aplicar filtros adicionais
            if exchange:
                query = query.where(OrderModel.exchange == exchange)
            
            if strategy_id:
                query = query.where(OrderModel.strategy_id == strategy_id)
            
            # Ordenar por data de criação (mais antigas primeiro)
            query = query.order_by(OrderModel.created_at)
            
            # Executar query
            result = await session.execute(query)
            order_models = result.scalars().all()
            
            # Converter para entidades
            return [model.to_entity() for model in order_models]
    
    async def close(self):
        """
        Fecha a conexão com o banco de dados.
        """
        await self.engine.dispose()
        logger.info("Conexão com o banco de dados fechada")