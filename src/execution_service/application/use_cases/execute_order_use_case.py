"""
Caso de uso para execução de ordens.

Este módulo implementa o caso de uso para execução de ordens de trading,
coordenando os diferentes componentes do sistema.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from execution_service.algorithms.iceberg import IcebergAlgorithm
from execution_service.algorithms.smart_routing import SmartRoutingAlgorithm
from execution_service.algorithms.twap import TwapAlgorithm
from execution_service.application.services.execution_optimization_service import (
    ExecutionOptimizationService,
)
from execution_service.application.services.order_routing_service import (
    OrderRoutingService,
)
from execution_service.domain.entities.execution import Execution
from execution_service.domain.entities.order import Order
from execution_service.domain.value_objects.execution_parameters import (
    DirectParameters,
    ExecutionParameters,
    IcebergParameters,
    SmartRoutingParameters,
    TwapParameters,
    create_execution_parameters,
)
from execution_service.infrastructure.exchange_client import ExchangeClient
from execution_service.infrastructure.execution_event_publisher import (
    ExecutionEventPublisher,
)
from execution_service.infrastructure.order_repository import OrderRepository

logger = logging.getLogger(__name__)


class ExecuteOrderUseCase:
    """
    Caso de uso para execução de ordens.
    
    Este caso de uso é responsável por coordenar o processo de execução
    de ordens, incluindo roteamento, otimização e execução propriamente dita.
    """
    
    def __init__(
        self,
        exchange_client: ExchangeClient,
        order_repository: OrderRepository,
        order_routing_service: OrderRoutingService,
        execution_optimization_service: ExecutionOptimizationService,
        event_publisher: ExecutionEventPublisher,
        config: Dict,
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            exchange_client: Cliente para comunicação com exchanges.
            order_repository: Repositório de ordens.
            order_routing_service: Serviço de roteamento de ordens.
            execution_optimization_service: Serviço de otimização de execução.
            event_publisher: Publisher de eventos de execução.
            config: Configuração do caso de uso.
        """
        self.exchange_client = exchange_client
        self.order_repository = order_repository
        self.order_routing_service = order_routing_service
        self.execution_optimization_service = execution_optimization_service
        self.event_publisher = event_publisher
        self.config = config
        
        # Inicializar algoritmos
        self.twap_algorithm = TwapAlgorithm(exchange_client)
        self.iceberg_algorithm = IcebergAlgorithm(exchange_client)
        self.smart_routing_algorithm = SmartRoutingAlgorithm(exchange_client)
    
    async def execute(self, order: Order) -> Order:
        """
        Executa uma ordem.
        
        Args:
            order: Ordem a ser executada.
            
        Returns:
            Order: Ordem após a execução.
            
        Raises:
            ValueError: Se a ordem for inválida.
            Exception: Se houver um erro na execução.
        """
        logger.info(f"Iniciando execução da ordem {order.id} - {order.side} {order.quantity} {order.trading_pair}")
        
        # Validar a ordem
        self._validate_order(order)
        
        # Salvar a ordem no repositório
        await self.order_repository.save(order)
        
        # Publicar evento de criação da ordem
        self.event_publisher.publish_order_created(order)
        
        try:
            # Determinar rota e algoritmo de execução
            exchanges, algorithm, params = await self.order_routing_service.route_order(order)
            
            # Otimizar parâmetros de execução
            optimized_params = await self._optimize_parameters(order, algorithm, params, exchanges)
            
            # Criar objeto de execução
            execution = Execution(
                order_id=order.id,
                algorithm=algorithm,
                parameters=optimized_params,
            )
            
            # Executar a ordem com o algoritmo apropriado
            success, sub_orders, message = await self._execute_with_algorithm(
                order, execution, algorithm, optimized_params
            )
            
            # Atualizar a execução com o resultado
            execution.complete(success, message, sub_orders)
            
            # Atualizar a ordem pai com base nas sub-ordens
            if sub_orders:
                # Adicionar sub-ordens à ordem pai
                order.child_orders = sub_orders
                
                # Atualizar a ordem pai com base nas sub-ordens
                order.update_from_child_orders()
            else:
                # Se não houver sub-ordens, atualizar diretamente
                if success:
                    order.status = "filled"
                    order.executed_at = datetime.utcnow()
                else:
                    order.status = "failed"
                    order.error_message = message
            
            # Calcular métricas de execução
            execution.calculate_execution_metrics(order)
            
            # Salvar a ordem atualizada
            await self.order_repository.save(order)
            
            # Publicar evento de execução
            self.event_publisher.publish_order_executed(execution, order)
            
            logger.info(
                f"Ordem {order.id} executada com {algorithm}. "
                f"Status: {order.status}, "
                f"Quantidade preenchida: {order.filled_quantity}/{order.quantity}"
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Erro ao executar ordem {order.id}: {str(e)}")
            
            # Atualizar status da ordem para falha
            order.status = "failed"
            order.error_message = str(e)
            await self.order_repository.save(order)
            
            # Re-lançar a exceção
            raise
    
    def _validate_order(self, order: Order) -> None:
        """
        Valida uma ordem antes da execução.
        
        Args:
            order: Ordem a ser validada.
            
        Raises:
            ValueError: Se a ordem for inválida.
        """
        # Verificar campos obrigatórios
        if not order.trading_pair:
            raise ValueError("Trading pair não especificado")
        
        if not order.side or order.side not in ["buy", "sell"]:
            raise ValueError(f"Side inválido: {order.side}")
        
        if not order.order_type or order.order_type not in ["market", "limit"]:
            raise ValueError(f"Tipo de ordem inválido: {order.order_type}")
        
        if order.quantity <= 0:
            raise ValueError(f"Quantidade inválida: {order.quantity}")
        
        # Verificar se o preço está presente para ordens limit
        if order.order_type == "limit" and order.price is None:
            raise ValueError("Preço é obrigatório para ordens limit")
        
        # Verificar se a exchange está especificada
        if not order.exchange and order.exchange != "auto":
            raise ValueError("Exchange não especificada")
    
    async def _optimize_parameters(
        self, order: Order, algorithm: str, params: ExecutionParameters, exchanges: List[str]
    ) -> ExecutionParameters:
        """
        Otimiza os parâmetros de execução.
        
        Args:
            order: Ordem a ser executada.
            algorithm: Algoritmo de execução.
            params: Parâmetros iniciais.
            exchanges: Exchanges disponíveis.
            
        Returns:
            ExecutionParameters: Parâmetros otimizados.
        """
        try:
            # Otimizar parâmetros com base no algoritmo
            if algorithm == "twap" and isinstance(params, TwapParameters):
                return await self.execution_optimization_service.optimize_twap_parameters(order, params)
                
            elif algorithm == "iceberg" and isinstance(params, IcebergParameters):
                return await self.execution_optimization_service.optimize_iceberg_parameters(order, params)
                
            elif algorithm == "smart_routing" and isinstance(params, SmartRoutingParameters):
                return await self.execution_optimization_service.optimize_smart_routing_parameters(
                    order, exchanges, params
                )
                
            else:
                # Para outros algoritmos ou se a otimização falhar, retornar os parâmetros originais
                return params
                
        except Exception as e:
            logger.warning(f"Erro ao otimizar parâmetros: {str(e)}. Usando parâmetros originais.")
            return params
    
    async def _execute_with_algorithm(
        self, order: Order, execution: Execution, algorithm: str, params: ExecutionParameters
    ) -> Tuple[bool, List[Order], Optional[str]]:
        """
        Executa uma ordem com o algoritmo especificado.
        
        Args:
            order: Ordem a ser executada.
            execution: Objeto de execução.
            algorithm: Algoritmo de execução.
            params: Parâmetros de execução.
            
        Returns:
            Tuple contendo:
            - bool: Indicador de sucesso.
            - List[Order]: Lista de sub-ordens.
            - Optional[str]: Mensagem de erro ou resultado.
            
        Raises:
            ValueError: Se o algoritmo for inválido.
        """
        # Executar com o algoritmo apropriado
        if algorithm == "twap":
            return await self.twap_algorithm.execute(order, params)
            
        elif algorithm == "iceberg":
            return await self.iceberg_algorithm.execute(order, params)
            
        elif algorithm == "smart_routing":
            return await self.smart_routing_algorithm.execute(order, params)
            
        elif algorithm == "direct":
            # Execução direta sem algoritmo especial
            return await self._execute_direct(order, params)
            
        else:
            raise ValueError(f"Algoritmo não suportado: {algorithm}")
    
    async def _execute_direct(
        self, order: Order, params: DirectParameters
    ) -> Tuple[bool, List[Order], Optional[str]]:
        """
        Executa uma ordem diretamente na exchange.
        
        Args:
            order: Ordem a ser executada.
            params: Parâmetros de execução direta.
            
        Returns:
            Tuple contendo:
            - bool: Indicador de sucesso.
            - List[Order]: Lista contendo apenas a ordem original.
            - Optional[str]: Mensagem de erro ou resultado.
        """
        # Validar preço de mercado se configurado
        if params.validate_market_price and order.price is not None:
            try:
                market_data = await self.exchange_client.get_market_data(order.exchange, order.trading_pair)
                market_price = market_data.get("last_price", 0)
                
                if market_price > 0:
                    # Calcular desvio
                    deviation = abs(order.price - market_price) / market_price
                    
                    # Se o desvio for muito grande, abortar
                    if deviation > params.max_price_deviation:
                        return False, [], f"Preço ({order.price}) desvia muito do preço de mercado ({market_price}): {deviation:.2%}"
            except Exception as e:
                logger.warning(f"Erro ao validar preço de mercado: {str(e)}")
        
        # Tentativas de execução
        max_retries = params.max_retries if params.retry_on_failure else 0
        retries = 0
        
        while True:
            try:
                # Executar a ordem na exchange
                result = await self.exchange_client.place_order(
                    exchange=order.exchange,
                    trading_pair=order.trading_pair,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=order.quantity,
                    price=order.price,
                )
                
                # Atualizar a ordem com o resultado
                order.exchange_order_id = result.get("order_id")
                order.status = result.get("status", "pending")
                order.filled_quantity = result.get("filled_quantity", 0)
                order.average_price = result.get("average_price")
                order.fees = result.get("fees", 0)
                order.executed_at = datetime.utcnow()
                
                # Verificar sucesso
                success = order.status in ["filled", "partial"]
                message = None
                
                if not success:
                    message = f"Ordem não preenchida na exchange. Status: {order.status}"
                
                return success, [order], message
                
            except Exception as e:
                logger.error(f"Erro ao executar ordem diretamente: {str(e)}")
                retries += 1
                
                if retries <= max_retries:
                    logger.info(f"Tentando novamente ({retries}/{max_retries})...")
                    await asyncio.sleep(params.retry_delay_seconds)
                else:
                    order.status = "failed"
                    order.error_message = str(e)
                    return False, [order], f"Falha após {retries} tentativas: {str(e)}"