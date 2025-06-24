"""
TWAP (Time-Weighted Average Price) execution algorithm.

Este módulo implementa o algoritmo TWAP para execução de ordens,
que divide uma ordem grande em partes iguais ao longo do tempo
para minimizar o impacto no mercado.
"""
import asyncio
import logging
import math
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from execution_service.domain.entities.order import Order
from execution_service.domain.value_objects.execution_parameters import (
    ExecutionParameters,
    TwapParameters,
)

logger = logging.getLogger(__name__)


class TwapAlgorithm:
    """
    Implementação do algoritmo Time-Weighted Average Price (TWAP).
    
    O TWAP divide uma ordem grande em partes menores distribuídas uniformemente
    ao longo de um período de tempo especificado, com o objetivo de obter um preço
    médio que se aproxime do preço médio do mercado durante esse período.
    """

    def __init__(self, exchange_client):
        """
        Inicializa o algoritmo TWAP.
        
        Args:
            exchange_client: Cliente para comunicação com a exchange.
        """
        self.exchange_client = exchange_client

    async def execute(
        self, order: Order, params: ExecutionParameters
    ) -> Tuple[bool, List[Order], Optional[str]]:
        """
        Executa a ordem usando o algoritmo TWAP.
        
        Args:
            order: A ordem a ser executada.
            params: Parâmetros de execução, incluindo configurações específicas do TWAP.
            
        Returns:
            Tuple contendo:
            - bool: Indicador de sucesso da execução.
            - List[Order]: Lista de sub-ordens criadas e executadas.
            - Optional[str]: Mensagem de erro, se houver.
        """
        if not isinstance(params, TwapParameters):
            return False, [], "Parâmetros de execução inválidos para TWAP"

        logger.info(
            f"Iniciando execução TWAP para ordem {order.id} - "
            f"Quantidade: {order.quantity}, Duração: {params.duration_minutes} minutos, "
            f"Intervalos: {params.num_slices}"
        )

        # Calcular o tamanho de cada fatia
        slice_size = order.quantity / params.num_slices
        slice_size = round(slice_size, 8)  # Arredondar para 8 casas decimais (típico para crypto)
        
        # Se a quantidade for muito pequena para o número de slices, ajustar
        if slice_size * params.num_slices < order.quantity:
            remainder = order.quantity - (slice_size * params.num_slices)
            logger.debug(f"Ajustando quantidade final para cobrir arredondamento: {remainder}")
        
        # Calcular o intervalo de tempo entre cada execução
        interval_seconds = (params.duration_minutes * 60) / params.num_slices
        
        sub_orders = []
        success_count = 0
        
        # Criar e executar cada slice
        for i in range(params.num_slices):
            # Calcular a quantidade para este slice (adicionar o remainder no último slice)
            current_slice_size = slice_size
            if i == params.num_slices - 1 and slice_size * params.num_slices < order.quantity:
                current_slice_size = order.quantity - (slice_size * (params.num_slices - 1))
            
            # Criar sub-ordem
            sub_order = Order(
                trading_pair=order.trading_pair,
                side=order.side,
                order_type=order.order_type,
                quantity=current_slice_size,
                price=order.price,
                status="pending",
                parent_order_id=order.id,
                exchange=order.exchange,
            )
            
            try:
                # Executar a sub-ordem
                result = await self.exchange_client.place_order(
                    exchange=sub_order.exchange,
                    trading_pair=sub_order.trading_pair,
                    side=sub_order.side,
                    order_type=sub_order.order_type,
                    quantity=sub_order.quantity,
                    price=sub_order.price,
                )
                
                # Atualizar a sub-ordem com o resultado
                sub_order.exchange_order_id = result.get("order_id")
                sub_order.status = "filled" if result.get("status") == "filled" else "partial"
                sub_order.filled_quantity = result.get("filled_quantity", 0)
                sub_order.average_price = result.get("average_price")
                sub_order.fees = result.get("fees", 0)
                sub_order.executed_at = datetime.utcnow()
                
                logger.info(
                    f"TWAP slice {i+1}/{params.num_slices} executada: "
                    f"Quantidade: {sub_order.quantity}, "
                    f"Status: {sub_order.status}"
                )
                
                if sub_order.status in ["filled", "partial"]:
                    success_count += 1
                
                sub_orders.append(sub_order)
                
            except Exception as e:
                logger.error(f"Erro ao executar slice TWAP {i+1}: {str(e)}")
                sub_order.status = "failed"
                sub_order.error_message = str(e)
                sub_orders.append(sub_order)
            
            # Aguardar o intervalo para a próxima execução, exceto na última iteração
            if i < params.num_slices - 1:
                await asyncio.sleep(interval_seconds)
        
        # Verificar o status geral da execução
        success = success_count == params.num_slices
        
        # Se tiver pelo menos uma execução parcial, mas não todas com sucesso
        if not success and success_count > 0:
            message = f"Execução TWAP parcial: {success_count}/{params.num_slices} slices executadas com sucesso"
            logger.warning(message)
            return True, sub_orders, message
        
        # Se todas as execuções falharam
        if success_count == 0:
            message = "Falha na execução TWAP: todas as slices falharam"
            logger.error(message)
            return False, sub_orders, message
        
        logger.info(f"Execução TWAP concluída com sucesso para ordem {order.id}")
        return True, sub_orders, None

    async def calculate_expected_metrics(
        self, order: Order, params: TwapParameters
    ) -> dict:
        """
        Calcula métricas esperadas para a execução TWAP.
        
        Args:
            order: A ordem a ser executada.
            params: Parâmetros específicos do TWAP.
            
        Returns:
            dict: Dicionário com métricas estimadas.
        """
        # Obter dados do mercado para estimativas
        try:
            market_data = await self.exchange_client.get_market_data(
                exchange=order.exchange,
                trading_pair=order.trading_pair,
            )
            
            current_price = market_data.get("last_price", order.price or 0)
            
            # Estimar o preço médio baseado na volatilidade histórica recente
            volatility = market_data.get("volatility_24h", 0.02)  # 2% padrão se não disponível
            expected_price_range = current_price * volatility * (params.duration_minutes / 1440)  # Escala para duração
            
            # Estimar o slippage baseado na profundidade do livro de ordens
            book_depth = market_data.get("book_depth", {})
            slippage_estimate = self._estimate_slippage(
                order.side, order.quantity / params.num_slices, book_depth
            )
            
            return {
                "expected_average_price": current_price,
                "expected_price_range": (
                    current_price - expected_price_range,
                    current_price + expected_price_range
                ),
                "estimated_slippage": slippage_estimate,
                "estimated_duration": params.duration_minutes,
                "slice_size": order.quantity / params.num_slices,
                "slice_interval_seconds": (params.duration_minutes * 60) / params.num_slices,
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas TWAP esperadas: {str(e)}")
            return {
                "error": str(e),
                "slice_size": order.quantity / params.num_slices,
                "slice_interval_seconds": (params.duration_minutes * 60) / params.num_slices,
            }

    def _estimate_slippage(self, side: str, quantity: float, book_depth: dict) -> float:
        """
        Estima o slippage com base na profundidade do livro de ordens.
        
        Args:
            side: Lado da ordem ('buy' ou 'sell').
            quantity: Quantidade a ser executada.
            book_depth: Dados de profundidade do livro.
            
        Returns:
            float: Slippage estimado em percentual.
        """
        if not book_depth:
            return 0.001  # 0.1% estimativa padrão se não houver dados
        
        side_data = book_depth.get("bids" if side == "buy" else "asks", [])
        if not side_data:
            return 0.001
        
        # Simular o impacto no livro
        remaining_qty = quantity
        weighted_price = 0
        best_price = side_data[0][0] if side_data else 0
        
        for price, qty in side_data:
            if remaining_qty <= 0:
                break
                
            matched_qty = min(remaining_qty, qty)
            weighted_price += matched_qty * price
            remaining_qty -= matched_qty
        
        if quantity - remaining_qty <= 0:
            return 0.001
        
        avg_price = weighted_price / (quantity - remaining_qty)
        slippage = abs(avg_price - best_price) / best_price
        
        return slippage