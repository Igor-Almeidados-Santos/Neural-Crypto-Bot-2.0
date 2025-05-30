"""
Iceberg execution algorithm.

Este módulo implementa o algoritmo Iceberg para execução de ordens,
que divide uma ordem grande em partes menores para ocultar o tamanho
total da ordem, reduzindo o impacto no mercado.
"""
import asyncio
import logging
import random
from datetime import datetime
from typing import List, Optional, Tuple

from src.execution_service.domain.entities.order import Order
from src.execution_service.domain.value_objects.execution_parameters import (
    ExecutionParameters,
    IcebergParameters,
)

logger = logging.getLogger(__name__)


class IcebergAlgorithm:
    """
    Implementação do algoritmo Iceberg.
    
    O algoritmo Iceberg divide uma ordem grande em várias ordens menores
    (a "ponta do iceberg") que são submetidas sequencialmente à medida que 
    as anteriores são executadas, ocultando o tamanho total da ordem.
    """

    def __init__(self, exchange_client):
        """
        Inicializa o algoritmo Iceberg.
        
        Args:
            exchange_client: Cliente para comunicação com a exchange.
        """
        self.exchange_client = exchange_client

    async def execute(
        self, order: Order, params: ExecutionParameters
    ) -> Tuple[bool, List[Order], Optional[str]]:
        """
        Executa a ordem usando o algoritmo Iceberg.
        
        Args:
            order: A ordem a ser executada.
            params: Parâmetros de execução, incluindo configurações específicas do Iceberg.
            
        Returns:
            Tuple contendo:
            - bool: Indicador de sucesso da execução.
            - List[Order]: Lista de sub-ordens criadas e executadas.
            - Optional[str]: Mensagem de erro, se houver.
        """
        if not isinstance(params, IcebergParameters):
            return False, [], "Parâmetros de execução inválidos para Iceberg"

        logger.info(
            f"Iniciando execução Iceberg para ordem {order.id} - "
            f"Quantidade total: {order.quantity}, "
            f"Tamanho da ponta do iceberg: {params.display_size}, "
            f"Variação: {params.size_variance * 100}%"
        )

        # Quantidade total a ser executada
        remaining_quantity = order.quantity
        executed_quantity = 0
        sub_orders = []
        
        # Executar até que toda a quantidade seja processada
        while remaining_quantity > 0:
            # Determinar o tamanho da próxima sub-ordem
            if params.size_variance > 0:
                # Aplicar variação aleatória ao tamanho de exibição
                variance_factor = 1.0 + random.uniform(
                    -params.size_variance, params.size_variance
                )
                display_size = params.display_size * variance_factor
            else:
                display_size = params.display_size
                
            # Garantir que não exceda a quantidade restante
            current_size = min(display_size, remaining_quantity)
            current_size = round(current_size, 8)  # Arredondar para 8 casas decimais
            
            # Criar sub-ordem
            sub_order = Order(
                trading_pair=order.trading_pair,
                side=order.side,
                order_type=order.order_type,
                quantity=current_size,
                price=order.price,
                status="pending",
                parent_order_id=order.id,
                exchange=order.exchange,
            )
            
            try:
                # Verificar se o mercado mudou significativamente
                if params.price_adjustment_threshold > 0 and sub_orders:
                    await self._check_and_adjust_price(order, sub_order, params)
                
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
                
                # Atualizar quantidades
                executed_this_round = sub_order.filled_quantity
                executed_quantity += executed_this_round
                remaining_quantity -= executed_this_round
                
                # Arredondar para evitar problemas de precisão
                remaining_quantity = round(remaining_quantity, 8)
                
                logger.info(
                    f"Iceberg slice executada: "
                    f"Quantidade: {sub_order.quantity}, "
                    f"Preenchida: {sub_order.filled_quantity}, "
                    f"Restante total: {remaining_quantity}"
                )
                
                sub_orders.append(sub_order)
                
                # Se a ordem foi apenas parcialmente preenchida, tentar novamente
                if sub_order.status == "partial" and sub_order.filled_quantity < sub_order.quantity:
                    unfilled = sub_order.quantity - sub_order.filled_quantity
                    remaining_quantity += unfilled
                    logger.info(f"Ordem parcialmente preenchida, adicionando {unfilled} de volta à quantidade restante")
                
                # Aguardar intervalo para evitar detecção de algoritmo
                if remaining_quantity > 0 and params.interval_seconds > 0:
                    jitter = params.interval_seconds * random.uniform(0, params.interval_variance)
                    wait_time = params.interval_seconds + jitter
                    logger.debug(f"Aguardando {wait_time:.2f}s até a próxima fatia")
                    await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Erro ao executar fatia Iceberg: {str(e)}")
                sub_order.status = "failed"
                sub_order.error_message = str(e)
                sub_orders.append(sub_order)
                
                # Verificar se devemos continuar ou abortar
                if params.continue_on_failure:
                    logger.warning(f"Continuando execução Iceberg apesar do erro na fatia")
                else:
                    return False, sub_orders, f"Falha na execução: {str(e)}"
                
                # Pequena pausa antes de tentar novamente
                await asyncio.sleep(1)
        
        # Verificar se toda a ordem foi executada
        total_executed = sum(order.filled_quantity for order in sub_orders)
        success = abs(total_executed - order.quantity) < 0.00000001  # Tolerância para erro de ponto flutuante
        
        if success:
            logger.info(f"Execução Iceberg concluída com sucesso para ordem {order.id}")
            return True, sub_orders, None
        else:
            message = f"Execução Iceberg parcial: {total_executed}/{order.quantity} executado"
            logger.warning(message)
            return True, sub_orders, message

    async def _check_and_adjust_price(
        self, original_order: Order, current_order: Order, params: IcebergParameters
    ) -> None:
        """
        Verifica as condições de mercado e ajusta o preço se necessário.
        
        Args:
            original_order: A ordem original.
            current_order: A sub-ordem atual sendo preparada.
            params: Parâmetros de execução do Iceberg.
        """
        try:
            # Obter último preço do mercado
            market_data = await self.exchange_client.get_market_data(
                exchange=original_order.exchange,
                trading_pair=original_order.trading_pair,
            )
            
            market_price = market_data.get("last_price")
            if not market_price:
                return
                
            # Se não há preço definido na ordem original, usar preço de mercado
            if not original_order.price:
                current_order.price = market_price
                return
                
            # Calcular desvio do preço atual em relação ao preço original
            price_deviation = abs(market_price - original_order.price) / original_order.price
            
            # Se o desvio exceder o limite, ajustar o preço
            if price_deviation > params.price_adjustment_threshold:
                logger.info(
                    f"Preço de mercado ({market_price}) desviou-se significativamente do preço original "
                    f"({original_order.price}). Ajustando preço da sub-ordem."
                )
                
                # Para ordens de compra, garantir que não compramos muito acima do mercado
                if original_order.side == "buy":
                    # Se o preço original for menor que o preço de mercado atual, usar o preço de mercado
                    if original_order.price < market_price:
                        current_order.price = market_price
                    # Caso contrário, manter o preço original
                    else:
                        current_order.price = original_order.price
                        
                # Para ordens de venda, garantir que não vendemos muito abaixo do mercado
                else:  # side == "sell"
                    # Se o preço original for maior que o preço de mercado atual, usar o preço de mercado
                    if original_order.price > market_price:
                        current_order.price = market_price
                    # Caso contrário, manter o preço original
                    else:
                        current_order.price = original_order.price
            else:
                # Se não exceder o limite, manter o preço original
                current_order.price = original_order.price
                
        except Exception as e:
            logger.warning(f"Erro ao tentar ajustar preço: {str(e)}. Usando preço original.")
            current_order.price = original_order.price

    async def calculate_expected_metrics(
        self, order: Order, params: IcebergParameters
    ) -> dict:
        """
        Calcula métricas esperadas para a execução Iceberg.
        
        Args:
            order: A ordem a ser executada.
            params: Parâmetros específicos do Iceberg.
            
        Returns:
            dict: Dicionário com métricas estimadas.
        """
        try:
            market_data = await self.exchange_client.get_market_data(
                exchange=order.exchange,
                trading_pair=order.trading_pair,
            )
            
            # Estimar número de fatias
            num_slices = order.quantity / params.display_size
            if params.size_variance > 0:
                # Ajustar para variação
                num_slices = num_slices * (1 + params.size_variance / 2)
            num_slices = int(num_slices) + 1  # Arredondar para cima
            
            # Estimar tempo total
            estimated_seconds = num_slices * params.interval_seconds
            # Adicionar tempo para execução das ordens
            estimated_seconds += num_slices * 2  # ~2 segundos por ordem em média
            
            # Estimar preço médio
            current_price = market_data.get("last_price", order.price or 0)
            
            # Estimar o slippage baseado na profundidade do livro
            book_depth = market_data.get("book_depth", {})
            slippage_estimate = 0.001  # 0.1% padrão
            if book_depth:
                slippage_estimate = self._estimate_slippage(
                    order.side, params.display_size, book_depth
                )
            
            return {
                "estimated_slices": num_slices,
                "estimated_duration_seconds": estimated_seconds,
                "display_size": params.display_size,
                "estimated_slippage": slippage_estimate,
                "expected_average_price": current_price * (1 + slippage_estimate if order.side == "buy" else 1 - slippage_estimate),
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas Iceberg esperadas: {str(e)}")
            return {
                "error": str(e),
                "estimated_slices": order.quantity / params.display_size,
                "display_size": params.display_size,
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