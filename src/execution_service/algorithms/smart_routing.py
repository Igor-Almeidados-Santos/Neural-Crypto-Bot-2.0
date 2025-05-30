"""
Smart Routing execution algorithm.

Este módulo implementa o algoritmo Smart Routing para execução de ordens,
que distribui uma ordem entre múltiplas exchanges para obter o melhor preço
e minimizar o impacto no mercado.
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.execution_service.domain.entities.order import Order
from src.execution_service.domain.value_objects.execution_parameters import (
    ExecutionParameters,
    SmartRoutingParameters,
)

logger = logging.getLogger(__name__)


class SmartRoutingAlgorithm:
    """
    Implementação do algoritmo Smart Routing.
    
    O algoritmo Smart Routing distribui uma ordem entre múltiplas exchanges
    com base na liquidez, spread e taxas disponíveis, visando obter o melhor
    preço médio de execução.
    """

    def __init__(self, exchange_client):
        """
        Inicializa o algoritmo Smart Routing.
        
        Args:
            exchange_client: Cliente para comunicação com as exchanges.
        """
        self.exchange_client = exchange_client

    async def execute(
        self, order: Order, params: ExecutionParameters
    ) -> Tuple[bool, List[Order], Optional[str]]:
        """
        Executa a ordem usando o algoritmo Smart Routing.
        
        Args:
            order: A ordem a ser executada.
            params: Parâmetros de execução, incluindo configurações específicas do Smart Routing.
            
        Returns:
            Tuple contendo:
            - bool: Indicador de sucesso da execução.
            - List[Order]: Lista de sub-ordens criadas e executadas.
            - Optional[str]: Mensagem de erro, se houver.
        """
        if not isinstance(params, SmartRoutingParameters):
            return False, [], "Parâmetros de execução inválidos para Smart Routing"

        logger.info(
            f"Iniciando execução Smart Routing para ordem {order.id} - "
            f"Quantidade: {order.quantity}, "
            f"Exchanges: {params.exchanges}"
        )

        # Validar as exchanges especificadas
        if not params.exchanges:
            return False, [], "Nenhuma exchange especificada para Smart Routing"
        
        # Obter informações de mercado de todas as exchanges
        market_data = {}
        try:
            market_data = await self._gather_market_data(order.trading_pair, params.exchanges)
        except Exception as e:
            logger.error(f"Erro ao coletar dados de mercado: {str(e)}")
            return False, [], f"Falha ao obter dados de mercado: {str(e)}"
        
        # Calcular a alocação ótima entre as exchanges
        allocation = await self._calculate_optimal_allocation(
            order.quantity,
            order.side,
            market_data,
            params
        )
        
        if not allocation:
            return False, [], "Não foi possível determinar uma alocação ótima"
        
        # Executar as ordens em cada exchange
        sub_orders = []
        success_count = 0
        total_executed = 0
        
        # Criar tarefas para execução paralela
        tasks = []
        for exchange, quantity in allocation.items():
            # Criar sub-ordem para esta exchange
            sub_order = Order(
                trading_pair=order.trading_pair,
                side=order.side,
                order_type=order.order_type,
                quantity=quantity,
                price=order.price,
                status="pending",
                parent_order_id=order.id,
                exchange=exchange,
            )
            sub_orders.append(sub_order)
            
            # Criar tarefa para execução
            task = self._execute_on_exchange(sub_order)
            tasks.append(task)
        
        # Executar todas as ordens em paralelo
        if params.execute_parallel:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Processar resultados
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Erro na execução em {sub_orders[i].exchange}: {str(result)}")
                    sub_orders[i].status = "failed"
                    sub_orders[i].error_message = str(result)
                else:
                    success_count += 1
                    total_executed += sub_orders[i].filled_quantity
        else:
            # Execução sequencial
            for task in tasks:
                try:
                    await task
                    success_count += 1
                    # total_executed será atualizado pela função _execute_on_exchange
                except Exception as e:
                    logger.error(f"Erro na execução sequencial: {str(e)}")
        
        # Atualizar a quantidade total executada
        total_executed = sum(order.filled_quantity for order in sub_orders if order.status in ["filled", "partial"])
        
        # Verificar se a execução foi bem-sucedida
        if success_count == 0:
            return False, sub_orders, "Todas as sub-ordens falharam"
        
        if total_executed < order.quantity * 0.95:  # Menos de 95% executado
            return True, sub_orders, f"Execução parcial: {total_executed}/{order.quantity} executado"
        
        logger.info(f"Execução Smart Routing concluída com sucesso para ordem {order.id}")
        return True, sub_orders, None

    async def _execute_on_exchange(self, sub_order: Order) -> None:
        """
        Executa uma sub-ordem em uma exchange específica.
        
        Args:
            sub_order: A sub-ordem a ser executada.
        """
        try:
            # Executar a ordem na exchange
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
                f"Ordem executada na exchange {sub_order.exchange}: "
                f"Quantidade: {sub_order.quantity}, "
                f"Preenchida: {sub_order.filled_quantity}, "
                f"Preço médio: {sub_order.average_price}"
            )
            
        except Exception as e:
            logger.error(f"Erro ao executar ordem na exchange {sub_order.exchange}: {str(e)}")
            sub_order.status = "failed"
            sub_order.error_message = str(e)
            raise

    async def _gather_market_data(
        self, trading_pair: str, exchanges: List[str]
    ) -> Dict[str, dict]:
        """
        Coleta dados de mercado de múltiplas exchanges.
        
        Args:
            trading_pair: Par de trading.
            exchanges: Lista de exchanges para coletar dados.
            
        Returns:
            Dict[str, dict]: Dicionário com dados de mercado por exchange.
        """
        market_data = {}
        tasks = []
        
        for exchange in exchanges:
            tasks.append(self._get_exchange_market_data(exchange, trading_pair))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            exchange = exchanges[i]
            if isinstance(result, Exception):
                logger.warning(f"Não foi possível obter dados de mercado da exchange {exchange}: {str(result)}")
            else:
                market_data[exchange] = result
        
        return market_data

    async def _get_exchange_market_data(self, exchange: str, trading_pair: str) -> dict:
        """
        Obtém dados de mercado de uma exchange específica.
        
        Args:
            exchange: Nome da exchange.
            trading_pair: Par de trading.
            
        Returns:
            dict: Dados de mercado da exchange.
        """
        return await self.exchange_client.get_market_data(
            exchange=exchange,
            trading_pair=trading_pair,
        )

    async def _calculate_optimal_allocation(
        self,
        total_quantity: float,
        side: str,
        market_data: Dict[str, dict],
        params: SmartRoutingParameters,
    ) -> Dict[str, float]:
        """
        Calcula a alocação ótima entre as exchanges disponíveis.
        
        Args:
            total_quantity: Quantidade total a ser executada.
            side: Lado da ordem ('buy' ou 'sell').
            market_data: Dados de mercado coletados das exchanges.
            params: Parâmetros de smart routing.
            
        Returns:
            Dict[str, float]: Alocação de quantidade por exchange.
        """
        if not market_data:
            logger.warning("Nenhum dado de mercado disponível para cálculo de alocação")
            return {}
        
        # Filtrar exchanges com liquidez insuficiente
        valid_exchanges = {}
        for exchange, data in market_data.items():
            liquidity = self._calculate_available_liquidity(data, side, total_quantity)
            if liquidity > 0:
                valid_exchanges[exchange] = {
                    "liquidity": liquidity,
                    "price": self._get_effective_price(data, side),
                    "fee": data.get("taker_fee", 0.001),  # Default 0.1% se não disponível
                }
        
        if not valid_exchanges:
            logger.warning("Nenhuma exchange com liquidez suficiente")
            return {}
        
        # Ordena exchanges pelo melhor preço (considerando taxas)
        sorted_exchanges = sorted(
            valid_exchanges.items(),
            key=lambda x: x[1]["price"] * (1 + x[1]["fee"]) if side == "buy" else x[1]["price"] * (1 - x[1]["fee"]),
            reverse=(side == "sell")  # Maior preço primeiro para vendas, menor para compras
        )
        
        # Alocar com base na estratégia escolhida
        allocation = {}
        
        if params.allocation_strategy == "best_price":
            # Aloca para as exchanges com melhor preço primeiro
            remaining = total_quantity
            for exchange, data in sorted_exchanges:
                if remaining <= 0:
                    break
                    
                amount = min(remaining, data["liquidity"])
                if amount > 0:
                    allocation[exchange] = amount
                    remaining -= amount
            
            # Se ainda tiver quantidade restante, tentar distribuir proporcionalmente
            if remaining > 0 and allocation:
                logger.warning(f"Liquidez insuficiente para alocação completa, faltando {remaining}")
                # Distribuir o restante proporcionalmente à alocação inicial
                total_allocated = sum(allocation.values())
                for exchange in allocation:
                    ratio = allocation[exchange] / total_allocated
                    additional = remaining * ratio
                    allocation[exchange] += additional
        
        elif params.allocation_strategy == "proportional":
            # Aloca proporcionalmente à liquidez disponível
            total_liquidity = sum(data["liquidity"] for _, data in valid_exchanges.items())
            for exchange, data in valid_exchanges.items():
                ratio = data["liquidity"] / total_liquidity
                allocation[exchange] = total_quantity * ratio
        
        else:  # balanced ou estratégia padrão
            # Equilibra entre preço e liquidez
            # Calcula um score composto para cada exchange
            total_score = 0
            scores = {}
            
            for exchange, data in valid_exchanges.items():
                # Normalizar preço (0-1 onde 1 é melhor)
                best_price = sorted_exchanges[0][1]["price"]
                price_factor = best_price / data["price"] if side == "buy" else data["price"] / best_price
                
                # Normalizar liquidez (0-1 onde 1 é melhor)
                max_liquidity = max(e[1]["liquidity"] for e in valid_exchanges.items())
                liquidity_factor = data["liquidity"] / max_liquidity
                
                # Calcular score composto (peso maior para preço)
                score = (price_factor * 0.7) + (liquidity_factor * 0.3)
                scores[exchange] = score
                total_score += score
            
            # Alocar proporcionalmente ao score
            for exchange, score in scores.items():
                ratio = score / total_score
                allocation[exchange] = total_quantity * ratio
        
        # Arredondar para 8 casas decimais
        for exchange in allocation:
            allocation[exchange] = round(allocation[exchange], 8)
        
        logger.info(f"Alocação calculada: {allocation}")
        return allocation

    def _calculate_available_liquidity(self, market_data: dict, side: str, total_quantity: float) -> float:
        """
        Calcula a liquidez disponível para uma ordem específica.
        
        Args:
            market_data: Dados de mercado da exchange.
            side: Lado da ordem ('buy' ou 'sell').
            total_quantity: Quantidade total da ordem.
            
        Returns:
            float: Liquidez disponível.
        """
        # Usar book_depth se disponível para cálculo preciso
        book_depth = market_data.get("book_depth", {})
        if book_depth:
            side_data = book_depth.get("asks" if side == "buy" else "bids", [])
            liquidity = sum(qty for _, qty in side_data)
            return min(liquidity, total_quantity)
        
        # Usar 24h volume como proxy se book_depth não disponível
        volume_24h = market_data.get("volume_24h", 0)
        # Estimar que podemos executar até 5% do volume diário sem impacto significativo
        return min(volume_24h * 0.05, total_quantity)

    def _get_effective_price(self, market_data: dict, side: str) -> float:
        """
        Obtém o preço efetivo para uma ordem.
        
        Args:
            market_data: Dados de mercado da exchange.
            side: Lado da ordem ('buy' ou 'sell').
            
        Returns:
            float: Preço efetivo.
        """
        # Para compras, usamos o preço de ask (mais alto)
        # Para vendas, usamos o preço de bid (mais baixo)
        if side == "buy":
            return market_data.get("ask", market_data.get("last_price", 0))
        else:
            return market_data.get("bid", market_data.get("last_price", 0))

    async def calculate_expected_metrics(
        self, order: Order, params: SmartRoutingParameters
    ) -> dict:
        """
        Calcula métricas esperadas para a execução Smart Routing.
        
        Args:
            order: A ordem a ser executada.
            params: Parâmetros específicos do Smart Routing.
            
        Returns:
            dict: Dicionário com métricas estimadas.
        """
        try:
            # Obter dados de mercado
            market_data = await self._gather_market_data(order.trading_pair, params.exchanges)
            
            # Calcular alocação
            allocation = await self._calculate_optimal_allocation(
                order.quantity,
                order.side,
                market_data,
                params
            )
            
            # Calcular preço médio esperado
            weighted_price = 0
            total_allocated = sum(allocation.values())
            
            for exchange, quantity in allocation.items():
                if exchange in market_data:
                    price = self._get_effective_price(market_data[exchange], order.side)
                    fee = market_data[exchange].get("taker_fee", 0.001)
                    
                    # Ajustar preço para incluir taxa
                    effective_price = price * (1 + fee) if order.side == "buy" else price * (1 - fee)
                    weighted_price += (quantity / total_allocated) * effective_price
            
            return {
                "allocation": allocation,
                "exchanges_used": len(allocation),
                "expected_average_price": weighted_price,
                "estimated_fees": sum(
                    market_data.get(ex, {}).get("taker_fee", 0.001) * qty * weighted_price
                    for ex, qty in allocation.items()
                ),
                "execution_mode": "parallel" if params.execute_parallel else "sequential",
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular métricas Smart Routing esperadas: {str(e)}")
            return {
                "error": str(e),
                "exchanges": params.exchanges,
            }