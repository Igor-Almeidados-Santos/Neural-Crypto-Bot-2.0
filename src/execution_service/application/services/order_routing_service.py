"""
Serviço de roteamento de ordens.

Este módulo implementa o serviço responsável por determinar a melhor
exchange e estratégia de execução para uma ordem.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.execution_service.domain.entities.order import Order
from src.execution_service.domain.value_objects.execution_parameters import (
    DirectParameters,
    ExecutionParameters,
    SmartRoutingParameters,
)
from src.execution_service.infrastructure.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)


class OrderRoutingService:
    """
    Serviço de roteamento de ordens.
    
    Este serviço é responsável por determinar a melhor exchange e estratégia
    de execução para uma ordem, com base em critérios como liquidez, preço,
    taxas e outros fatores.
    """
    
    def __init__(self, exchange_client: ExchangeClient, config: Dict):
        """
        Inicializa o serviço de roteamento.
        
        Args:
            exchange_client: Cliente para comunicação com exchanges.
            config: Configuração do serviço.
        """
        self.exchange_client = exchange_client
        self.config = config
        self.default_exchanges = config.get("default_exchanges", [])
        self.routing_rules = config.get("routing_rules", {})
        self.exchange_priority = config.get("exchange_priority", {})
    
    async def route_order(
        self, order: Order
    ) -> Tuple[List[str], str, ExecutionParameters]:
        """
        Determina a rota ótima para uma ordem.
        
        Args:
            order: Ordem a ser roteada.
            
        Returns:
            Tuple contendo:
            - List[str]: Lista de exchanges para execução.
            - str: Algoritmo de execução recomendado.
            - ExecutionParameters: Parâmetros de execução recomendados.
        """
        # Se a ordem já tem uma exchange especificada, usá-la
        if order.exchange and order.exchange != "auto":
            logger.info(f"Usando exchange especificada na ordem: {order.exchange}")
            return [order.exchange], self._determine_algorithm(order, [order.exchange]), self._create_parameters(order, [order.exchange])
        
        # Obter exchanges disponíveis para o par de trading
        available_exchanges = await self._get_available_exchanges(order.trading_pair)
        if not available_exchanges:
            logger.warning(f"Nenhuma exchange disponível para o par {order.trading_pair}, usando defaults")
            available_exchanges = self.default_exchanges
        
        # Se não houver exchanges disponíveis, lançar exceção
        if not available_exchanges:
            raise ValueError(f"Nenhuma exchange disponível para o par {order.trading_pair}")
        
        # Verificar se a ordem tem requisitos específicos
        if order.metadata and "preferred_exchanges" in order.metadata:
            preferred = order.metadata["preferred_exchanges"]
            if isinstance(preferred, list) and preferred:
                # Filtrar exchanges preferidas que estão disponíveis
                filtered = [ex for ex in preferred if ex in available_exchanges]
                if filtered:
                    logger.info(f"Usando exchanges preferidas: {filtered}")
                    return filtered, self._determine_algorithm(order, filtered), self._create_parameters(order, filtered)
        
        # Aplicar regras de roteamento específicas por par
        pair_rules = self.routing_rules.get(order.trading_pair, {})
        if pair_rules:
            # Verificar regras por tamanho de ordem
            size_rules = pair_rules.get("size_rules", [])
            for rule in size_rules:
                min_size = rule.get("min_size", 0)
                max_size = rule.get("max_size", float("inf"))
                if min_size <= order.quantity <= max_size:
                    if "exchanges" in rule:
                        exchanges = [ex for ex in rule["exchanges"] if ex in available_exchanges]
                        if exchanges:
                            algorithm = rule.get("algorithm", self._determine_algorithm(order, exchanges))
                            logger.info(f"Aplicando regra de tamanho: {exchanges}, algoritmo: {algorithm}")
                            return exchanges, algorithm, self._create_parameters(order, exchanges, algorithm)
        
        # Se chegamos aqui, não há regras específicas - usar smart routing ou melhor exchange
        if len(available_exchanges) > 1 and order.quantity >= self.config.get("smart_routing_min_size", 0):
            # Smart routing entre múltiplas exchanges
            logger.info(f"Usando smart routing entre exchanges: {available_exchanges}")
            return available_exchanges, "smart_routing", self._create_parameters(order, available_exchanges, "smart_routing")
        else:
            # Usar a melhor exchange
            best_exchange = await self._get_best_exchange(order, available_exchanges)
            logger.info(f"Usando melhor exchange: {best_exchange}")
            return [best_exchange], self._determine_algorithm(order, [best_exchange]), self._create_parameters(order, [best_exchange])
    
    async def _get_available_exchanges(self, trading_pair: str) -> List[str]:
        """
        Obtém as exchanges disponíveis para um par de trading.
        
        Args:
            trading_pair: Par de trading.
            
        Returns:
            List[str]: Lista de exchanges disponíveis.
        """
        available = []
        for exchange in self.exchange_client.exchanges.keys():
            try:
                # Verificar se o par é suportado na exchange
                market_data = await self.exchange_client.get_market_data(exchange, trading_pair)
                if market_data and market_data.get("last_price", 0) > 0:
                    available.append(exchange)
            except Exception as e:
                logger.warning(f"Erro ao verificar disponibilidade de {trading_pair} em {exchange}: {str(e)}")
        
        # Ordenar por prioridade
        return sorted(available, key=lambda ex: self.exchange_priority.get(ex, 999))
    
    async def _get_best_exchange(self, order: Order, available_exchanges: List[str]) -> str:
        """
        Determina a melhor exchange para uma ordem.
        
        Args:
            order: Ordem a ser executada.
            available_exchanges: Lista de exchanges disponíveis.
            
        Returns:
            str: ID da melhor exchange.
        """
        if not available_exchanges:
            raise ValueError("Lista de exchanges disponíveis está vazia")
        
        if len(available_exchanges) == 1:
            return available_exchanges[0]
        
        # Coletar dados de mercado para comparação
        exchange_data = {}
        for exchange in available_exchanges:
            try:
                market_data = await self.exchange_client.get_market_data(exchange, order.trading_pair)
                exchange_data[exchange] = market_data
            except Exception as e:
                logger.warning(f"Erro ao obter dados de mercado para {order.trading_pair} em {exchange}: {str(e)}")
        
        if not exchange_data:
            logger.warning(f"Não foi possível obter dados de mercado para nenhuma exchange, usando a primeira disponível")
            return available_exchanges[0]
        
        # Pontuação baseada em múltiplos fatores
        scores = {}
        for exchange, data in exchange_data.items():
            score = 0
            
            # Preço (para compras, menor é melhor; para vendas, maior é melhor)
            price = data.get("ask" if order.side == "buy" else "bid", 0)
            if price > 0:
                # Normalizar preço em relação ao melhor preço
                best_price = min([d.get("ask", float("inf")) for d in exchange_data.values()]) if order.side == "buy" else \
                           max([d.get("bid", 0) for d in exchange_data.values()])
                
                if best_price > 0:
                    price_score = (best_price / price) if order.side == "buy" else (price / best_price)
                    score += price_score * 5  # Peso maior para preço
            
            # Liquidez
            volume = data.get("volume_24h", 0)
            max_volume = max([d.get("volume_24h", 0) for d in exchange_data.values()])
            if max_volume > 0:
                liquidity_score = volume / max_volume
                score += liquidity_score * 3  # Peso médio para liquidez
            
            # Profundidade do livro
            depth = data.get("bid_depth" if order.side == "sell" else "ask_depth", 0)
            max_depth = max([d.get("bid_depth" if order.side == "sell" else "ask_depth", 0) for d in exchange_data.values()])
            if max_depth > 0:
                depth_score = depth / max_depth
                score += depth_score * 4  # Peso alto para profundidade
            
            # Taxas (menor é melhor)
            fee = data.get("taker_fee", 0.001)
            min_fee = min([d.get("taker_fee", 0.001) for d in exchange_data.values()])
            if min_fee > 0:
                fee_score = min_fee / fee
                score += fee_score * 2  # Peso menor para taxas
            
            # Prioridade configurada
            priority_score = 1 - (self.exchange_priority.get(exchange, 999) / 1000)
            score += priority_score * 2  # Peso menor para prioridade
            
            scores[exchange] = score
        
        # Escolher a exchange com maior pontuação
        best_exchange = max(scores, key=scores.get)
        logger.info(f"Pontuações de exchanges: {scores}, melhor: {best_exchange}")
        
        return best_exchange
    
    def _determine_algorithm(self, order: Order, exchanges: List[str]) -> str:
        """
        Determina o melhor algoritmo de execução para uma ordem.
        
        Args:
            order: Ordem a ser executada.
            exchanges: Exchanges disponíveis.
            
        Returns:
            str: Nome do algoritmo recomendado.
        """
        # Verificar se há um algoritmo especificado na ordem
        if order.metadata and "algorithm" in order.metadata:
            return order.metadata["algorithm"]
        
        # Regras para determinação de algoritmo
        quantity = order.quantity
        
        # Para ordens pequenas, executar diretamente
        if quantity < self.config.get("twap_min_size", 1.0):
            return "direct"
        
        # Para ordens médias, usar TWAP
        elif quantity < self.config.get("iceberg_min_size", 5.0):
            return "twap"
        
        # Para ordens grandes, usar Iceberg
        else:
            return "iceberg"
    
    def _create_parameters(
        self, order: Order, exchanges: List[str], algorithm: Optional[str] = None
    ) -> ExecutionParameters:
        """
        Cria parâmetros de execução para um algoritmo.
        
        Args:
            order: Ordem a ser executada.
            exchanges: Exchanges disponíveis.
            algorithm: Algoritmo de execução (opcional).
            
        Returns:
            ExecutionParameters: Parâmetros de execução.
        """
        if algorithm is None:
            algorithm = self._determine_algorithm(order, exchanges)
        
        # Verificar se há parâmetros na ordem
        if order.metadata and "execution_params" in order.metadata:
            params = order.metadata["execution_params"]
            if algorithm in params:
                return params[algorithm]
        
        # Criar parâmetros padrão
        if algorithm == "smart_routing":
            return SmartRoutingParameters(
                exchanges=exchanges,
                allocation_strategy="balanced",
                execute_parallel=True,
                retry_failed=True,
                max_price_deviation=0.02,
            )
        elif algorithm == "twap":
            # Parâmetros TWAP baseados no tamanho da ordem
            duration_minutes = 30
            slices = 5
            
            if order.quantity >= self.config.get("large_order_size", 10.0):
                duration_minutes = 60
                slices = 10
            
            from src.execution_service.domain.value_objects.execution_parameters import TwapParameters
            return TwapParameters(
                duration_minutes=duration_minutes,
                num_slices=slices,
                max_participation_rate=0.3,
            )
        elif algorithm == "iceberg":
            # Parâmetros Iceberg baseados no tamanho da ordem
            display_size = min(1.0, order.quantity * 0.2)  # 20% do total, no máximo 1.0
            
            from src.execution_service.domain.value_objects.execution_parameters import IcebergParameters
            return IcebergParameters(
                display_size=display_size,
                size_variance=0.2,
                interval_seconds=30,
                interval_variance=0.3,
                price_adjustment_threshold=0.01,
                continue_on_failure=True,
            )
        else:  # direct
            from src.execution_service.domain.value_objects.execution_parameters import DirectParameters
            return DirectParameters(
                retry_on_failure=True,
                max_retries=3,
                retry_delay_seconds=1.0,
                validate_market_price=True,
                max_price_deviation=0.05,
            )