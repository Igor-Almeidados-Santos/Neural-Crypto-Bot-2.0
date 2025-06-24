"""
Serviço de otimização de execução.

Este módulo implementa o serviço responsável por otimizar os parâmetros
de execução de ordens para diferentes algoritmos.
"""
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from execution_service.domain.entities.order import Order
from execution_service.domain.value_objects.execution_parameters import (
    ExecutionParameters,
    IcebergParameters,
    SmartRoutingParameters,
    TwapParameters,
)
from execution_service.infrastructure.exchange_client import ExchangeClient

logger = logging.getLogger(__name__)


class ExecutionOptimizationService:
    """
    Serviço de otimização de execução.
    
    Este serviço é responsável por otimizar os parâmetros de execução de
    ordens para diferentes algoritmos, com base em dados históricos, condições
    de mercado e feedback de execuções anteriores.
    """
    
    def __init__(self, exchange_client: ExchangeClient, config: Dict):
        """
        Inicializa o serviço de otimização.
        
        Args:
            exchange_client: Cliente para comunicação com exchanges.
            config: Configuração do serviço.
        """
        self.exchange_client = exchange_client
        self.config = config
        
        # Configurações para otimização de TWAP
        self.twap_config = config.get("twap", {})
        self.twap_min_duration = self.twap_config.get("min_duration_minutes", 5)
        self.twap_max_duration = self.twap_config.get("max_duration_minutes", 240)
        self.twap_min_slices = self.twap_config.get("min_slices", 2)
        self.twap_max_slices = self.twap_config.get("max_slices", 60)
        
        # Configurações para otimização de Iceberg
        self.iceberg_config = config.get("iceberg", {})
        self.iceberg_min_display = self.iceberg_config.get("min_display_percent", 0.05)
        self.iceberg_max_display = self.iceberg_config.get("max_display_percent", 0.3)
        
        # Limiares de volatilidade
        self.volatility_thresholds = config.get("volatility_thresholds", {
            "low": 0.01,  # < 1% por dia
            "medium": 0.025,  # 1-2.5% por dia
            "high": 0.05,  # 2.5-5% por dia
            "extreme": 0.1,  # > 5% por dia
        })
    
    async def optimize_twap_parameters(
        self, order: Order, base_params: Optional[TwapParameters] = None
    ) -> TwapParameters:
        """
        Otimiza os parâmetros para o algoritmo TWAP.
        
        Args:
            order: Ordem a ser executada.
            base_params: Parâmetros base para otimização (opcional).
            
        Returns:
            TwapParameters: Parâmetros otimizados.
        """
        # Se houver parâmetros base, começar com eles
        if base_params:
            params = TwapParameters(**base_params.dict())
        else:
            # Parâmetros padrão
            params = TwapParameters(
                duration_minutes=30,
                num_slices=5,
                max_participation_rate=0.3,
            )
        
        try:
            # Obter dados de mercado
            market_data = await self.exchange_client.get_market_data(
                order.exchange, order.trading_pair
            )
            
            # Ajustar com base na volatilidade
            volatility = market_data.get("volatility_24h", 0.02)  # 2% padrão
            
            # Ajustar duração e número de slices com base na volatilidade
            if volatility <= self.volatility_thresholds["low"]:
                # Mercado de baixa volatilidade - menos slices, duração mais curta
                duration = max(self.twap_min_duration, min(30, params.duration_minutes))
                slices = max(self.twap_min_slices, min(5, params.num_slices))
                
            elif volatility <= self.volatility_thresholds["medium"]:
                # Volatilidade média - valores moderados
                duration = max(self.twap_min_duration, min(60, params.duration_minutes))
                slices = max(3, min(10, params.num_slices))
                
            elif volatility <= self.volatility_thresholds["high"]:
                # Alta volatilidade - mais slices, duração mais longa
                duration = max(30, min(120, params.duration_minutes))
                slices = max(5, min(15, params.num_slices))
                
            else:  # volatilidade extrema
                # Volatilidade extrema - máxima fragmentação
                duration = max(60, min(self.twap_max_duration, params.duration_minutes))
                slices = max(10, min(self.twap_max_slices, params.num_slices))
            
            # Ajustar com base no tamanho da ordem e volume do mercado
            volume_24h = market_data.get("volume_24h", 0)
            if volume_24h > 0:
                # Calcular a porcentagem que a ordem representa do volume diário
                order_percentage = (order.quantity / volume_24h) * 100
                
                # Ajustar com base na porcentagem
                if order_percentage > 5:  # Ordem grande (>5% do volume diário)
                    duration = max(duration, 120)  # Pelo menos 2 horas
                    slices = max(slices, 20)  # Pelo menos 20 slices
                elif order_percentage > 1:  # Ordem média (1-5% do volume diário)
                    duration = max(duration, 60)  # Pelo menos 1 hora
                    slices = max(slices, 10)  # Pelo menos 10 slices
            
            # Ajustar a taxa de participação
            # Taxa de participação é a porcentagem do volume de mercado que queremos representar por slice
            if volume_24h > 0:
                hourly_volume = volume_24h / 24
                slice_duration_hours = (duration / 60) / slices
                slice_volume = hourly_volume * slice_duration_hours
                
                if slice_volume > 0:
                    participation_rate = min(0.3, (order.quantity / slices) / slice_volume)
                else:
                    participation_rate = 0.3
            else:
                participation_rate = 0.3
            
            # Atualizar parâmetros
            params.duration_minutes = duration
            params.num_slices = slices
            params.max_participation_rate = participation_rate
            
            logger.info(
                f"Parâmetros TWAP otimizados para {order.trading_pair}: "
                f"duração={duration}min, slices={slices}, participation_rate={participation_rate:.2f}"
            )
            
            return params
            
        except Exception as e:
            logger.error(f"Erro ao otimizar parâmetros TWAP: {str(e)}")
            # Em caso de erro, retornar os parâmetros originais
            return params
    
    async def optimize_iceberg_parameters(
        self, order: Order, base_params: Optional[IcebergParameters] = None
    ) -> IcebergParameters:
        """
        Otimiza os parâmetros para o algoritmo Iceberg.
        
        Args:
            order: Ordem a ser executada.
            base_params: Parâmetros base para otimização (opcional).
            
        Returns:
            IcebergParameters: Parâmetros otimizados.
        """
        # Se houver parâmetros base, começar com eles
        if base_params:
            params = IcebergParameters(**base_params.dict())
        else:
            # Parâmetros padrão
            display_size = min(1.0, order.quantity * 0.2)  # 20% do total, no máximo 1.0
            params = IcebergParameters(
                display_size=display_size,
                size_variance=0.2,
                interval_seconds=30,
                interval_variance=0.3,
                price_adjustment_threshold=0.01,
                continue_on_failure=True,
            )
        
        try:
            # Obter dados de mercado
            market_data = await self.exchange_client.get_market_data(
                order.exchange, order.trading_pair
            )
            
            # Ajustar com base na volatilidade
            volatility = market_data.get("volatility_24h", 0.02)  # 2% padrão
            
            # Ajustar tamanho de exibição com base na volatilidade
            if volatility <= self.volatility_thresholds["low"]:
                # Mercado calmo - pode mostrar mais
                display_percent = min(self.iceberg_max_display, max(self.iceberg_min_display, 0.25))
                variance = 0.1
                interval = 20
                price_threshold = 0.005
                
            elif volatility <= self.volatility_thresholds["medium"]:
                # Volatilidade média
                display_percent = min(self.iceberg_max_display, max(self.iceberg_min_display, 0.15))
                variance = 0.2
                interval = 30
                price_threshold = 0.01
                
            elif volatility <= self.volatility_thresholds["high"]:
                # Alta volatilidade
                display_percent = min(self.iceberg_max_display, max(self.iceberg_min_display, 0.1))
                variance = 0.3
                interval = 45
                price_threshold = 0.015
                
            else:  # volatilidade extrema
                # Volatilidade extrema - mostrar o mínimo
                display_percent = self.iceberg_min_display
                variance = 0.4
                interval = 60
                price_threshold = 0.02
            
            # Ajustar com base na profundidade do livro de ordens
            book_depth = market_data.get("book_depth", {})
            side_depth = 0
            
            if book_depth:
                side_key = "bids" if order.side == "sell" else "asks"
                side_data = book_depth.get(side_key, [])
                
                if side_data:
                    # Calcular a profundidade total para o lado relevante
                    side_depth = sum(qty for _, qty in side_data)
            
            # Se tivermos dados de profundidade, ajustar o tamanho de exibição
            if side_depth > 0:
                # Não mostrar mais do que uma porcentagem da profundidade
                max_display_by_depth = side_depth * 0.1  # 10% da profundidade
                
                # Mas não menos que o mínimo configurado
                display_size = min(
                    max_display_by_depth,
                    order.quantity * display_percent
                )
            else:
                # Sem dados de profundidade, usar a porcentagem padrão
                display_size = order.quantity * display_percent
            
            # Limitar o tamanho mínimo e máximo
            min_size = self.iceberg_config.get("min_display_size", 0.001)
            display_size = max(min_size, min(display_size, order.quantity * self.iceberg_max_display))
            
            # Ajustar intervalo com base no volume
            volume_24h = market_data.get("volume_24h", 0)
            if volume_24h > 0:
                # Calcular a porcentagem que a ordem representa do volume diário
                order_percentage = (order.quantity / volume_24h) * 100
                
                # Ajustar o intervalo
                if order_percentage > 5:  # Ordem grande
                    interval = max(interval, 90)  # Pelo menos 90 segundos
                elif order_percentage > 1:  # Ordem média
                    interval = max(interval, 60)  # Pelo menos 60 segundos
            
            # Atualizar parâmetros
            params.display_size = display_size
            params.size_variance = variance
            params.interval_seconds = interval
            params.interval_variance = variance
            params.price_adjustment_threshold = price_threshold
            
            logger.info(
                f"Parâmetros Iceberg otimizados para {order.trading_pair}: "
                f"display_size={display_size:.6f}, variance={variance:.2f}, "
                f"interval={interval}s, price_threshold={price_threshold:.4f}"
            )
            
            return params
            
        except Exception as e:
            logger.error(f"Erro ao otimizar parâmetros Iceberg: {str(e)}")
            # Em caso de erro, retornar os parâmetros originais
            return params
    
    async def optimize_smart_routing_parameters(
        self, order: Order, available_exchanges: List[str], base_params: Optional[SmartRoutingParameters] = None
    ) -> SmartRoutingParameters:
        """
        Otimiza os parâmetros para o algoritmo Smart Routing.
        
        Args:
            order: Ordem a ser executada.
            available_exchanges: Lista de exchanges disponíveis.
            base_params: Parâmetros base para otimização (opcional).
            
        Returns:
            SmartRoutingParameters: Parâmetros otimizados.
        """
        # Se houver parâmetros base, começar com eles
        if base_params:
            params = SmartRoutingParameters(**base_params.dict())
        else:
            # Parâmetros padrão
            params = SmartRoutingParameters(
                exchanges=available_exchanges,
                allocation_strategy="balanced",
                execute_parallel=True,
                retry_failed=True,
                max_price_deviation=0.02,
            )
        
        try:
            # Coletar dados de mercado para todas as exchanges
            market_data = {}
            for exchange in available_exchanges:
                try:
                    data = await self.exchange_client.get_market_data(exchange, order.trading_pair)
                    market_data[exchange] = data
                except Exception as e:
                    logger.warning(f"Erro ao obter dados de mercado para {order.trading_pair} em {exchange}: {str(e)}")
            
            if not market_data:
                logger.warning("Não foi possível obter dados de mercado para nenhuma exchange")
                return params
            
            # Determinar a melhor estratégia de alocação com base no contexto
            
            # Verificar se há uma grande diferença de preço entre as exchanges
            prices = [data.get("last_price", 0) for data in market_data.values() if data.get("last_price", 0) > 0]
            if prices:
                max_price = max(prices)
                min_price = min(prices)
                price_diff_percent = (max_price - min_price) / min_price if min_price > 0 else 0
                
                # Se houver diferença significativa de preço
                if price_diff_percent > 0.01:  # > 1%
                    # Para compras, priorizar exchanges com preços mais baixos
                    if order.side == "buy":
                        params.allocation_strategy = "best_price"
                    # Para vendas, também priorizar exchanges com preços mais altos
                    else:
                        params.allocation_strategy = "best_price"
                    
                    logger.info(f"Usando estratégia best_price devido à diferença de preço de {price_diff_percent:.2%}")
                else:
                    # Sem grandes diferenças de preço, distribuir baseado na liquidez
                    params.allocation_strategy = "proportional"
                    logger.info("Usando estratégia proportional devido à pequena diferença de preço")
            
            # Verificar se o tamanho da ordem é grande em relação à liquidez
            volumes = {ex: data.get("volume_24h", 0) for ex, data in market_data.items()}
            total_volume = sum(volumes.values())
            
            if total_volume > 0:
                order_percentage = (order.quantity / total_volume) * 100
                
                # Se a ordem for grande, usar estratégia balanceada
                if order_percentage > 1:  # > 1% do volume total
                    params.allocation_strategy = "balanced"
                    logger.info(f"Usando estratégia balanced devido ao tamanho da ordem ({order_percentage:.2f}% do volume)")
            
            # Ajustar max_price_deviation com base na volatilidade
            max_volatility = max([data.get("volatility_24h", 0.02) for data in market_data.values()])
            
            # Definir desvio máximo com base na volatilidade
            if max_volatility <= self.volatility_thresholds["low"]:
                params.max_price_deviation = 0.01  # 1%
            elif max_volatility <= self.volatility_thresholds["medium"]:
                params.max_price_deviation = 0.02  # 2%
            elif max_volatility <= self.volatility_thresholds["high"]:
                params.max_price_deviation = 0.03  # 3%
            else:
                params.max_price_deviation = 0.05  # 5%
            
            logger.info(
                f"Parâmetros Smart Routing otimizados para {order.trading_pair}: "
                f"estratégia={params.allocation_strategy}, "
                f"exchanges={params.exchanges}, "
                f"max_deviation={params.max_price_deviation:.2%}"
            )
            
            return params
            
        except Exception as e:
            logger.error(f"Erro ao otimizar parâmetros Smart Routing: {str(e)}")
            # Em caso de erro, retornar os parâmetros originais
            return params
    
    def adapt_parameters_to_market_conditions(
        self, algorithm: str, params: ExecutionParameters, market_data: Dict
    ) -> ExecutionParameters:
        """
        Adapta os parâmetros de execução às condições atuais de mercado.
        
        Args:
            algorithm: Algoritmo de execução.
            params: Parâmetros de execução.
            market_data: Dados de mercado.
            
        Returns:
            ExecutionParameters: Parâmetros adaptados.
        """
        # Implementação específica para cada algoritmo
        if algorithm == "twap" and isinstance(params, TwapParameters):
            return self._adapt_twap_parameters(params, market_data)
        elif algorithm == "iceberg" and isinstance(params, IcebergParameters):
            return self._adapt_iceberg_parameters(params, market_data)
        elif algorithm == "smart_routing" and isinstance(params, SmartRoutingParameters):
            return self._adapt_smart_routing_parameters(params, market_data)
        else:
            return params
    
    def _adapt_twap_parameters(self, params: TwapParameters, market_data: Dict) -> TwapParameters:
        """
        Adapta os parâmetros TWAP às condições de mercado.
        
        Args:
            params: Parâmetros TWAP.
            market_data: Dados de mercado.
            
        Returns:
            TwapParameters: Parâmetros adaptados.
        """
        # Copiar parâmetros para não modificar o original
        adapted_params = TwapParameters(**params.dict())
        
        # Verificar se há condições de mercado extremas
        volatility = market_data.get("volatility_24h", 0.02)
        spread = market_data.get("spread", 0.001)
        
        # Condições de alta volatilidade
        if volatility > self.volatility_thresholds["extreme"]:
            logger.warning(f"Volatilidade extrema detectada: {volatility:.2%}")
            # Aumentar o número de slices e duração para diluir o impacto
            adapted_params.num_slices = min(self.twap_max_slices, int(adapted_params.num_slices * 1.5))
            adapted_params.duration_minutes = min(self.twap_max_duration, int(adapted_params.duration_minutes * 1.5))
        
        # Spread anormalmente alto
        if spread > 0.01:  # > 1%
            logger.warning(f"Spread alto detectado: {spread:.2%}")
            # Ajustar a taxa de participação para evitar mover o mercado
            adapted_params.max_participation_rate = min(adapted_params.max_participation_rate, 0.1)
        
        return adapted_params
    
    def _adapt_iceberg_parameters(self, params: IcebergParameters, market_data: Dict) -> IcebergParameters:
        """
        Adapta os parâmetros Iceberg às condições de mercado.
        
        Args:
            params: Parâmetros Iceberg.
            market_data: Dados de mercado.
            
        Returns:
            IcebergParameters: Parâmetros adaptados.
        """
        # Copiar parâmetros para não modificar o original
        adapted_params = IcebergParameters(**params.dict())
        
        # Verificar se há condições de mercado extremas
        volatility = market_data.get("volatility_24h", 0.02)
        spread = market_data.get("spread", 0.001)
        
        # Condições de alta volatilidade
        if volatility > self.volatility_thresholds["extreme"]:
            logger.warning(f"Volatilidade extrema detectada: {volatility:.2%}")
            # Reduzir o tamanho de exibição
            adapted_params.display_size = max(params.display_size * 0.5, self.iceberg_config.get("min_display_size", 0.001))
            # Aumentar a variância para dificultar a detecção
            adapted_params.size_variance = min(0.5, params.size_variance * 1.5)
            # Aumentar o intervalo para espaçar mais as ordens
            adapted_params.interval_seconds = min(300, params.interval_seconds * 2)
            # Aumentar o limiar de ajuste de preço
            adapted_params.price_adjustment_threshold = min(0.05, params.price_adjustment_threshold * 2)
        
        # Spread anormalmente alto
        if spread > 0.01:  # > 1%
            logger.warning(f"Spread alto detectado: {spread:.2%}")
            # Aumentar o limiar de ajuste de preço para evitar reajustes frequentes
            adapted_params.price_adjustment_threshold = max(params.price_adjustment_threshold, spread * 2)
        
        return adapted_params
    
    def _adapt_smart_routing_parameters(self, params: SmartRoutingParameters, market_data: Dict) -> SmartRoutingParameters:
        """
        Adapta os parâmetros Smart Routing às condições de mercado.
        
        Args:
            params: Parâmetros Smart Routing.
            market_data: Dados de mercado agregados.
            
        Returns:
            SmartRoutingParameters: Parâmetros adaptados.
        """
        # Copiar parâmetros para não modificar o original
        adapted_params = SmartRoutingParameters(**params.dict())
        
        # Verificar se há condições de mercado extremas
        max_volatility = max([data.get("volatility_24h", 0.02) for ex, data in market_data.items() if isinstance(data, dict)])
        
        # Condições de alta volatilidade
        if max_volatility > self.volatility_thresholds["extreme"]:
            logger.warning(f"Volatilidade extrema detectada: {max_volatility:.2%}")
            # Aumentar o desvio máximo de preço
            adapted_params.max_price_deviation = min(0.1, params.max_price_deviation * 2)
            # Em condições extremas, priorizar preço sobre liquidez
            adapted_params.allocation_strategy = "best_price"
        
        return adapted_params