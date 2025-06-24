"""
Serviço de validação de dados.

Este módulo fornece funções para validar dados de mercado,
identificando valores anômalos, inconsistências e outros problemas
que podem afetar a qualidade dos dados utilizados pelo sistema.
"""
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Set, Tuple

from data_collection.domain.entities.candle import Candle, TimeFrame
from data_collection.domain.entities.orderbook import OrderBook
from data_collection.domain.entities.trade import Trade


logger = logging.getLogger(__name__)


class DataValidationService:
    """
    Serviço responsável por validar dados de mercado.
    
    Implementa funções para detectar anomalias, inconsistências
    e outros problemas nos dados coletados.
    """
    
    def __init__(
        self,
        max_price_deviation_percent: float = 10.0,
        max_volume_deviation_percent: float = 50.0,
        max_spread_percent: float = 5.0,
        min_orderbook_depth: int = 5,
        max_timestamp_delay_seconds: int = 30,
        enable_strict_validation: bool = False
    ):
        """
        Inicializa o serviço de validação com os parâmetros de configuração.
        
        Args:
            max_price_deviation_percent: Desvio máximo de preço permitido em porcentagem
            max_volume_deviation_percent: Desvio máximo de volume permitido em porcentagem
            max_spread_percent: Spread máximo permitido em porcentagem
            min_orderbook_depth: Profundidade mínima requerida para o orderbook
            max_timestamp_delay_seconds: Atraso máximo permitido em segundos
            enable_strict_validation: Se True, falhas de validação lançam exceções
        """
        self.max_price_deviation_percent = max_price_deviation_percent
        self.max_volume_deviation_percent = max_volume_deviation_percent
        self.max_spread_percent = max_spread_percent
        self.min_orderbook_depth = min_orderbook_depth
        self.max_timestamp_delay_seconds = max_timestamp_delay_seconds
        self.enable_strict_validation = enable_strict_validation
        
        # Cache para valores de referência
        self._price_reference: Dict[str, Decimal] = {}
        self._volume_reference: Dict[str, Dict[TimeFrame, Decimal]] = {}
        
    def validate_trade(self, trade: Trade) -> bool:
        """
        Valida os dados de uma transação.
        
        Args:
            trade: Entidade Trade a ser validada
            
        Returns:
            bool: True se os dados são válidos, False caso contrário
            
        Raises:
            ValueError: Se a validação falhar e enable_strict_validation for True
        """
        key = f"{trade.exchange}:{trade.trading_pair}"
        validation_errors = []
        
        # Valida o timestamp
        if not self._validate_timestamp(trade.timestamp):
            validation_errors.append(f"Timestamp inválido: {trade.timestamp}")
        
        # Valida preço e quantidade
        if trade.price <= 0:
            validation_errors.append(f"Preço inválido: {trade.price}")
        
        if trade.amount <= 0:
            validation_errors.append(f"Quantidade inválida: {trade.amount}")
        
        # Valida o custo (price * amount)
        expected_cost = trade.price * trade.amount
        if abs(trade.cost - expected_cost) > Decimal('0.000001') * expected_cost:
            validation_errors.append(f"Custo inconsistente: {trade.cost} != {trade.price} * {trade.amount}")
        
        # Verifica se o preço está dentro do desvio permitido
        if key in self._price_reference and self._price_reference[key] > 0:
            reference_price = self._price_reference[key]
            deviation = abs(trade.price - reference_price) / reference_price * 100
            
            if deviation > self.max_price_deviation_percent:
                validation_errors.append(
                    f"Desvio de preço excessivo: {trade.price} (referência: {reference_price}, desvio: {deviation:.2f}%)"
                )
        
        # Atualiza o preço de referência
        self._price_reference[key] = trade.price
        
        # Verifica se houve erros
        if validation_errors:
            error_message = f"Validação de trade falhou para {key}: {'; '.join(validation_errors)}"
            logger.warning(error_message)
            
            if self.enable_strict_validation:
                raise ValueError(error_message)
                
            return False
            
        return True
    
    def validate_orderbook(self, orderbook: OrderBook) -> bool:
        """
        Valida os dados de um orderbook.
        
        Args:
            orderbook: Entidade OrderBook a ser validada
            
        Returns:
            bool: True se os dados são válidos, False caso contrário
            
        Raises:
            ValueError: Se a validação falhar e enable_strict_validation for True
        """
        key = f"{orderbook.exchange}:{orderbook.trading_pair}"
        validation_errors = []
        
        # Valida o timestamp
        if not self._validate_timestamp(orderbook.timestamp):
            validation_errors.append(f"Timestamp inválido: {orderbook.timestamp}")
        
        # Valida a profundidade do orderbook
        if len(orderbook.bids) < self.min_orderbook_depth:
            validation_errors.append(f"Profundidade insuficiente de bids: {len(orderbook.bids)}")
            
        if len(orderbook.asks) < self.min_orderbook_depth:
            validation_errors.append(f"Profundidade insuficiente de asks: {len(orderbook.asks)}")
        
        # Valida a ordenação do orderbook
        if not self._validate_orderbook_sorting(orderbook):
            validation_errors.append("Ordenação incorreta do orderbook")
        
        # Valida os preços e quantidades
        if not self._validate_orderbook_values(orderbook):
            validation_errors.append("Valores inválidos no orderbook")
        
        # Valida o spread
        try:
            spread_percent = orderbook.spread_percentage
            if spread_percent < 0 or spread_percent > self.max_spread_percent:
                validation_errors.append(f"Spread anormal: {spread_percent:.2f}%")
        except Exception as e:
            validation_errors.append(f"Erro ao calcular spread: {str(e)}")
        
        # Verifica se houve erros
        if validation_errors:
            error_message = f"Validação de orderbook falhou para {key}: {'; '.join(validation_errors)}"
            logger.warning(error_message)
            
            if self.enable_strict_validation:
                raise ValueError(error_message)
                
            return False
            
        return True
    
    def validate_candle(self, candle: Candle, previous_candle: Optional[Candle] = None) -> bool:
        """
        Valida os dados de uma vela.
        
        Args:
            candle: Entidade Candle a ser validada
            previous_candle: Vela anterior para validação de continuidade (opcional)
            
        Returns:
            bool: True se os dados são válidos, False caso contrário
            
        Raises:
            ValueError: Se a validação falhar e enable_strict_validation for True
        """
        key = f"{candle.exchange}:{candle.trading_pair}:{candle.timeframe.value}"
        validation_errors = []
        
        # Valida o timestamp
        if not self._validate_timestamp(candle.timestamp):
            validation_errors.append(f"Timestamp inválido: {candle.timestamp}")
        
        # Valida os preços
        if candle.open <= 0 or candle.high <= 0 or candle.low <= 0 or candle.close <= 0:
            validation_errors.append("Preços inválidos (devem ser positivos)")
        
        # Valida a consistência dos preços (high >= open, close, low & low <= open, close)
        if not (candle.high >= candle.open and candle.high >= candle.close and
                candle.low <= candle.open and candle.low <= candle.close):
            validation_errors.append(
                f"Preços inconsistentes: open={candle.open}, high={candle.high}, "
                f"low={candle.low}, close={candle.close}"
            )
        
        # Valida o volume
        if candle.volume < 0:
            validation_errors.append(f"Volume inválido: {candle.volume}")
        
        # Verifica se o volume está dentro do desvio permitido
        volume_key = f"{candle.exchange}:{candle.trading_pair}"
        if (volume_key in self._volume_reference and 
            candle.timeframe in self._volume_reference[volume_key] and
            self._volume_reference[volume_key][candle.timeframe] > 0):
            
            reference_volume = self._volume_reference[volume_key][candle.timeframe]
            deviation = abs(candle.volume - reference_volume) / reference_volume * 100
            
            if deviation > self.max_volume_deviation_percent:
                validation_errors.append(
                    f"Desvio de volume excessivo: {candle.volume} "
                    f"(referência: {reference_volume}, desvio: {deviation:.2f}%)"
                )
        
        # Atualiza o volume de referência
        if volume_key not in self._volume_reference:
            self._volume_reference[volume_key] = {}
        self._volume_reference[volume_key][candle.timeframe] = candle.volume
        
        # Valida a continuidade com a vela anterior
        if previous_candle is not None:
            if not self._validate_candle_continuity(candle, previous_candle):
                validation_errors.append("Descontinuidade com a vela anterior")
        
        # Verifica se houve erros
        if validation_errors:
            error_message = f"Validação de candle falhou para {key}: {'; '.join(validation_errors)}"
            logger.warning(error_message)
            
            if self.enable_strict_validation:
                raise ValueError(error_message)
                
            return False
            
        return True
    
    def validate_candles_batch(self, candles: List[Candle]) -> Tuple[bool, List[Candle]]:
        """
        Valida um lote de velas e retorna apenas as válidas.
        
        Args:
            candles: Lista de entidades Candle a serem validadas
            
        Returns:
            Tuple[bool, List[Candle]]: Tupla contendo um indicador de sucesso e
                                      a lista de velas válidas
        """
        if not candles:
            return True, []
            
        valid_candles = []
        previous_candle = None
        all_valid = True
        
        # Ordena as velas por timestamp
        sorted_candles = sorted(candles, key=lambda x: x.timestamp)
        
        for candle in sorted_candles:
            if self.validate_candle(candle, previous_candle):
                valid_candles.append(candle)
                previous_candle = candle
            else:
                all_valid = False
        
        return all_valid, valid_candles
    
    def _validate_timestamp(self, timestamp: datetime) -> bool:
        """
        Valida se um timestamp está dentro do intervalo permitido.
        
        Args:
            timestamp: Timestamp a ser validado
            
        Returns:
            bool: True se o timestamp é válido, False caso contrário
        """
        now = datetime.utcnow()
        max_delay = timedelta(seconds=self.max_timestamp_delay_seconds)
        max_future = timedelta(seconds=5)  # Pequena margem para timestamps no futuro
        
        return timestamp <= now + max_future and timestamp >= now - max_delay
    
    def _validate_orderbook_sorting(self, orderbook: OrderBook) -> bool:
        """
        Valida se o orderbook está corretamente ordenado.
        
        Args:
            orderbook: Orderbook a ser validado
            
        Returns:
            bool: True se a ordenação está correta, False caso contrário
        """
        # Verifica se os bids estão em ordem decrescente de preço
        for i in range(1, len(orderbook.bids)):
            if orderbook.bids[i].price >= orderbook.bids[i-1].price:
                return False
        
        # Verifica se os asks estão em ordem crescente de preço
        for i in range(1, len(orderbook.asks)):
            if orderbook.asks[i].price <= orderbook.asks[i-1].price:
                return False
        
        return True
    
    def _validate_orderbook_values(self, orderbook: OrderBook) -> bool:
        """
        Valida se os valores no orderbook são válidos.
        
        Args:
            orderbook: Orderbook a ser validado
            
        Returns:
            bool: True se os valores são válidos, False caso contrário
        """
        # Verifica se há pelo menos um bid e um ask
        if not orderbook.bids or not orderbook.asks:
            return False
        
        # Verifica se o melhor bid é menor que o melhor ask
        if orderbook.bids[0].price >= orderbook.asks[0].price:
            return False
        
        # Verifica se todos os preços e quantidades são positivos
        for level in orderbook.bids + orderbook.asks:
            if level.price <= 0 or level.amount <= 0:
                return False
        
        return True
    
    def _validate_candle_continuity(self, candle: Candle, previous_candle: Candle) -> bool:
        """
        Valida a continuidade entre duas velas consecutivas.
        
        Args:
            candle: Vela atual
            previous_candle: Vela anterior
            
        Returns:
            bool: True se a continuidade é válida, False caso contrário
        """
        # Verifica se as velas são do mesmo par e timeframe
        if (candle.exchange != previous_candle.exchange or
            candle.trading_pair != previous_candle.trading_pair or
            candle.timeframe != previous_candle.timeframe):
            return False
        
        # Verifica a continuidade do timestamp
        expected_timestamp = self._get_next_candle_timestamp(
            previous_candle.timestamp, 
            previous_candle.timeframe
        )
        
        if candle.timestamp != expected_timestamp:
            return False
        
        # Verifica se o preço de abertura não é muito diferente do fechamento anterior
        price_diff = abs(candle.open - previous_candle.close)
        avg_price = (candle.open + previous_candle.close) / Decimal('2')
        
        if avg_price > 0 and price_diff / avg_price > Decimal(str(self.max_price_deviation_percent / 100)):
            return False
        
        return True
    
    def _get_next_candle_timestamp(self, timestamp: datetime, timeframe: TimeFrame) -> datetime:
        """
        Calcula o timestamp esperado para a próxima vela.
        
        Args:
            timestamp: Timestamp da vela atual
            timeframe: Timeframe das velas
            
        Returns:
            datetime: Timestamp esperado para a próxima vela
        """
        if timeframe == TimeFrame.MINUTE_1:
            return timestamp + timedelta(minutes=1)
        elif timeframe == TimeFrame.MINUTE_3:
            return timestamp + timedelta(minutes=3)
        elif timeframe == TimeFrame.MINUTE_5:
            return timestamp + timedelta(minutes=5)
        elif timeframe == TimeFrame.MINUTE_15:
            return timestamp + timedelta(minutes=15)
        elif timeframe == TimeFrame.MINUTE_30:
            return timestamp + timedelta(minutes=30)
        elif timeframe == TimeFrame.HOUR_1:
            return timestamp + timedelta(hours=1)
        elif timeframe == TimeFrame.HOUR_2:
            return timestamp + timedelta(hours=2)
        elif timeframe == TimeFrame.HOUR_4:
            return timestamp + timedelta(hours=4)
        elif timeframe == TimeFrame.HOUR_6:
            return timestamp + timedelta(hours=6)
        elif timeframe == TimeFrame.HOUR_8:
            return timestamp + timedelta(hours=8)
        elif timeframe == TimeFrame.HOUR_12:
            return timestamp + timedelta(hours=12)
        elif timeframe == TimeFrame.DAY_1:
            return timestamp + timedelta(days=1)
        elif timeframe == TimeFrame.DAY_3:
            return timestamp + timedelta(days=3)
        elif timeframe == TimeFrame.WEEK_1:
            return timestamp + timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTH_1:
            # Aproximação de um mês (30 dias)
            return timestamp + timedelta(days=30)
        else:
            raise ValueError(f"Timeframe não suportado: {timeframe}")
    
    def reset_references(self) -> None:
        """
        Reseta todas as referências utilizadas para validação.
        """
        self._price_reference.clear()
        self._volume_reference.clear()