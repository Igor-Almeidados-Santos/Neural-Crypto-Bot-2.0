"""
Entidade Candle para representação de velas em gráficos de mercado.

Esta classe define a estrutura e comportamento de uma vela (candle)
em gráficos de mercados de criptomoedas, mantendo informações como
preços de abertura, fechamento, máxima, mínima, volume, etc.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any


class TimeFrame(str, Enum):
    """Enumeração dos possíveis timeframes para candles."""
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass(frozen=True)
class Candle:
    """
    Representa uma vela (candle) em um gráfico de mercado de criptomoedas.
    
    Attributes:
        exchange: Nome da exchange de onde os dados foram obtidos
        trading_pair: Par de negociação (ex: BTC/USDT)
        timestamp: Timestamp de abertura da vela (UTC)
        timeframe: Intervalo de tempo da vela
        open: Preço de abertura
        high: Preço máximo
        low: Preço mínimo
        close: Preço de fechamento
        volume: Volume negociado no período
        trades: Número de transações no período (opcional)
        raw_data: Dados brutos da vela conforme recebidos da exchange
    """
    exchange: str
    trading_pair: str
    timestamp: datetime
    timeframe: TimeFrame
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    trades: Optional[int] = None
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a entidade para um dicionário.
        
        Returns:
            Dict[str, Any]: Representação da entidade em formato de dicionário
        """
        result = {
            "exchange": self.exchange,
            "trading_pair": self.trading_pair,
            "timestamp": self.timestamp.isoformat(),
            "timeframe": self.timeframe.value,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume)
        }
        
        if self.trades is not None:
            result["trades"] = self.trades
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candle':
        """
        Cria uma instância da entidade a partir de um dicionário.
        
        Args:
            data: Dicionário contendo os dados da vela
            
        Returns:
            Candle: Instância da entidade Candle
        """
        return cls(
            exchange=data["exchange"],
            trading_pair=data["trading_pair"],
            timestamp=datetime.fromisoformat(data["timestamp"]) 
                     if isinstance(data["timestamp"], str) 
                     else data["timestamp"],
            timeframe=TimeFrame(data["timeframe"]),
            open=Decimal(str(data["open"])),
            high=Decimal(str(data["high"])),
            low=Decimal(str(data["low"])),
            close=Decimal(str(data["close"])),
            volume=Decimal(str(data["volume"])),
            trades=data.get("trades"),
            raw_data=data.get("raw_data")
        )
    
    @property
    def is_bullish(self) -> bool:
        """
        Verifica se a vela é de alta (fechamento maior que abertura).
        
        Returns:
            bool: True se a vela for de alta, False caso contrário
        """
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """
        Verifica se a vela é de baixa (fechamento menor que abertura).
        
        Returns:
            bool: True se a vela for de baixa, False caso contrário
        """
        return self.close < self.open
    
    @property
    def body_size(self) -> Decimal:
        """
        Calcula o tamanho do corpo da vela (diferença absoluta entre abertura e fechamento).
        
        Returns:
            Decimal: Tamanho do corpo da vela
        """
        return abs(self.close - self.open)
    
    @property
    def range(self) -> Decimal:
        """
        Calcula a amplitude da vela (diferença entre máxima e mínima).
        
        Returns:
            Decimal: Amplitude da vela
        """
        return self.high - self.low