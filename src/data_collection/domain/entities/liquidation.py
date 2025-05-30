"""
Entidade Liquidation para representação de eventos de liquidação.

Esta classe define a estrutura e comportamento de um evento de liquidação
em mercados de criptomoedas, mantendo informações como preço, quantidade,
lado da liquidação, etc.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any


class LiquidationSide(str, Enum):
    """Enumeração dos possíveis lados de uma liquidação."""
    BUY = "buy"      # Posição curta liquidada, resultando em compra forçada
    SELL = "sell"    # Posição longa liquidada, resultando em venda forçada
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Liquidation:
    """
    Representa um evento de liquidação em um mercado de criptomoedas.
    
    Attributes:
        exchange: Nome da exchange onde a liquidação ocorreu
        trading_pair: Par de negociação (ex: BTC/USDT)
        timestamp: Timestamp em que a liquidação ocorreu (UTC)
        price: Preço da liquidação
        amount: Quantidade liquidada
        side: Lado da liquidação (buy/sell)
        raw_data: Dados brutos da liquidação conforme recebidos da exchange
    """
    exchange: str
    trading_pair: str
    timestamp: datetime
    price: Decimal
    amount: Decimal
    side: LiquidationSide
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a entidade para um dicionário.
        
        Returns:
            Dict[str, Any]: Representação da entidade em formato de dicionário
        """
        return {
            "exchange": self.exchange,
            "trading_pair": self.trading_pair,
            "timestamp": self.timestamp.isoformat(),
            "price": float(self.price),
            "amount": float(self.amount),
            "side": self.side.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Liquidation':
        """
        Cria uma instância da entidade a partir de um dicionário.
        
        Args:
            data: Dicionário contendo os dados da liquidação
            
        Returns:
            Liquidation: Instância da entidade Liquidation
        """
        return cls(
            exchange=data["exchange"],
            trading_pair=data["trading_pair"],
            timestamp=datetime.fromisoformat(data["timestamp"]) 
                     if isinstance(data["timestamp"], str) 
                     else data["timestamp"],
            price=Decimal(str(data["price"])),
            amount=Decimal(str(data["amount"])),
            side=LiquidationSide(data["side"]),
            raw_data=data.get("raw_data")
        )
    
    @property
    def value(self) -> Decimal:
        """
        Calcula o valor total da liquidação (preço * quantidade).
        
        Returns:
            Decimal: Valor total da liquidação
        """
        return self.price * self.amount
    
    @property
    def is_long_liquidation(self) -> bool:
        """
        Verifica se a liquidação é de uma posição longa.
        
        Returns:
            bool: True se for uma liquidação de posição longa, False caso contrário
        """
        return self.side == LiquidationSide.SELL
    
    @property
    def is_short_liquidation(self) -> bool:
        """
        Verifica se a liquidação é de uma posição curta.
        
        Returns:
            bool: True se for uma liquidação de posição curta, False caso contrário
        """
        return self.side == LiquidationSide.BUY