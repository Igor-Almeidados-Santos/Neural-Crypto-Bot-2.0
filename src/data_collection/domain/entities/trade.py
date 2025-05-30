"""
Entidade Trade para representação de transações de mercado.

Esta classe define a estrutura e comportamento de uma transação (trade)
em mercados de criptomoedas, mantendo informações como par de negociação,
preço, quantidade, timestamp, etc.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any


class TradeSide(str, Enum):
    """Enumeração dos possíveis lados de uma transação."""
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Trade:
    """
    Representa uma transação (trade) em um mercado de criptomoedas.
    
    Attributes:
        id: Identificador único da transação
        exchange: Nome da exchange onde a transação ocorreu
        trading_pair: Par de negociação (ex: BTC/USDT)
        price: Preço da transação
        amount: Quantidade negociada
        cost: Custo total da transação (preço * quantidade)
        timestamp: Timestamp em que a transação ocorreu (UTC)
        side: Lado da transação (compra/venda)
        taker: Se o trade foi executado como taker (True) ou maker (False)
        raw_data: Dados brutos da transação conforme recebidos da exchange
    """
    id: str
    exchange: str
    trading_pair: str
    price: Decimal
    amount: Decimal
    cost: Decimal
    timestamp: datetime
    side: TradeSide
    taker: bool
    raw_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a entidade para um dicionário.
        
        Returns:
            Dict[str, Any]: Representação da entidade em formato de dicionário
        """
        return {
            "id": self.id,
            "exchange": self.exchange,
            "trading_pair": self.trading_pair,
            "price": float(self.price),
            "amount": float(self.amount),
            "cost": float(self.cost),
            "timestamp": self.timestamp.isoformat(),
            "side": self.side.value,
            "taker": self.taker
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """
        Cria uma instância da entidade a partir de um dicionário.
        
        Args:
            data: Dicionário contendo os dados da transação
            
        Returns:
            Trade: Instância da entidade Trade
        """
        return cls(
            id=data["id"],
            exchange=data["exchange"],
            trading_pair=data["trading_pair"],
            price=Decimal(str(data["price"])),
            amount=Decimal(str(data["amount"])),
            cost=Decimal(str(data["cost"])),
            timestamp=datetime.fromisoformat(data["timestamp"]) 
                    if isinstance(data["timestamp"], str) 
                    else data["timestamp"],
            side=TradeSide(data["side"]),
            taker=data["taker"],
            raw_data=data.get("raw_data")
        )