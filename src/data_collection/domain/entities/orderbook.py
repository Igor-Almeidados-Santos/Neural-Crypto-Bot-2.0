"""
Entidade OrderBook para representação de livros de ofertas.

Esta classe define a estrutura e comportamento de um livro de ofertas (orderbook)
em mercados de criptomoedas, mantendo informações como ofertas de compra e venda,
timestamp, etc.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Tuple, Dict, Any, Optional


@dataclass(frozen=True)
class OrderBookLevel:
    """
    Representa um nível de preço no livro de ofertas.
    
    Attributes:
        price: Preço do nível
        amount: Quantidade disponível neste preço
        count: Número de ordens neste nível (opcional)
    """
    price: Decimal
    amount: Decimal
    count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a entidade para um dicionário.
        
        Returns:
            Dict[str, Any]: Representação da entidade em formato de dicionário
        """
        result = {
            "price": float(self.price),
            "amount": float(self.amount)
        }
        
        if self.count is not None:
            result["count"] = self.count
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBookLevel':
        """
        Cria uma instância da entidade a partir de um dicionário.
        
        Args:
            data: Dicionário contendo os dados do nível
            
        Returns:
            OrderBookLevel: Instância da entidade OrderBookLevel
        """
        return cls(
            price=Decimal(str(data["price"])),
            amount=Decimal(str(data["amount"])),
            count=data.get("count")
        )
    
    @classmethod
    def from_list(cls, data: List) -> 'OrderBookLevel':
        """
        Cria uma instância da entidade a partir de uma lista [price, amount, count?].
        
        Args:
            data: Lista contendo os dados do nível
            
        Returns:
            OrderBookLevel: Instância da entidade OrderBookLevel
        """
        if len(data) == 2:
            return cls(
                price=Decimal(str(data[0])),
                amount=Decimal(str(data[1]))
            )
        elif len(data) >= 3:
            return cls(
                price=Decimal(str(data[0])),
                amount=Decimal(str(data[1])),
                count=data[2]
            )
        else:
            raise ValueError("Lista deve conter pelo menos preço e quantidade")


@dataclass(frozen=True)
class OrderBook:
    """
    Representa um livro de ofertas (orderbook) em um mercado de criptomoedas.
    
    Attributes:
        exchange: Nome da exchange onde o orderbook foi capturado
        trading_pair: Par de negociação (ex: BTC/USDT)
        timestamp: Timestamp em que o orderbook foi capturado (UTC)
        bids: Lista de ofertas de compra ordenadas por preço descendente
        asks: Lista de ofertas de venda ordenadas por preço ascendente
        raw_data: Dados brutos do orderbook conforme recebidos da exchange
    """
    exchange: str
    trading_pair: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
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
            "bids": [bid.to_dict() for bid in self.bids],
            "asks": [ask.to_dict() for ask in self.asks]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderBook':
        """
        Cria uma instância da entidade a partir de um dicionário.
        
        Args:
            data: Dicionário contendo os dados do orderbook
            
        Returns:
            OrderBook: Instância da entidade OrderBook
        """
        return cls(
            exchange=data["exchange"],
            trading_pair=data["trading_pair"],
            timestamp=datetime.fromisoformat(data["timestamp"]) 
                     if isinstance(data["timestamp"], str) 
                     else data["timestamp"],
            bids=[OrderBookLevel.from_dict(bid) if isinstance(bid, dict) else OrderBookLevel.from_list(bid) 
                  for bid in data["bids"]],
            asks=[OrderBookLevel.from_dict(ask) if isinstance(ask, dict) else OrderBookLevel.from_list(ask) 
                  for ask in data["asks"]],
            raw_data=data.get("raw_data")
        )
    
    @property
    def mid_price(self) -> Decimal:
        """
        Calcula o preço médio entre a melhor oferta de compra e a melhor oferta de venda.
        
        Returns:
            Decimal: Preço médio
        """
        if not self.bids or not self.asks:
            raise ValueError("Orderbook vazio: bids ou asks não contém dados")
            
        return (self.bids[0].price + self.asks[0].price) / Decimal('2')
    
    @property
    def spread(self) -> Decimal:
        """
        Calcula o spread entre a melhor oferta de compra e a melhor oferta de venda.
        
        Returns:
            Decimal: Valor absoluto do spread
        """
        if not self.bids or not self.asks:
            raise ValueError("Orderbook vazio: bids ou asks não contém dados")
            
        return self.asks[0].price - self.bids[0].price
    
    @property
    def spread_percentage(self) -> Decimal:
        """
        Calcula o spread percentual em relação ao preço médio.
        
        Returns:
            Decimal: Percentual do spread
        """
        mid = self.mid_price
        return (self.spread / mid) * Decimal('100')
    
    def get_volume_at_price(self, price: Decimal) -> Tuple[Decimal, Decimal]:
        """
        Retorna o volume disponível nos lados de compra e venda para um determinado preço.
        
        Args:
            price: Preço a ser consultado
            
        Returns:
            Tuple[Decimal, Decimal]: (volume_bids, volume_asks) para o preço especificado
        """
        bid_volume = sum(bid.amount for bid in self.bids if bid.price == price)
        ask_volume = sum(ask.amount for ask in self.asks if ask.price == price)
        
        return (bid_volume, ask_volume)
    
    def get_cumulative_volume(self, side: str, price_limit: Optional[Decimal] = None) -> Decimal:
        """
        Calcula o volume cumulativo até um determinado preço limite.
        
        Args:
            side: Lado do orderbook ('bids' ou 'asks')
            price_limit: Preço limite para o cálculo (opcional)
            
        Returns:
            Decimal: Volume cumulativo
        """
        if side.lower() not in ['bids', 'asks']:
            raise ValueError("Side deve ser 'bids' ou 'asks'")
            
        levels = self.bids if side.lower() == 'bids' else self.asks
        
        if price_limit is None:
            return sum(level.amount for level in levels)
            
        if side.lower() == 'bids':
            # Para bids, considerar preços maiores ou iguais ao limite
            return sum(level.amount for level in levels if level.price >= price_limit)
        else:
            # Para asks, considerar preços menores ou iguais ao limite
            return sum(level.amount for level in levels if level.price <= price_limit)