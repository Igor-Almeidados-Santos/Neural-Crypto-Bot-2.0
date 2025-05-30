"""
Entidade FundingRate para representação de taxas de financiamento.

Esta classe define a estrutura e comportamento de uma taxa de financiamento
em contratos perpétuos de criptomoedas, mantendo informações como
valor da taxa, timestamp, próximo timestamp de cobrança, etc.
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class FundingRate:
    """
    Representa uma taxa de financiamento em um contrato perpétuo.
    
    Attributes:
        exchange: Nome da exchange onde a taxa foi capturada
        trading_pair: Par de negociação (ex: BTC/USDT)
        timestamp: Timestamp em que a taxa foi estabelecida (UTC)
        rate: Valor da taxa de financiamento
        next_timestamp: Timestamp da próxima cobrança de financiamento (opcional)
        raw_data: Dados brutos da taxa conforme recebidos da exchange
    """
    exchange: str
    trading_pair: str
    timestamp: datetime
    rate: Decimal
    next_timestamp: Optional[datetime] = None
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
            "rate": float(self.rate)
        }
        
        if self.next_timestamp:
            result["next_timestamp"] = self.next_timestamp.isoformat()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FundingRate':
        """
        Cria uma instância da entidade a partir de um dicionário.
        
        Args:
            data: Dicionário contendo os dados da taxa de financiamento
            
        Returns:
            FundingRate: Instância da entidade FundingRate
        """
        # Converte o timestamp para datetime
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        
        # Converte o próximo timestamp para datetime, se fornecido
        next_timestamp = None
        if "next_timestamp" in data:
            next_timestamp = datetime.fromisoformat(data["next_timestamp"]) if isinstance(data["next_timestamp"], str) else data["next_timestamp"]
        
        return cls(
            exchange=data["exchange"],
            trading_pair=data["trading_pair"],
            timestamp=timestamp,
            rate=Decimal(str(data["rate"])),
            next_timestamp=next_timestamp,
            raw_data=data.get("raw_data")
        )
    
    @property
    def is_positive(self) -> bool:
        """
        Verifica se a taxa de financiamento é positiva.
        
        Returns:
            bool: True se a taxa for positiva, False caso contrário
        """
        return self.rate > Decimal('0')
    
    @property
    def is_negative(self) -> bool:
        """
        Verifica se a taxa de financiamento é negativa.
        
        Returns:
            bool: True se a taxa for negativa, False caso contrário
        """
        return self.rate < Decimal('0')
    
    @property
    def is_neutral(self) -> bool:
        """
        Verifica se a taxa de financiamento é neutra (igual a zero).
        
        Returns:
            bool: True se a taxa for igual a zero, False caso contrário
        """
        return self.rate == Decimal('0')
    
    @property
    def annual_percentage_rate(self) -> Decimal:
        """
        Calcula a taxa anual equivalente, considerando que a cobrança ocorre 3 vezes por dia.
        
        Returns:
            Decimal: Taxa anual equivalente
        """
        # A maioria das exchanges cobra a taxa a cada 8 horas, ou 3 vezes por dia
        return self.rate * Decimal('3') * Decimal('365')