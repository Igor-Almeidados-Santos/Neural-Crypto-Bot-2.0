"""
Neural Crypto Bot - Strategy DTOs

Este módulo contém os Data Transfer Objects (DTOs) relacionados às estratégias de trading.
Os DTOs são utilizados para validar e estruturar os dados entre a API e o domínio da aplicação.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class StrategyType(str, Enum):
    """Enumeração dos tipos de estratégia suportados."""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CUSTOM = "custom"


class TimeFrame(str, Enum):
    """Enumeração dos timeframes suportados para análise de dados."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


class OrderType(str, Enum):
    """Enumeração dos tipos de ordem suportados."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class PositionSide(str, Enum):
    """Enumeração dos lados de posição suportados."""
    LONG = "long"
    SHORT = "short"


class StrategyStatus(str, Enum):
    """Enumeração dos estados possíveis de uma estratégia."""
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    BACKTEST = "backtest"
    ERROR = "error"


class StrategyParameterBase(BaseModel):
    """Modelo base para parâmetros de estratégia."""
    name: str = Field(..., description="Nome do parâmetro")
    description: str = Field(..., description="Descrição do parâmetro")
    required: bool = Field(True, description="Indica se o parâmetro é obrigatório")
    default_value: Optional[Union[str, int, float, bool]] = Field(
        None, description="Valor padrão do parâmetro"
    )


class NumericParameter(StrategyParameterBase):
    """Parâmetro numérico para estratégias."""
    type: str = Field("numeric", const=True)
    min_value: Optional[float] = Field(None, description="Valor mínimo permitido")
    max_value: Optional[float] = Field(None, description="Valor máximo permitido")
    step: Optional[float] = Field(None, description="Incremento entre valores permitidos")


class CategoricalParameter(StrategyParameterBase):
    """Parâmetro categórico para estratégias."""
    type: str = Field("categorical", const=True)
    options: List[str] = Field(..., description="Opções válidas para o parâmetro")


class BooleanParameter(StrategyParameterBase):
    """Parâmetro booleano para estratégias."""
    type: str = Field("boolean", const=True)


class StrategyParameter(BaseModel):
    """Modelo de parâmetro de estratégia com valor."""
    name: str = Field(..., description="Nome do parâmetro")
    value: Union[str, int, float, bool] = Field(..., description="Valor do parâmetro")


class StrategyCreate(BaseModel):
    """DTO para criação de uma nova estratégia."""
    name: str = Field(..., description="Nome da estratégia")
    description: str = Field(..., description="Descrição da estratégia")
    type: StrategyType = Field(..., description="Tipo da estratégia")
    trading_pairs: List[str] = Field(..., description="Pares de trading para esta estratégia")
    timeframe: TimeFrame = Field(..., description="Timeframe para análise de dados")
    parameters: List[StrategyParameter] = Field(
        default_factory=list, description="Parâmetros específicos da estratégia"
    )
    
    # Configurações opcionais de risk management
    max_position_size: Optional[float] = Field(
        None, description="Tamanho máximo da posição como porcentagem do capital"
    )
    max_drawdown_percent: Optional[float] = Field(
        None, description="Drawdown máximo permitido em porcentagem"
    )
    take_profit_percent: Optional[float] = Field(
        None, description="Porcentagem de take profit"
    )
    stop_loss_percent: Optional[float] = Field(
        None, description="Porcentagem de stop loss"
    )
    risk_reward_ratio: Optional[float] = Field(
        None, description="Proporção de risco/recompensa desejada"
    )
    is_active: bool = Field(False, description="Indica se a estratégia está ativa")

    @validator('trading_pairs')
    def validate_trading_pairs(cls, value):
        """Valida o formato dos pares de trading."""
        for pair in value:
            if '/' not in pair:
                raise ValueError(f"Par de trading inválido: {pair}. Formato esperado: 'BTC/USDT'")
        return value


class StrategyUpdate(BaseModel):
    """DTO para atualização de uma estratégia existente."""
    name: Optional[str] = None
    description: Optional[str] = None
    trading_pairs: Optional[List[str]] = None
    timeframe: Optional[TimeFrame] = None
    parameters: Optional[List[StrategyParameter]] = None
    max_position_size: Optional[float] = None
    max_drawdown_percent: Optional[float] = None
    take_profit_percent: Optional[float] = None
    stop_loss_percent: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    is_active: Optional[bool] = None

    @validator('trading_pairs')
    def validate_trading_pairs(cls, value):
        """Valida o formato dos pares de trading."""
        if value is None:
            return value
        
        for pair in value:
            if '/' not in pair:
                raise ValueError(f"Par de trading inválido: {pair}. Formato esperado: 'BTC/USDT'")
        return value


class StrategyResponse(BaseModel):
    """DTO para resposta contendo uma estratégia."""
    id: UUID
    name: str
    description: str
    type: StrategyType
    trading_pairs: List[str]
    timeframe: TimeFrame
    parameters: List[StrategyParameter]
    max_position_size: Optional[float]
    max_drawdown_percent: Optional[float]
    take_profit_percent: Optional[float]
    stop_loss_percent: Optional[float]
    risk_reward_ratio: Optional[float]
    status: StrategyStatus
    created_at: datetime
    updated_at: datetime
    stats: Optional[Dict] = None


class StrategyListResponse(BaseModel):
    """DTO para resposta contendo uma lista de estratégias."""
    items: List[StrategyResponse]
    total: int
    page: int
    size: int


class BacktestRequest(BaseModel):
    """DTO para solicitação de backtesting."""
    strategy_id: UUID = Field(..., description="ID da estratégia para backtesting")
    start_date: datetime = Field(..., description="Data de início do backtesting")
    end_date: datetime = Field(..., description="Data de fim do backtesting")
    initial_capital: float = Field(..., description="Capital inicial para o backtesting")
    trading_fee: float = Field(0.001, description="Taxa de trading aplicada")
    slippage: float = Field(0.0005, description="Slippage estimado para simulação")
    use_historical_data: bool = Field(
        True, description="Usar dados históricos em vez de simulação"
    )
    include_detailed_trades: bool = Field(
        False, description="Incluir detalhes de cada trade no resultado"
    )
    
    @validator('end_date')
    def validate_dates(cls, end_date, values):
        """Valida que a data de fim é posterior à data de início."""
        if 'start_date' in values and end_date <= values['start_date']:
            raise ValueError("A data de fim deve ser posterior à data de início")
        return end_date


class BacktestResult(BaseModel):
    """DTO para resultado de backtesting."""
    strategy_id: UUID
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    detailed_trades: Optional[List[Dict]] = None
    equity_curve: List[Dict]
    monthly_returns: Dict[str, float]
    trade_distribution: Dict


class SignalRequest(BaseModel):
    """DTO para solicitação de sinais de uma estratégia."""
    strategy_id: UUID = Field(..., description="ID da estratégia para gerar sinais")
    trading_pairs: Optional[List[str]] = Field(
        None, description="Lista opcional de pares para gerar sinais (sobrepõe a configuração da estratégia)"
    )
    generate_orders: bool = Field(
        False, description="Se deve gerar ordens a partir dos sinais"
    )


class SignalResponse(BaseModel):
    """DTO para resposta contendo sinais de trading."""
    strategy_id: UUID
    timestamp: datetime
    trading_pair: str
    side: PositionSide
    price: float
    confidence: float
    indicators: Dict[str, float]
    suggested_entry: float
    suggested_take_profit: Optional[float] = None
    suggested_stop_loss: Optional[float] = None
    notes: Optional[str] = None


class OrderRequest(BaseModel):
    """DTO para solicitação de criação de ordem."""
    strategy_id: Optional[UUID] = Field(
        None, description="ID da estratégia associada à ordem (se aplicável)"
    )
    trading_pair: str = Field(..., description="Par de trading, ex: BTC/USDT")
    side: PositionSide = Field(..., description="Lado da ordem (long ou short)")
    type: OrderType = Field(..., description="Tipo da ordem")
    quantity: float = Field(..., description="Quantidade a ser negociada")
    price: Optional[float] = Field(
        None, description="Preço para ordens limit, stop, etc. (não necessário para market)"
    )
    take_profit: Optional[float] = Field(None, description="Nível de take profit")
    stop_loss: Optional[float] = Field(None, description="Nível de stop loss")
    trailing_percent: Optional[float] = Field(
        None, description="Percentual de trailing para trailing stops"
    )
    exchange: Optional[str] = Field(
        None, description="Exchange específica para esta ordem (se não for usar a padrão)"
    )
    time_in_force: Optional[str] = Field("GTC", description="Tempo de validade da ordem")
    execution_strategy: Optional[str] = Field(
        None, description="Estratégia de execução (TWAP, VWAP, Iceberg, etc.)"
    )


class OrderResponse(BaseModel):
    """DTO para resposta contendo informações de uma ordem."""
    id: UUID
    strategy_id: Optional[UUID]
    trading_pair: str
    side: PositionSide
    type: OrderType
    quantity: float
    price: Optional[float]
    status: str
    filled_quantity: float
    average_fill_price: Optional[float]
    exchange: str
    exchange_order_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    take_profit: Optional[float]
    stop_loss: Optional[float]
    trailing_percent: Optional[float]
    execution_strategy: Optional[str]
    execution_details: Optional[Dict]