import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.risk_management.domain.entities.risk_metric import MetricType, RiskMetric, TimeFrame
from src.risk_management.domain.entities.risk_profile import RiskProfile
from src.risk_management.models.var_model import VaRModel
from src.risk_management.models.expected_shortfall import ExpectedShortfallModel
from src.risk_management.models.drawdown_control import DrawdownControl


@dataclass
class PortfolioPosition:
    """Representa uma posição em um ativo no portfólio."""
    asset_id: str
    quantity: float
    current_price: float
    avg_entry_price: float
    unrealized_pnl: float
    position_value: float
    weight: float  # Percentual do portfólio
    metadata: Dict[str, Any] = None


@dataclass
class PortfolioStats:
    """Estatísticas gerais de um portfólio."""
    total_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    num_positions: int
    max_position_value: float
    max_position_weight: float
    max_position_asset: str
    concentration_index: float  # Índice de Herfindahl-Hirschman (HHI)
    correlation_avg: float
    volatility_daily: float
    sharpe_ratio: float
    sortino_ratio: float
    var_95: float
    var_99: float
    expected_shortfall_95: float
    max_drawdown: float
    drawdown_current: float
    drawdown_period: int  # Dias
    timestamp: datetime


class PortfolioAnalyticsService:
    """
    Serviço responsável por calcular métricas avançadas de risco e análise de portfólio.
    """
    
    def __init__(
        self,
        var_model: VaRModel,
        es_model: ExpectedShortfallModel,
        drawdown_control: DrawdownControl,
        risk_free_rate: float = 0.0,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o serviço de análise de portfólio.
        
        Args:
            var_model: Modelo para cálculo de Value at Risk
            es_model: Modelo para cálculo de Expected Shortfall
            drawdown_control: Controlador de drawdown
            risk_free_rate: Taxa livre de risco anualizada para cálculo de Sharpe/Sortino
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._var_model = var_model
        self._es_model = es_model
        self._drawdown_control = drawdown_control
        self._risk_free_rate = risk_free_rate
        self._price_history: Dict[str, pd.DataFrame] = {}
        self._returns_history: Dict[str, pd.DataFrame] = {}
        self._portfolio_history: List[Dict[str, Any]] = []
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_update = datetime.utcnow()
    
    def update_portfolio_data(
        self,
        positions: List[PortfolioPosition],
        prices: Dict[str, float],
        returns: Dict[str, List[float]],
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Atualiza os dados do portfólio e preços/retornos históricos.
        
        Args:
            positions: Lista de posições atuais do portfólio
            prices: Dicionário de preços atuais por ativo
            returns: Dicionário de retornos históricos por ativo
            timestamp: Data/hora da atualização (usa timestamp atual se não fornecido)
        """
        now = timestamp or datetime.utcnow()
        
        # Armazenar snapshot do portfólio
        portfolio_snapshot = {
            "timestamp": now,
            "total_value": sum(p.position_value for p in positions),
            "positions": {p.asset_id: p.position_value for p in positions},
            "weights": {p.asset_id: p.weight for p in positions},
            "prices": prices.copy(),
        }
        self._portfolio_history.append(portfolio_snapshot)
        
        # Limitar histórico a 1 ano de dados (365 dias)
        max_history = 365
        if len(self._portfolio_history) > max_history:
            self._portfolio_history = self._portfolio_history[-max_history:]
        
        # Atualizar histórico de preços
        for asset_id, price in prices.items():
            if asset_id not in self._price_history:
                self._price_history[asset_id] = pd.DataFrame(columns=["timestamp", "price"])
            
            new_row = pd.DataFrame([{"timestamp": now, "price": price}])
            self._price_history[asset_id] = pd.concat([self._price_history[asset_id], new_row])
            
            # Limitar histórico de preços
            self._price_history[asset_id] = self._price_history[asset_id].tail(max_history)
        
        # Atualizar histórico de retornos
        for asset_id, asset_returns in returns.items():
            if asset_id not in self._returns_history:
                self._returns_history[asset_id] = pd.DataFrame(columns=["timestamp", "return"])
            
            for i, ret in enumerate(asset_returns):
                # Assumindo que os retornos estão em ordem cronológica, com o mais recente por último
                ret_timestamp = now - timedelta(days=len(asset_returns) - i - 1)
                new_row = pd.DataFrame([{"timestamp": ret_timestamp, "return": ret}])
                self._returns_history[asset_id] = pd.concat([self._returns_history[asset_id], new_row])
            
            # Limitar histórico de retornos
            self._returns_history[asset_id] = self._returns_history[asset_id].tail(max_history)
        
        # Atualizar matriz de correlação se tiver pelo menos 2 ativos com dados suficientes
        assets_with_data = [asset_id for asset_id in returns if len(returns[asset_id]) >= 30]
        if len(assets_with_data) >= 2:
            correlation_data = {}
            for asset_id in assets_with_data:
                correlation_data[asset_id] = returns[asset_id][-30:]  # Usar últimos 30 dias
            
            # Criar DataFrame com retornos
            df = pd.DataFrame(correlation_data)
            # Calcular matriz de correlação
            self._correlation_matrix = df.corr()
        
        self._last_update = now
        self._logger.debug(f"Dados de portfólio atualizados: {len(positions)} posições, {len(prices)} preços")
    
    def calculate_portfolio_stats(self, portfolio_id: str) -> PortfolioStats:
        """
        Calcula estatísticas completas do portfólio.
        
        Args:
            portfolio_id: Identificador do portfólio
            
        Returns:
            Estatísticas detalhadas do portfólio
        """
        if not self._portfolio_history:
            raise ValueError("Nenhum dado de portfólio disponível. Execute update_portfolio_data primeiro.")
        
        current = self._portfolio_history[-1]
        positions = current["positions"]
        weights = current["weights"]
        total_value = current["total_value"]
        
        # Encontrar a posição com maior valor
        max_position_asset = max(positions.items(), key=lambda x: x[1])[0] if positions else ""
        max_position_value = positions.get(max_position_asset, 0)
        max_position_weight = weights.get(max_position_asset, 0)
        
        # Calcular índice de concentração (HHI)
        concentration_index = sum(w**2 for w in weights.values()) if weights else 0
        
        # Calcular correlação média (excluindo diagonal)
        correlation_avg = 0.0
        if self._correlation_matrix is not None and len(self._correlation_matrix) > 1:
            # Extrair valores sem a diagonal (correlações entre ativos diferentes)
            corr_values = []
            for i in range(len(self._correlation_matrix)):
                for j in range(len(self._correlation_matrix)):
                    if i != j:
                        corr_values.append(self._correlation_matrix.iloc[i, j])
            correlation_avg = np.mean(corr_values)
        
        # Calcular retornos históricos do portfólio (se houver histórico suficiente)
        portfolio_returns = []
        if len(self._portfolio_history) >= 2:
            for i in range(1, len(self._portfolio_history)):
                prev = self._portfolio_history[i-1]
                curr = self._portfolio_history[i]
                
                if prev["total_value"] > 0:
                    portfolio_return = (curr["total_value"] - prev["total_value"]) / prev["total_value"]
                    portfolio_returns.append(portfolio_return)
        
        # Calcular métricas de risco com base nos retornos do portfólio
        volatility_daily = np.std(portfolio_returns) * 100 if portfolio_returns else 0
        
        # Calcular Sharpe e Sortino Ratio (anualizado)
        avg_return = np.mean(portfolio_returns) if portfolio_returns else 0
        daily_risk_free = (1 + self._risk_free_rate) ** (1/252) - 1  # Taxa diária equivalente
        excess_returns = [r - daily_risk_free for r in portfolio_returns]
        
        sharpe_ratio = 0
        if volatility_daily > 0 and portfolio_returns:
            sharpe_ratio = (np.mean(excess_returns) / (volatility_daily / 100)) * np.sqrt(252)
        
        # Sortino Ratio (considera apenas retornos negativos na volatilidade)
        downside_returns = [r for r in excess_returns if r < 0]
        downside_volatility = np.std(downside_returns) * 100 if downside_returns else 0
        
        sortino_ratio = 0
        if downside_volatility > 0 and portfolio_returns:
            sortino_ratio = (np.mean(excess_returns) / (downside_volatility / 100)) * np.sqrt(252)
        
        # Calcular VaR e Expected Shortfall
        if portfolio_returns:
            weights_dict = {asset: weights.get(asset, 0) for asset in weights}
            var_95 = self._var_model.calculate(portfolio_returns, confidence_level=0.95)
            var_99 = self._var_model.calculate(portfolio_returns, confidence_level=0.99)
            expected_shortfall_95 = self._es_model.calculate(portfolio_returns, confidence_level=0.95)
        else:
            var_95 = 0
            var_99 = 0
            expected_shortfall_95 = 0
        
        # Calcular drawdown
        equity_curve = [hist["total_value"] for hist in self._portfolio_history]
        max_drawdown, current_drawdown, drawdown_days = self._drawdown_control.calculate_drawdown(equity_curve)
        
        # Calcular PnL não realizado
        # Assumindo que temos valores de entrada médios em algum lugar
        # Como não temos, vamos apenas usar um valor fictício baseado nos primeiros dados
        if len(self._portfolio_history) > 1:
            first_snapshot = self._portfolio_history[0]
            unrealized_pnl = total_value - sum(first_snapshot["positions"].values())
            unrealized_pnl_pct = (unrealized_pnl / sum(first_snapshot["positions"].values())) * 100 if sum(first_snapshot["positions"].values()) > 0 else 0
        else:
            unrealized_pnl = 0
            unrealized_pnl_pct = 0
        
        return PortfolioStats(
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            num_positions=len(positions),
            max_position_value=max_position_value,
            max_position_weight=max_position_weight,
            max_position_asset=max_position_asset,
            concentration_index=concentration_index,
            correlation_avg=correlation_avg,
            volatility_daily=volatility_daily,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=expected_shortfall_95,
            max_drawdown=max_drawdown,
            drawdown_current=current_drawdown,
            drawdown_period=drawdown_days,
            timestamp=self._last_update
        )
    
    def generate_risk_metrics(self, portfolio_id: str, profile: RiskProfile) -> List[RiskMetric]:
        """
        Gera métricas de risco completas para o portfólio.
        
        Args:
            portfolio_id: Identificador do portfólio
            profile: Perfil de risco para referência de limites
            
        Returns:
            Lista de métricas de risco calculadas
        """
        stats = self.calculate_portfolio_stats(portfolio_id)
        metrics = []
        
        # Value at Risk
        metrics.append(RiskMetric.create(
            metric_type=MetricType.VALUE_AT_RISK,
            value=stats.var_95,
            portfolio_id=portfolio_id,
            confidence_level=0.95,
            metadata={
                "total_value": str(stats.total_value),
                "var_absolute": str(stats.var_95 * stats.total_value / 100),
            }
        ))
        
        metrics.append(RiskMetric.create(
            metric_type=MetricType.VALUE_AT_RISK,
            value=stats.var_99,
            portfolio_id=portfolio_id,
            confidence_level=0.99,
            metadata={
                "total_value": str(stats.total_value),
                "var_absolute": str(stats.var_99 * stats.total_value / 100),
            }
        ))
        
        # Expected Shortfall
        metrics.append(RiskMetric.create(
            metric_type=MetricType.EXPECTED_SHORTFALL,
            value=stats.expected_shortfall_95,
            portfolio_id=portfolio_id,
            confidence_level=0.95,
            metadata={
                "total_value": str(stats.total_value),
                "es_absolute": str(stats.expected_shortfall_95 * stats.total_value / 100),
            }
        ))
        
        # Drawdown
        metrics.append(RiskMetric.create(
            metric_type=MetricType.DRAWDOWN,
            value=stats.drawdown_current,
            portfolio_id=portfolio_id,
            metadata={
                "max_drawdown": str(stats.max_drawdown),
                "drawdown_days": str(stats.drawdown_period),
                "drawdown_absolute": str(stats.drawdown_current * stats.total_value / 100),
            }
        ))
        
        # Volatilidade
        metrics.append(RiskMetric.create(
            metric_type=MetricType.VOLATILITY,
            value=stats.volatility_daily,
            portfolio_id=portfolio_id,
            timeframe=TimeFrame.DAY_1,
            metadata={
                "annualized": str(stats.volatility_daily * np.sqrt(252)),
            }
        ))
        
        # Sharpe Ratio
        metrics.append(RiskMetric.create(
            metric_type=MetricType.SHARPE_RATIO,
            value=stats.sharpe_ratio,
            portfolio_id=portfolio_id,
            metadata={
                "sortino_ratio": str(stats.sortino_ratio),
                "risk_free_rate": str(self._risk_free_rate),
            }
        ))
        
        # Concentração
        metrics.append(RiskMetric.create(
            metric_type=MetricType.CONCENTRATION,
            value=stats.concentration_index,
            portfolio_id=portfolio_id,
            metadata={
                "max_position_asset": stats.max_position_asset,
                "max_position_weight": str(stats.max_position_weight),
            }
        ))
        
        # Correlação
        metrics.append(RiskMetric.create(
            metric_type=MetricType.CORRELATION,
            value=stats.correlation_avg,
            portfolio_id=portfolio_id,
            metadata={
                "num_assets": str(len(stats.num_positions)),
            }
        ))
        
        return metrics
    
    def get_correlation_matrix(self) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Retorna a matriz de correlação atual entre ativos.
        
        Returns:
            Matriz de correlação como dicionário ou None se não disponível
        """
        if self._correlation_matrix is None:
            return None
        
        result = {}
        for asset1 in self._correlation_matrix.index:
            result[asset1] = {}
            for asset2 in self._correlation_matrix.columns:
                result[asset1][asset2] = self._correlation_matrix.loc[asset1, asset2]
        
        return result
    
    def calculate_asset_risk_contribution(self, portfolio_id: str) -> Dict[str, Dict[str, float]]:
        """
        Calcula a contribuição de cada ativo para o risco total do portfólio.
        
        Args:
            portfolio_id: Identificador do portfólio
            
        Returns:
            Dicionário com contribuição de risco por ativo
        """
        if not self._portfolio_history or not self._correlation_matrix:
            raise ValueError("Dados insuficientes para calcular contribuição de risco")
        
        current = self._portfolio_history[-1]
        weights = current["weights"]
        
        # Filtrar para incluir apenas ativos que estão na matriz de correlação
        filtered_weights = {k: v for k, v in weights.items() if k in self._correlation_matrix.index}
        
        # Criar vetor de pesos na mesma ordem dos índices da matriz
        weight_vector = np.array([filtered_weights.get(asset, 0) for asset in self._correlation_matrix.index])
        
        # Extrair matriz de covariância dos retornos (usando correlação * volatilidades)
        volatilities = {}
        for asset in self._correlation_matrix.index:
            if asset in self._returns_history and not self._returns_history[asset].empty:
                returns = self._returns_history[asset]["return"].values
                volatilities[asset] = np.std(returns)
            else:
                volatilities[asset] = 0
        
        # Construir matriz de covariância a partir da correlação e volatilidades
        n = len(self._correlation_matrix.index)
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            asset_i = self._correlation_matrix.index[i]
            for j in range(n):
                asset_j = self._correlation_matrix.index[j]
                cov_matrix[i, j] = (
                    self._correlation_matrix.iloc[i, j] * 
                    volatilities.get(asset_i, 0) * 
                    volatilities.get(asset_j, 0)
                )
        
        # Calcular variância do portfólio
        portfolio_variance = weight_vector.T @ cov_matrix @ weight_vector
        
        # Calcular contribuição marginal de risco
        marginal_contribution = cov_matrix @ weight_vector
        
        # Calcular contribuição de risco total
        risk_contribution = weight_vector * marginal_contribution
        
        # Normalizar para percentual
        if portfolio_variance > 0:
            percent_contribution = risk_contribution / portfolio_variance
        else:
            percent_contribution = np.zeros_like(risk_contribution)
        
        # Construir resultado
        result = {}
        for i, asset in enumerate(self._correlation_matrix.index):
            if asset in filtered_weights:
                result[asset] = {
                    "weight": filtered_weights[asset],
                    "volatility": volatilities.get(asset, 0),
                    "marginal_contribution": marginal_contribution[i],
                    "risk_contribution": risk_contribution[i],
                    "percent_contribution": percent_contribution[i] * 100,
                }
        
        return result