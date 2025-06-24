"""
Serviço para atribuição de performance de estratégias de trading.

Este serviço calcula métricas avançadas de desempenho e realiza
análise de atribuição para identificar fontes de retorno.
"""
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal

from common.application.base_service import BaseService
from analytics.domain.value_objects.performance_metric import (
    PerformanceMetric, PerformanceMetricCollection, PerformanceAnalysis, MetricType
)


class PerformanceAttributionService(BaseService):
    """Serviço para atribuição de performance e métricas avançadas."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Inicializa o serviço de atribuição de performance."""
        super().__init__(logger)
    
    async def calculate_performance_metrics(
        self,
        returns: List[float],
        timestamps: List[datetime],
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.0,
        strategy_id: Optional[str] = None,
        strategy_name: Optional[str] = None,
        initial_capital: float = 10000.0,
        trades_data: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerformanceAnalysis:
        """
        Calcula métricas completas de desempenho para uma série de retornos.
        
        Args:
            returns: Lista de retornos percentuais (ex: 0.01 para 1%)
            timestamps: Lista de timestamps correspondentes aos retornos
            benchmark_returns: Lista de retornos do benchmark (opcional)
            risk_free_rate: Taxa livre de risco anualizada (ex: 0.02 para 2%)
            strategy_id: ID da estratégia (opcional)
            strategy_name: Nome da estratégia (opcional)
            initial_capital: Capital inicial
            trades_data: Dados de trades individuais (opcional)
            metadata: Metadados adicionais (opcional)
            
        Returns:
            Objeto PerformanceAnalysis com métricas calculadas
        """
        self.logger.info(f"Calculando métricas de performance para estratégia {strategy_id}")
        
        # Validar entradas
        if not returns or len(returns) < 2:
            raise ValueError("São necessários pelo menos dois pontos de retorno para calcular métricas")
        
        if len(returns) != len(timestamps):
            raise ValueError("O número de retornos deve ser igual ao número de timestamps")
        
        if benchmark_returns and len(benchmark_returns) != len(returns):
            raise ValueError("O número de retornos do benchmark deve ser igual ao número de retornos da estratégia")
        
        # Converter para arrays numpy para cálculos
        returns_array = np.array(returns)
        timestamps_array = np.array(timestamps)
        
        # Determinar período da análise
        start_date = min(timestamps)
        end_date = max(timestamps)
        time_range = self._determine_time_range(start_date, end_date)
        
        # Calcular retorno total e capital final
        cumulative_return = self._calculate_cumulative_return(returns_array)
        final_capital = initial_capital * (1 + cumulative_return)
        
        # Processar dados de trades se disponíveis
        trades_count = 0
        winning_trades_count = 0
        losing_trades_count = 0
        
        if trades_data:
            trades_count = len(trades_data)
            winning_trades_count = sum(1 for trade in trades_data if trade.get('profit_loss', 0) > 0)
            losing_trades_count = trades_count - winning_trades_count
        
        # Calcular todas as métricas
        metrics_dict = {}
        
        # 1. Retorno total
        total_return = Decimal(str(cumulative_return))
        metrics_dict[MetricType.TOTAL_RETURN] = PerformanceMetric(
            metric_type=MetricType.TOTAL_RETURN,
            value=total_return,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 2. Retorno anualizado
        annualized_return = Decimal(str(self._calculate_annualized_return(
            returns_array, start_date, end_date
        )))
        metrics_dict[MetricType.ANNUALIZED_RETURN] = PerformanceMetric(
            metric_type=MetricType.ANNUALIZED_RETURN,
            value=annualized_return,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 3. Drawdown máximo
        max_drawdown = Decimal(str(self._calculate_max_drawdown(returns_array)))
        metrics_dict[MetricType.MAX_DRAWDOWN] = PerformanceMetric(
            metric_type=MetricType.MAX_DRAWDOWN,
            value=max_drawdown,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 4. Volatilidade
        volatility = Decimal(str(self._calculate_volatility(returns_array, start_date, end_date)))
        metrics_dict[MetricType.VOLATILITY] = PerformanceMetric(
            metric_type=MetricType.VOLATILITY,
            value=volatility,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 5. Sharpe Ratio
        sharpe = Decimal(str(self._calculate_sharpe_ratio(
            returns_array, risk_free_rate, start_date, end_date
        )))
        metrics_dict[MetricType.SHARPE_RATIO] = PerformanceMetric(
            metric_type=MetricType.SHARPE_RATIO,
            value=sharpe,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 6. Sortino Ratio
        sortino = Decimal(str(self._calculate_sortino_ratio(
            returns_array, risk_free_rate, start_date, end_date
        )))
        metrics_dict[MetricType.SORTINO_RATIO] = PerformanceMetric(
            metric_type=MetricType.SORTINO_RATIO,
            value=sortino,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 7. Calmar Ratio
        calmar = Decimal(str(self._calculate_calmar_ratio(
            returns_array, start_date, end_date
        )))
        metrics_dict[MetricType.CALMAR_RATIO] = PerformanceMetric(
            metric_type=MetricType.CALMAR_RATIO,
            value=calmar,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # 8. Win Rate
        win_rate = Decimal(str(winning_trades_count / trades_count if trades_count > 0 else 0))
        metrics_dict[MetricType.WIN_RATE] = PerformanceMetric(
            metric_type=MetricType.WIN_RATE,
            value=win_rate,
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            strategy_id=strategy_id
        )
        
        # Adicionar Alpha e Beta se houver benchmark
        if benchmark_returns:
            benchmark_array = np.array(benchmark_returns)
            
            # 9. Alpha
            alpha = Decimal(str(self._calculate_alpha(
                returns_array, benchmark_array, risk_free_rate, start_date, end_date
            )))
            metrics_dict[MetricType.ALPHA] = PerformanceMetric(
                metric_type=MetricType.ALPHA,
                value=alpha,
                timestamp=datetime.utcnow(),
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
                strategy_id=strategy_id
            )
            
            # 10. Beta
            beta = Decimal(str(self._calculate_beta(returns_array, benchmark_array)))
            metrics_dict[MetricType.BETA] = PerformanceMetric(
                metric_type=MetricType.BETA,
                value=beta,
                timestamp=datetime.utcnow(),
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
                strategy_id=strategy_id
            )
            
            # 11. Information Ratio
            info_ratio = Decimal(str(self._calculate_information_ratio(
                returns_array, benchmark_array, start_date, end_date
            )))
            metrics_dict[MetricType.INFORMATION_RATIO] = PerformanceMetric(
                metric_type=MetricType.INFORMATION_RATIO,
                value=info_ratio,
                timestamp=datetime.utcnow(),
                time_range=time_range,
                start_date=start_date,
                end_date=end_date,
                strategy_id=strategy_id
            )
        
        # Criar coleção de métricas
        metric_collection = PerformanceMetricCollection(
            strategy_id=strategy_id or "unknown",
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics_dict,
            metadata=metadata or {}
        )
        
        # Criar análise de desempenho
        analysis = PerformanceAnalysis(
            strategy_id=strategy_id or "unknown",
            strategy_name=strategy_name or f"Strategy {strategy_id}",
            timestamp=datetime.utcnow(),
            time_range=time_range,
            start_date=start_date,
            end_date=end_date,
            metrics=metric_collection,
            initial_capital=Decimal(str(initial_capital)),
            final_capital=Decimal(str(final_capital)),
            trades_count=trades_count,
            winning_trades_count=winning_trades_count,
            losing_trades_count=losing_trades_count,
            additional_data={}
        )
        
        return analysis
    
    async def calculate_attribution_factors(
        self,
        strategy_returns: List[float],
        timestamps: List[datetime],
        factor_returns: Dict[str, List[float]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calcula fatores de atribuição de performance.
        
        Args:
            strategy_returns: Lista de retornos da estratégia
            timestamps: Lista de timestamps
            factor_returns: Dicionário com listas de retornos de fatores
            metadata: Metadados adicionais
            
        Returns:
            Resultado da atribuição de performance
        """
        self.logger.info("Calculando fatores de atribuição de performance")
        
        # Validar entradas
        if len(strategy_returns) != len(timestamps):
            raise ValueError("O número de retornos deve ser igual ao número de timestamps")
        
        for factor, returns in factor_returns.items():
            if len(returns) != len(strategy_returns):
                raise ValueError(f"O número de retornos do fator {factor} deve ser igual ao número de retornos da estratégia")
        
        # Converter para DataFrame
        data = {'strategy': strategy_returns}
        data.update(factor_returns)
        
        df = pd.DataFrame(data, index=timestamps)
        
        # Calcular correlações
        correlations = {}
        for factor in factor_returns.keys():
            correlations[factor] = np.corrcoef(strategy_returns, factor_returns[factor])[0, 1]
        
        # Realizar regressão linear para atribuição
        X = pd.DataFrame({k: v for k, v in factor_returns.items()})
        y = pd.Series(strategy_returns)
        
        # Adicionar constante para o alpha
        X = pd.concat([pd.Series(1, index=X.index, name='alpha'), X], axis=1)
        
        # Calcular coeficientes usando OLS (Ordinary Least Squares)
        coeffs = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
        
        # Extrair alpha e betas
        alpha = coeffs[0]
        betas = {factor: coeff for factor, coeff in zip(factor_returns.keys(), coeffs[1:])}
        
        # Calcular contribuição de cada fator
        contributions = {}
        for i, factor in enumerate(factor_returns.keys()):
            factor_mean = np.mean(factor_returns[factor])
            contributions[factor] = betas[factor] * factor_mean
        
        # Calcular contribuição do alpha
        alpha_contribution = alpha
        
        # Calcular R-squared
        y_pred = X.dot(coeffs)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Calcular contribuição percentual
        total_contribution = alpha_contribution + sum(contributions.values())
        
        contribution_pct = {
            'alpha': alpha_contribution / total_contribution if total_contribution != 0 else 0
        }
        
        for factor in factor_returns.keys():
            contribution_pct[factor] = contributions[factor] / total_contribution if total_contribution != 0 else 0
        
        # Montar resultado
        result = {
            'alpha': float(alpha),
            'betas': {k: float(v) for k, v in betas.items()},
            'correlations': {k: float(v) for k, v in correlations.items()},
            'contributions': {k: float(v) for k, v in contributions.items()},
            'alpha_contribution': float(alpha_contribution),
            'contribution_pct': {k: float(v) for k, v in contribution_pct.items()},
            'r_squared': float(r_squared),
            'start_date': min(timestamps).isoformat(),
            'end_date': max(timestamps).isoformat()
        }
        
        if metadata:
            result['metadata'] = metadata
        
        return result
    
    async def analyze_drawdowns(
        self,
        returns: List[float],
        timestamps: List[datetime],
        threshold: float = 0.0,
        max_drawdowns: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Analisa os principais drawdowns em uma série de retornos.
        
        Args:
            returns: Lista de retornos
            timestamps: Lista de timestamps
            threshold: Limite mínimo para considerar um drawdown (ex: 0.05 para 5%)
            max_drawdowns: Número máximo de drawdowns a retornar
            
        Returns:
            Lista com informações sobre os principais drawdowns
        """
        self.logger.info("Analisando drawdowns")
        
        # Validar entradas
        if len(returns) != len(timestamps):
            raise ValueError("O número de retornos deve ser igual ao número de timestamps")
        
        # Calcular retornos cumulativos
        cumulative = np.cumprod(np.array(returns) + 1) - 1
        
        # Encontrar drawdowns
        drawdowns = []
        high_watermark = 0
        drawdown_start = None
        current_drawdown = 0
        
        for i, (cum_return, timestamp) in enumerate(zip(cumulative, timestamps)):
            if cum_return > high_watermark:
                # Novo pico
                high_watermark = cum_return
                
                # Se estávamos em um drawdown e ele terminou, armazená-lo
                if drawdown_start is not None and current_drawdown <= -threshold:
                    drawdown_end = timestamps[i-1]
                    recovery_time = (drawdown_end - drawdown_start).days
                    
                    drawdowns.append({
                        'start_date': drawdown_start,
                        'end_date': drawdown_end,
                        'magnitude': float(current_drawdown),
                        'recovery_time_days': recovery_time,
                        'high_before': float(high_watermark)
                    })
                
                drawdown_start = None
                current_drawdown = 0
            else:
                # Estamos em drawdown
                if drawdown_start is None:
                    drawdown_start = timestamp
                
                current_drawdown = (cum_return - high_watermark) / (1 + high_watermark)
        
        # Verificar se ainda estamos em drawdown no fim da série
        if drawdown_start is not None and current_drawdown <= -threshold:
            drawdowns.append({
                'start_date': drawdown_start,
                'end_date': timestamps[-1],
                'magnitude': float(current_drawdown),
                'recovery_time_days': (timestamps[-1] - drawdown_start).days,
                'high_before': float(high_watermark),
                'recovered': False
            })
        
        # Ordenar por magnitude (do pior para o melhor)
        drawdowns.sort(key=lambda x: x['magnitude'])
        
        # Limitar ao número máximo especificado
        return drawdowns[:max_drawdowns]
    
    async def get_underwater_chart_data(
        self,
        returns: List[float],
        timestamps: List[datetime]
    ) -> Dict[str, Any]:
        """
        Gera dados para um gráfico underwater (drawdown ao longo do tempo).
        
        Args:
            returns: Lista de retornos
            timestamps: Lista de timestamps
            
        Returns:
            Dados para plotar o gráfico underwater
        """
        self.logger.info("Gerando dados para gráfico underwater")
        
        # Validar entradas
        if len(returns) != len(timestamps):
            raise ValueError("O número de retornos deve ser igual ao número de timestamps")
        
        # Calcular retornos cumulativos
        cumulative = np.cumprod(np.array(returns) + 1) - 1
        
        # Calcular high watermark (pico máximo até o momento)
        high_watermark = np.maximum.accumulate(cumulative)
        
        # Calcular drawdown
        underwater = (cumulative - high_watermark) / (1 + high_watermark)
        
        # Preparar dados para o gráfico
        chart_data = {
            'timestamps': [ts.isoformat() for ts in timestamps],
            'underwater': [float(dd) for dd in underwater],
            'max_drawdown': float(np.min(underwater)),
            'current_drawdown': float(underwater[-1])
        }
        
        return chart_data
    
    def _determine_time_range(self, start_date: datetime, end_date: datetime) -> str:
        """Determina o intervalo de tempo com base nas datas."""
        delta = end_date - start_date
        
        if delta <= timedelta(days=1):
            return "1d"
        elif delta <= timedelta(days=7):
            return "1w"
        elif delta <= timedelta(days=30):
            return "1m"
        elif delta <= timedelta(days=90):
            return "3m"
        elif delta <= timedelta(days=180):
            return "6m"
        elif delta <= timedelta(days=365):
            return "1y"
        elif delta <= timedelta(days=365*2):
            return "2y"
        else:
            return "all"
    
    def _calculate_cumulative_return(self, returns: np.ndarray) -> float:
        """Calcula o retorno cumulativo de uma série de retornos."""
        return np.prod(1 + returns) - 1
    
    def _calculate_annualized_return(
        self, 
        returns: np.ndarray, 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula o retorno anualizado."""
        cumulative_return = self._calculate_cumulative_return(returns)
        years = (end_date - start_date).days / 365.25
        
        if years < 0.01:  # Se for menos de ~3.65 dias
            # Extrapolação para período anual
            return ((1 + cumulative_return) ** (1 / years)) - 1 if years > 0 else 0
        
        return ((1 + cumulative_return) ** (1 / years)) - 1
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calcula o drawdown máximo."""
        cumulative = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (1 + running_max)
        return abs(float(np.min(drawdown)))
    
    def _calculate_volatility(
        self, 
        returns: np.ndarray, 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula a volatilidade anualizada."""
        # Estimar número de períodos por ano
        days = (end_date - start_date).days
        if days <= 0:
            return 0
            
        periods_per_year = len(returns) * 365.25 / days
        
        return float(np.std(returns) * np.sqrt(periods_per_year))
    
    def _calculate_sharpe_ratio(
        self, 
        returns: np.ndarray, 
        risk_free_rate: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula o Sharpe Ratio."""
        volatility = self._calculate_volatility(returns, start_date, end_date)
        if volatility == 0:
            return 0
            
        annualized_return = self._calculate_annualized_return(returns, start_date, end_date)
        excess_return = annualized_return - risk_free_rate
        
        return excess_return / volatility
    
    def _calculate_sortino_ratio(
        self, 
        returns: np.ndarray, 
        risk_free_rate: float, 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula o Sortino Ratio."""
        # Calcular retorno anualizado
        annualized_return = self._calculate_annualized_return(returns, start_date, end_date)
        excess_return = annualized_return - risk_free_rate
        
        # Calcular dowside deviation (volatilidade apenas dos retornos negativos)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0 or excess_return < 0:
            return 0
            
        # Estimar número de períodos por ano
        days = (end_date - start_date).days
        if days <= 0:
            return 0
            
        periods_per_year = len(returns) * 365.25 / days
        
        downside_deviation = np.std(negative_returns) * np.sqrt(periods_per_year)
        
        if downside_deviation == 0:
            return 0
            
        return excess_return / downside_deviation
    
    def _calculate_calmar_ratio(
        self, 
        returns: np.ndarray, 
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula o Calmar Ratio."""
        max_drawdown = self._calculate_max_drawdown(returns)
        annualized_return = self._calculate_annualized_return(returns, start_date, end_date)
        
        if max_drawdown == 0:
            return 0
            
        return annualized_return / max_drawdown
    
    def _calculate_alpha(
        self, 
        returns: np.ndarray, 
        benchmark_returns: np.ndarray, 
        risk_free_rate: float,
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula o Alpha."""
        beta = self._calculate_beta(returns, benchmark_returns)
        
        # Calcular retornos anualizados
        strategy_annual_return = self._calculate_annualized_return(returns, start_date, end_date)
        benchmark_annual_return = self._calculate_annualized_return(benchmark_returns, start_date, end_date)
        
        # Alpha = Retorno da Estratégia - (Taxa Livre de Risco + Beta * (Retorno do Benchmark - Taxa Livre de Risco))
        return strategy_annual_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
    
    def _calculate_beta(self, returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calcula o Beta."""
        if len(returns) != len(benchmark_returns):
            raise ValueError("Os arrays de retorno devem ter o mesmo tamanho")
            
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 0
            
        return covariance / benchmark_variance
    
    def _calculate_information_ratio(
        self, 
        returns: np.ndarray, 
        benchmark_returns: np.ndarray,
        start_date: datetime, 
        end_date: datetime
    ) -> float:
        """Calcula o Information Ratio."""
        # Calcular tracking error (volatilidade do retorno ativo)
        tracking_difference = returns - benchmark_returns
        
        # Estimar número de períodos por ano
        days = (end_date - start_date).days
        if days <= 0:
            return 0
            
        periods_per_year = len(returns) * 365.25 / days
        
        tracking_error = np.std(tracking_difference) * np.sqrt(periods_per_year)
        
        if tracking_error == 0:
            return 0
            
        # Calcular retornos anualizados
        strategy_annual_return = self._calculate_annualized_return(returns, start_date, end_date)
        benchmark_annual_return = self._calculate_annualized_return(benchmark_returns, start_date, end_date)
        
        # Information Ratio = (Retorno da Estratégia - Retorno do Benchmark) / Tracking Error
        return (strategy_annual_return - benchmark_annual_return) / tracking_error