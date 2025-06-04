import logging
from enum import Enum
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class VaRMethod(Enum):
    """Métodos de cálculo de Value at Risk."""
    HISTORICAL = "historical"  # Baseado em dados históricos empíricos
    PARAMETRIC = "parametric"  # Baseado em distribuição normal ou t-student
    MONTE_CARLO = "monte_carlo"  # Simulação Monte Carlo


class VaRModel:
    """
    Modelo para cálculo de Value at Risk (VaR) com diferentes metodologias.
    O VaR representa a perda máxima esperada com determinado nível de confiança.
    """
    
    def __init__(
        self,
        default_method: VaRMethod = VaRMethod.HISTORICAL,
        default_confidence_level: float = 0.95,
        time_horizon: int = 1,
        use_ewma: bool = True,
        ewma_lambda: float = 0.94,
        mcmc_simulations: int = 10000,
        distribution: str = "normal",  # "normal" ou "t-student"
        degrees_of_freedom: int = 5,  # Para t-student
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o modelo de VaR.
        
        Args:
            default_method: Método de cálculo padrão
            default_confidence_level: Nível de confiança padrão (0-1)
            time_horizon: Horizonte de tempo em dias
            use_ewma: Se deve usar EWMA para volatilidade
            ewma_lambda: Parâmetro lambda para EWMA (0-1)
            mcmc_simulations: Número de simulações para Monte Carlo
            distribution: Distribuição para método paramétrico
            degrees_of_freedom: Graus de liberdade para t-student
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._default_method = default_method
        self._default_confidence_level = default_confidence_level
        self._time_horizon = time_horizon
        self._use_ewma = use_ewma
        self._ewma_lambda = ewma_lambda
        self._mcmc_simulations = mcmc_simulations
        self._distribution = distribution
        self._degrees_of_freedom = degrees_of_freedom
    
    def calculate(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        method: Optional[VaRMethod] = None,
        confidence_level: Optional[float] = None,
        weights: Optional[List[float]] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        portfolio_value: float = 100.0
    ) -> float:
        """
        Calcula o Value at Risk para um ativo ou portfólio.
        
        Args:
            returns: Série histórica de retornos percentuais
            method: Método de cálculo (usa padrão se None)
            confidence_level: Nível de confiança (usa padrão se None)
            weights: Pesos dos ativos para VaR de portfólio (opcional)
            covariance_matrix: Matriz de covariância para VaR paramétrico de portfólio
            portfolio_value: Valor do portfólio para cálculo de VaR absoluto
            
        Returns:
            Valor do VaR em percentual
        """
        method = method or self._default_method
        confidence_level = confidence_level or self._default_confidence_level
        
        # Converter para numpy array se necessário
        if isinstance(returns, pd.Series):
            returns = returns.values
        elif isinstance(returns, list):
            returns = np.array(returns)
        
        # Verificar dados
        if len(returns) < 30:
            self._logger.warning(f"Poucos dados para cálculo confiável de VaR: {len(returns)} pontos")
        
        # Calcular VaR usando método selecionado
        if method == VaRMethod.HISTORICAL:
            var_value = self._calculate_historical_var(returns, confidence_level)
        elif method == VaRMethod.PARAMETRIC:
            if weights is not None and covariance_matrix is not None:
                var_value = self._calculate_parametric_portfolio_var(
                    returns, weights, covariance_matrix, confidence_level
                )
            else:
                var_value = self._calculate_parametric_var(returns, confidence_level)
        elif method == VaRMethod.MONTE_CARLO:
            if weights is not None and covariance_matrix is not None:
                var_value = self._calculate_monte_carlo_portfolio_var(
                    returns, weights, covariance_matrix, confidence_level
                )
            else:
                var_value = self._calculate_monte_carlo_var(returns, confidence_level)
        else:
            raise ValueError(f"Método de cálculo de VaR não suportado: {method}")
        
        # Ajustar para horizonte de tempo se necessário
        if self._time_horizon > 1:
            # Regra da raiz quadrada do tempo
            var_value = var_value * np.sqrt(self._time_horizon)
        
        return var_value
    
    def _calculate_historical_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcula VaR usando o método histórico.
        
        Args:
            returns: Série histórica de retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do VaR em percentual
        """
        # Ordenar retornos em ordem crescente
        sorted_returns = np.sort(returns)
        
        # Encontrar o percentil correspondente ao nível de confiança
        var_percentile = 1 - confidence_level
        var_index = int(np.floor(var_percentile * len(sorted_returns)))
        
        # O VaR é o valor negativo do retorno neste percentil
        var_value = -sorted_returns[var_index]
        
        return var_value * 100  # Converter para percentual
    
    def _calculate_parametric_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcula VaR usando o método paramétrico (distribuição normal ou t-student).
        
        Args:
            returns: Série histórica de retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do VaR em percentual
        """
        # Calcular média e desvio padrão dos retornos
        mean_return = np.mean(returns)
        
        # Usar EWMA se configurado
        if self._use_ewma:
            volatility = self._calculate_ewma_volatility(returns)
        else:
            volatility = np.std(returns)
        
        # Calcular fator Z baseado na distribuição e nível de confiança
        if self._distribution == "normal":
            z_score = stats.norm.ppf(confidence_level)
        else:  # t-student
            z_score = stats.t.ppf(confidence_level, self._degrees_of_freedom)
        
        # Calcular VaR
        var_value = -(mean_return + volatility * z_score)
        
        return var_value * 100  # Converter para percentual
    
    def _calculate_parametric_portfolio_var(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        confidence_level: float
    ) -> float:
        """
        Calcula VaR paramétrico para um portfólio.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do VaR em percentual
        """
        # Converter pesos para array numpy
        weights_array = np.array(weights)
        
        # Calcular média dos retornos para cada ativo
        if isinstance(returns, np.ndarray) and len(returns.shape) == 2:
            # Se returns for matriz de retornos (ativos x tempo)
            mean_returns = np.mean(returns, axis=1)
        else:
            # Se for apenas um ativo
            mean_returns = np.mean(returns)
        
        # Calcular retorno esperado do portfólio
        portfolio_return = np.dot(weights_array, mean_returns)
        
        # Calcular volatilidade do portfólio
        portfolio_volatility = np.sqrt(np.dot(np.dot(weights_array, covariance_matrix), weights_array))
        
        # Calcular fator Z baseado na distribuição e nível de confiança
        if self._distribution == "normal":
            z_score = stats.norm.ppf(confidence_level)
        else:  # t-student
            z_score = stats.t.ppf(confidence_level, self._degrees_of_freedom)
        
        # Calcular VaR
        var_value = -(portfolio_return + portfolio_volatility * z_score)
        
        return var_value * 100  # Converter para percentual
    
    def _calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcula VaR usando simulação Monte Carlo.
        
        Args:
            returns: Série histórica de retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do VaR em percentual
        """
        # Calcular média e desvio padrão dos retornos
        mean_return = np.mean(returns)
        
        # Usar EWMA se configurado
        if self._use_ewma:
            volatility = self._calculate_ewma_volatility(returns)
        else:
            volatility = np.std(returns)
        
        # Gerar simulações usando distribuição escolhida
        if self._distribution == "normal":
            simulated_returns = np.random.normal(
                mean_return, volatility, self._mcmc_simulations
            )
        else:  # t-student
            # Escalar para match de volatilidade
            df = self._degrees_of_freedom
            scale = volatility * np.sqrt((df - 2) / df)
            simulated_returns = mean_return + stats.t.rvs(
                df=df, scale=scale, size=self._mcmc_simulations
            )
        
        # Calcular VaR a partir das simulações
        sorted_returns = np.sort(simulated_returns)
        var_percentile = 1 - confidence_level
        var_index = int(np.floor(var_percentile * len(sorted_returns)))
        var_value = -sorted_returns[var_index]
        
        return var_value * 100  # Converter para percentual
    
    def _calculate_monte_carlo_portfolio_var(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        confidence_level: float
    ) -> float:
        """
        Calcula VaR de portfólio usando simulação Monte Carlo.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do VaR em percentual
        """
        # Converter pesos para array numpy
        weights_array = np.array(weights)
        
        # Calcular média dos retornos para cada ativo
        if isinstance(returns, np.ndarray) and len(returns.shape) == 2:
            # Se returns for matriz de retornos (ativos x tempo)
            mean_returns = np.mean(returns, axis=1)
        else:
            # Se for apenas um ativo
            mean_returns = np.array([np.mean(returns)])
            covariance_matrix = np.array([[np.var(returns)]])
        
        # Gerar simulações usando distribuição multivariada
        if self._distribution == "normal":
            simulated_returns = np.random.multivariate_normal(
                mean_returns, covariance_matrix, self._mcmc_simulations
            )
        else:
            # Para t-student multivariada, vamos usar uma aproximação via normal-t
            simulated_returns = np.zeros((self._mcmc_simulations, len(mean_returns)))
            for i in range(self._mcmc_simulations):
                # Gerar chi-quadrado para scaling
                chi_sq = np.random.chisquare(self._degrees_of_freedom)
                # Gerar normal multivariada
                normal_sample = np.random.multivariate_normal(np.zeros_like(mean_returns), covariance_matrix)
                # Escalar para t multivariada
                t_sample = mean_returns + normal_sample * np.sqrt(self._degrees_of_freedom / chi_sq)
                simulated_returns[i] = t_sample
        
        # Calcular retornos do portfólio para cada simulação
        portfolio_returns = np.dot(simulated_returns, weights_array)
        
        # Calcular VaR a partir das simulações
        sorted_returns = np.sort(portfolio_returns)
        var_percentile = 1 - confidence_level
        var_index = int(np.floor(var_percentile * len(sorted_returns)))
        var_value = -sorted_returns[var_index]
        
        return var_value * 100  # Converter para percentual
    
    def _calculate_ewma_volatility(self, returns: np.ndarray) -> float:
        """
        Calcula a volatilidade usando o modelo EWMA (Exponentially Weighted Moving Average).
        
        Args:
            returns: Série histórica de retornos
            
        Returns:
            Volatilidade EWMA
        """
        if len(returns) <= 1:
            return np.std(returns) if len(returns) == 1 else 0
        
        # Inicializar com variância amostral dos primeiros n pontos
        n_init = min(30, len(returns) // 3)
        variance = np.var(returns[:n_init])
        
        # Aplicar EWMA
        for t in range(n_init, len(returns)):
            variance = self._ewma_lambda * variance + (1 - self._ewma_lambda) * returns[t-1]**2
        
        return np.sqrt(variance)
    
    def calculate_component_var(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        confidence_level: Optional[float] = None
    ) -> Dict[int, float]:
        """
        Calcula a contribuição de cada componente para o VaR do portfólio.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            confidence_level: Nível de confiança (usa padrão se None)
            
        Returns:
            Dicionário com contribuição de cada componente para o VaR
        """
        confidence_level = confidence_level or self._default_confidence_level
        weights_array = np.array(weights)
        
        # Calcular VaR do portfólio
        portfolio_var = self._calculate_parametric_portfolio_var(
            returns, weights, covariance_matrix, confidence_level
        )
        
        # Calcular contribuições marginais
        portfolio_volatility = np.sqrt(np.dot(np.dot(weights_array, covariance_matrix), weights_array))
        
        # Calcular fator Z baseado na distribuição e nível de confiança
        if self._distribution == "normal":
            z_score = stats.norm.ppf(confidence_level)
        else:  # t-student
            z_score = stats.t.ppf(confidence_level, self._degrees_of_freedom)
        
        # Contribuição de cada componente
        marginal_contributions = {}
        for i in range(len(weights)):
            # Calcular derivada parcial do VaR em relação ao peso do ativo i
            marginal_var = -z_score * np.dot(covariance_matrix[i], weights_array) / portfolio_volatility
            
            # Contribuição do componente = peso * derivada parcial
            component_var = weights[i] * marginal_var
            
            # Normalizar para soma = VaR total
            marginal_contributions[i] = component_var / np.sum(np.abs(component_var)) * portfolio_var
        
        return marginal_contributions
    
    def stress_test(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        scenarios: Dict[str, Dict[int, float]],
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Realiza teste de estresse no VaR sob diferentes cenários.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            scenarios: Dicionário de cenários com choques nos retornos
            confidence_level: Nível de confiança (usa padrão se None)
            
        Returns:
            Dicionário com VaR sob diferentes cenários
        """
        confidence_level = confidence_level or self._default_confidence_level
        weights_array = np.array(weights)
        
        # Calcular VaR base
        base_var = self._calculate_parametric_portfolio_var(
            returns, weights, covariance_matrix, confidence_level
        )
        
        results = {"base": base_var}
        
        # Calcular média dos retornos para cada ativo
        if isinstance(returns, np.ndarray) and len(returns.shape) == 2:
            mean_returns = np.mean(returns, axis=1)
        else:
            mean_returns = np.array([np.mean(returns)])
        
        # Aplicar cenários
        for scenario_name, shocks in scenarios.items():
            # Copiar retornos médios
            shocked_returns = mean_returns.copy()
            
            # Aplicar choques
            for asset_idx, shock_value in shocks.items():
                shocked_returns[asset_idx] += shock_value
            
            # Calcular VaR com retornos modificados
            scenario_var = self._calculate_parametric_portfolio_var(
                shocked_returns, weights, covariance_matrix, confidence_level
            )
            
            results[scenario_name] = scenario_var
        
        return results