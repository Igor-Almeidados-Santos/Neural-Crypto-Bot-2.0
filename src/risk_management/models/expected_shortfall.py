import logging
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


class ESMethod(Enum):
    """Métodos de cálculo de Expected Shortfall."""
    HISTORICAL = "historical"  # Baseado em dados históricos empíricos
    PARAMETRIC = "parametric"  # Baseado em distribuição normal ou t-student
    MONTE_CARLO = "monte_carlo"  # Simulação Monte Carlo


class ExpectedShortfallModel:
    """
    Modelo para cálculo de Expected Shortfall (ES) ou Conditional Value at Risk (CVaR).
    O ES representa a perda média esperada nos piores cenários, além do VaR.
    """
    
    def __init__(
        self,
        default_method: ESMethod = ESMethod.HISTORICAL,
        default_confidence_level: float = 0.95,
        time_horizon: int = 1,
        mcmc_simulations: int = 10000,
        distribution: str = "t-student",  # "normal" ou "t-student"
        degrees_of_freedom: int = 5,  # Para t-student
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o modelo de Expected Shortfall.
        
        Args:
            default_method: Método de cálculo padrão
            default_confidence_level: Nível de confiança padrão (0-1)
            time_horizon: Horizonte de tempo em dias
            mcmc_simulations: Número de simulações para Monte Carlo
            distribution: Distribuição para método paramétrico
            degrees_of_freedom: Graus de liberdade para t-student
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._default_method = default_method
        self._default_confidence_level = default_confidence_level
        self._time_horizon = time_horizon
        self._mcmc_simulations = mcmc_simulations
        self._distribution = distribution
        self._degrees_of_freedom = degrees_of_freedom
    
    def calculate(
        self,
        returns: Union[List[float], np.ndarray, pd.Series],
        method: Optional[ESMethod] = None,
        confidence_level: Optional[float] = None,
        weights: Optional[List[float]] = None,
        covariance_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calcula o Expected Shortfall para um ativo ou portfólio.
        
        Args:
            returns: Série histórica de retornos percentuais
            method: Método de cálculo (usa padrão se None)
            confidence_level: Nível de confiança (usa padrão se None)
            weights: Pesos dos ativos para ES de portfólio (opcional)
            covariance_matrix: Matriz de covariância para ES paramétrico de portfólio
            
        Returns:
            Valor do ES em percentual
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
            self._logger.warning(f"Poucos dados para cálculo confiável de ES: {len(returns)} pontos")
        
        # Calcular ES usando método selecionado
        if method == ESMethod.HISTORICAL:
            es_value = self._calculate_historical_es(returns, confidence_level)
        elif method == ESMethod.PARAMETRIC:
            if weights is not None and covariance_matrix is not None:
                es_value = self._calculate_parametric_portfolio_es(
                    returns, weights, covariance_matrix, confidence_level
                )
            else:
                es_value = self._calculate_parametric_es(returns, confidence_level)
        elif method == ESMethod.MONTE_CARLO:
            if weights is not None and covariance_matrix is not None:
                es_value = self._calculate_monte_carlo_portfolio_es(
                    returns, weights, covariance_matrix, confidence_level
                )
            else:
                es_value = self._calculate_monte_carlo_es(returns, confidence_level)
        else:
            raise ValueError(f"Método de cálculo de ES não suportado: {method}")
        
        # Ajustar para horizonte de tempo se necessário
        if self._time_horizon > 1:
            # Regra da raiz quadrada do tempo
            es_value = es_value * np.sqrt(self._time_horizon)
        
        return es_value
    
    def _calculate_historical_es(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcula ES usando o método histórico.
        
        Args:
            returns: Série histórica de retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do ES em percentual
        """
        # Ordenar retornos em ordem crescente
        sorted_returns = np.sort(returns)
        
        # Encontrar o índice correspondente ao VaR
        var_percentile = 1 - confidence_level
        var_index = int(np.floor(var_percentile * len(sorted_returns)))
        
        # O ES é a média dos retornos piores que o VaR
        es_value = -np.mean(sorted_returns[:var_index+1])
        
        return es_value * 100  # Converter para percentual
    
    def _calculate_parametric_es(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcula ES usando o método paramétrico (distribuição normal ou t-student).
        
        Args:
            returns: Série histórica de retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do ES em percentual
        """
        # Calcular média e desvio padrão dos retornos
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        # Calcular ES com base na distribuição
        alpha = 1 - confidence_level
        if self._distribution == "normal":
            # Para distribuição normal, ES = média + volatility * pdf(z) / alpha
            z_score = stats.norm.ppf(confidence_level)
            pdf_z = stats.norm.pdf(z_score)
            es_value = -(mean_return + volatility * pdf_z / alpha)
        else:  # t-student
            # Para t-student, ver fórmula em McNeil et al. (2005)
            t_quantile = stats.t.ppf(alpha, self._degrees_of_freedom)
            pdf_t = stats.t.pdf(t_quantile, self._degrees_of_freedom)
            factor = (self._degrees_of_freedom + t_quantile**2) / (self._degrees_of_freedom - 1)
            es_value = -(mean_return + volatility * (pdf_t / alpha) * factor)
        
        return es_value * 100  # Converter para percentual
    
    def _calculate_parametric_portfolio_es(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        confidence_level: float
    ) -> float:
        """
        Calcula ES paramétrico para um portfólio.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do ES em percentual
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
        
        # Calcular ES com base na distribuição
        alpha = 1 - confidence_level
        if self._distribution == "normal":
            # Para distribuição normal, ES = média + volatility * pdf(z) / alpha
            z_score = stats.norm.ppf(confidence_level)
            pdf_z = stats.norm.pdf(z_score)
            es_value = -(portfolio_return + portfolio_volatility * pdf_z / alpha)
        else:  # t-student
            # Para t-student, ver fórmula em McNeil et al. (2005)
            t_quantile = stats.t.ppf(alpha, self._degrees_of_freedom)
            pdf_t = stats.t.pdf(t_quantile, self._degrees_of_freedom)
            factor = (self._degrees_of_freedom + t_quantile**2) / (self._degrees_of_freedom - 1)
            es_value = -(portfolio_return + portfolio_volatility * (pdf_t / alpha) * factor)
        
        return es_value * 100  # Converter para percentual
    
    def _calculate_monte_carlo_es(self, returns: np.ndarray, confidence_level: float) -> float:
        """
        Calcula ES usando simulação Monte Carlo.
        
        Args:
            returns: Série histórica de retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do ES em percentual
        """
        # Calcular média e desvio padrão dos retornos
        mean_return = np.mean(returns)
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
        
        # Ordenar simulações
        sorted_returns = np.sort(simulated_returns)
        
        # Encontrar o índice correspondente ao VaR
        var_percentile = 1 - confidence_level
        var_index = int(np.floor(var_percentile * len(sorted_returns)))
        
        # O ES é a média dos retornos piores que o VaR
        es_value = -np.mean(sorted_returns[:var_index+1])
        
        return es_value * 100  # Converter para percentual
    
    def _calculate_monte_carlo_portfolio_es(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        confidence_level: float
    ) -> float:
        """
        Calcula ES de portfólio usando simulação Monte Carlo.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            confidence_level: Nível de confiança (0-1)
            
        Returns:
            Valor do ES em percentual
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
        
        # Ordenar simulações
        sorted_returns = np.sort(portfolio_returns)
        
        # Encontrar o índice correspondente ao VaR
        var_percentile = 1 - confidence_level
        var_index = int(np.floor(var_percentile * len(sorted_returns)))
        
        # O ES é a média dos retornos piores que o VaR
        es_value = -np.mean(sorted_returns[:var_index+1])
        
        return es_value * 100  # Converter para percentual
    
    def stress_test(
        self,
        returns: np.ndarray,
        weights: List[float],
        covariance_matrix: np.ndarray,
        stress_factors: List[float],
        confidence_level: Optional[float] = None
    ) -> List[float]:
        """
        Realiza teste de estresse no ES com diferentes fatores de estresse na volatilidade.
        
        Args:
            returns: Retornos históricos dos ativos
            weights: Pesos dos ativos no portfólio
            covariance_matrix: Matriz de covariância dos retornos
            stress_factors: Lista de fatores de estresse (ex: [1.2, 1.5, 2.0])
            confidence_level: Nível de confiança (usa padrão se None)
            
        Returns:
            Lista com ES sob diferentes níveis de estresse
        """
        confidence_level = confidence_level or self._default_confidence_level
        results = []
        
        for stress_factor in stress_factors:
            # Aplicar fator de estresse na matriz de covariância
            stressed_covariance = covariance_matrix * (stress_factor ** 2)
            
            # Calcular ES com covariância estressada
            stressed_es = self._calculate_parametric_portfolio_es(
                returns, weights, stressed_covariance, confidence_level
            )
            
            results.append(stressed_es)
        
        return results
    
    def compare_methods(
        self,
        returns: np.ndarray,
        confidence_level: Optional[float] = None,
        weights: Optional[List[float]] = None,
        covariance_matrix: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compara os resultados do ES usando diferentes métodos.
        
        Args:
            returns: Retornos históricos dos ativos
            confidence_level: Nível de confiança (usa padrão se None)
            weights: Pesos dos ativos no portfólio (opcional)
            covariance_matrix: Matriz de covariância dos retornos (opcional)
            
        Returns:
            Dicionário com resultados de cada método
        """
        confidence_level = confidence_level or self._default_confidence_level
        results = {}
        
        # Calcular ES com cada método
        for method in ESMethod:
            try:
                if weights is not None and covariance_matrix is not None:
                    es_value = self.calculate(
                        returns, method, confidence_level, weights, covariance_matrix
                    )
                else:
                    es_value = self.calculate(returns, method, confidence_level)
                
                results[method.value] = es_value
            except Exception as e:
                self._logger.error(f"Erro ao calcular ES com método {method.value}: {e}")
                results[method.value] = None
        
        return results