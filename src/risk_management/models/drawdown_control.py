import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class DrawdownControlStrategy(Enum):
    """Estratégias de controle de drawdown."""
    FIXED_THRESHOLD = "fixed_threshold"  # Limitar drawdown a um valor fixo
    ADAPTIVE_THRESHOLD = "adaptive_threshold"  # Threshold adaptativo baseado em volatilidade
    TIME_VARYING = "time_varying"  # Threshold que varia com o tempo
    TRAILING_STOP = "trailing_stop"  # Stop baseado em pico histórico


class DrawdownControl:
    """
    Modelo para controle e gestão de drawdown.
    O drawdown representa a queda percentual de um valor em relação ao seu pico histórico.
    """
    
    def __init__(
        self,
        max_drawdown_pct: float = 10.0,
        warning_threshold_pct: float = 7.0,
        strategy: DrawdownControlStrategy = DrawdownControlStrategy.FIXED_THRESHOLD,
        lookback_window: int = 60,  # Dias para cálculo de volatilidade adaptativa
        volatility_multiplier: float = 2.0,  # Para threshold adaptativo
        time_decay_factor: float = 0.5,  # Para threshold time-varying
        trailing_pct: float = 5.0,  # Para trailing stop
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o modelo de controle de drawdown.
        
        Args:
            max_drawdown_pct: Limite máximo de drawdown em percentual
            warning_threshold_pct: Limite de alerta em percentual
            strategy: Estratégia de controle de drawdown
            lookback_window: Janela para cálculo de volatilidade
            volatility_multiplier: Multiplicador de volatilidade para threshold adaptativo
            time_decay_factor: Fator de decaimento para threshold time-varying
            trailing_pct: Percentual para trailing stop
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._max_drawdown_pct = max_drawdown_pct
        self._warning_threshold_pct = warning_threshold_pct
        self._strategy = strategy
        self._lookback_window = lookback_window
        self._volatility_multiplier = volatility_multiplier
        self._time_decay_factor = time_decay_factor
        self._trailing_pct = trailing_pct
        
        # Histórico para cada portfólio/estratégia
        self._equity_history: Dict[str, List[float]] = {}
        self._peak_history: Dict[str, List[float]] = {}
        self._drawdown_history: Dict[str, List[float]] = {}
        self._timestamp_history: Dict[str, List[datetime]] = {}
    
    def update(
        self,
        entity_id: str,
        current_value: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[float, bool, bool]:
        """
        Atualiza o histórico e calcula o drawdown atual.
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            current_value: Valor atual do portfólio ou estratégia
            timestamp: Data/hora da atualização (usa timestamp atual se não fornecido)
            
        Returns:
            Tupla com (drawdown_atual, warning_triggered, critical_triggered)
        """
        now = timestamp or datetime.utcnow()
        
        # Inicializar históricos se necessário
        if entity_id not in self._equity_history:
            self._equity_history[entity_id] = []
            self._peak_history[entity_id] = []
            self._drawdown_history[entity_id] = []
            self._timestamp_history[entity_id] = []
        
        # Adicionar valor atual ao histórico
        self._equity_history[entity_id].append(current_value)
        self._timestamp_history[entity_id].append(now)
        
        # Calcular pico histórico
        if not self._peak_history[entity_id] or current_value > self._peak_history[entity_id][-1]:
            peak = current_value
        else:
            peak = self._peak_history[entity_id][-1]
        
        self._peak_history[entity_id].append(peak)
        
        # Calcular drawdown atual
        if peak > 0:
            drawdown_pct = (peak - current_value) / peak * 100
        else:
            drawdown_pct = 0
        
        self._drawdown_history[entity_id].append(drawdown_pct)
        
        # Calcular threshold atual com base na estratégia
        current_threshold = self._calculate_threshold(entity_id)
        
        # Verificar se os limites foram ultrapassados
        warning_triggered = drawdown_pct >= self._warning_threshold_pct
        critical_triggered = drawdown_pct >= current_threshold
        
        # Registrar se crítico
        if critical_triggered:
            self._logger.warning(
                f"Drawdown crítico para {entity_id}: {drawdown_pct:.2f}% (limite: {current_threshold:.2f}%)"
            )
        elif warning_triggered:
            self._logger.info(
                f"Alerta de drawdown para {entity_id}: {drawdown_pct:.2f}% (limite: {current_threshold:.2f}%)"
            )
        
        return drawdown_pct, warning_triggered, critical_triggered
    
    def _calculate_threshold(self, entity_id: str) -> float:
        """
        Calcula o threshold atual com base na estratégia.
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            
        Returns:
            Threshold atual
        """
        if self._strategy == DrawdownControlStrategy.FIXED_THRESHOLD:
            return self._max_drawdown_pct
        
        elif self._strategy == DrawdownControlStrategy.ADAPTIVE_THRESHOLD:
            # Calcular volatilidade recente
            if len(self._equity_history[entity_id]) < 2:
                return self._max_drawdown_pct
            
            # Usar apenas a janela de lookback
            lookback = min(self._lookback_window, len(self._equity_history[entity_id]) - 1)
            recent_values = self._equity_history[entity_id][-lookback:]
            
            # Calcular retornos diários
            returns = np.diff(recent_values) / recent_values[:-1]
            
            # Calcular volatilidade
            volatility = np.std(returns) * 100  # Em percentual
            
            # Ajustar threshold com base na volatilidade
            adaptive_threshold = min(
                volatility * self._volatility_multiplier,
                self._max_drawdown_pct * 1.5  # Limite superior
            )
            
            # Nunca permitir menos que 50% do threshold padrão
            adaptive_threshold = max(adaptive_threshold, self._max_drawdown_pct * 0.5)
            
            return adaptive_threshold
        
        elif self._strategy == DrawdownControlStrategy.TIME_VARYING:
            # Threshold que aumenta com o tempo sem drawdown
            if len(self._drawdown_history[entity_id]) < 2:
                return self._max_drawdown_pct
            
            # Encontrar quanto tempo passou desde o último drawdown significativo
            significant_dd = self._warning_threshold_pct * 0.8
            
            # Procurar pelo último drawdown significativo
            last_significant_idx = None
            for i in range(len(self._drawdown_history[entity_id]) - 1, -1, -1):
                if self._drawdown_history[entity_id][i] >= significant_dd:
                    last_significant_idx = i
                    break
            
            if last_significant_idx is None:
                # Nunca teve drawdown significativo, usar base
                days_since_dd = len(self._drawdown_history[entity_id])
            else:
                # Calcular dias desde o último drawdown significativo
                days_since_dd = 0
                if len(self._timestamp_history[entity_id]) > last_significant_idx:
                    last_dd_time = self._timestamp_history[entity_id][last_significant_idx]
                    current_time = self._timestamp_history[entity_id][-1]
                    days_since_dd = (current_time - last_dd_time).days
            
            # Ajustar threshold com base no tempo
            time_factor = 1 + (days_since_dd * self._time_decay_factor / 365)  # Anualizado
            time_varying_threshold = min(
                self._max_drawdown_pct * time_factor,
                self._max_drawdown_pct * 2  # Limite superior
            )
            
            return time_varying_threshold
        
        elif self._strategy == DrawdownControlStrategy.TRAILING_STOP:
            # Usar um trailing stop baseado no pico recente
            # Se o drawdown atual está abaixo do trailing stop, usar o trailing
            # Caso contrário, usar o threshold padrão
            if len(self._drawdown_history[entity_id]) < 2:
                return self._max_drawdown_pct
            
            current_drawdown = self._drawdown_history[entity_id][-1]
            
            if current_drawdown < self._trailing_pct:
                return self._trailing_pct
            else:
                return self._max_drawdown_pct
        
        else:
            return self._max_drawdown_pct
    
    def calculate_drawdown(self, equity_curve: List[float]) -> Tuple[float, float, int]:
        """
        Calcula o drawdown máximo e atual para uma curva de equity.
        
        Args:
            equity_curve: Lista com valores históricos do portfólio/estratégia
            
        Returns:
            Tupla com (max_drawdown, current_drawdown, drawdown_duration)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0.0, 0
        
        # Converter para numpy array
        equity_array = np.array(equity_curve)
        
        # Calcular picos (máximos acumulados)
        peaks = np.maximum.accumulate(equity_array)
        
        # Calcular drawdowns
        drawdowns = (peaks - equity_array) / peaks * 100
        
        # Encontrar drawdown máximo
        max_drawdown = np.max(drawdowns)
        
        # Calcular drawdown atual
        current_drawdown = drawdowns[-1]
        
        # Calcular duração do drawdown atual
        if current_drawdown > 0:
            # Encontrar o índice do último pico
            last_peak_idx = len(equity_array) - 1
            while last_peak_idx > 0 and equity_array[last_peak_idx] < peaks[last_peak_idx]:
                last_peak_idx -= 1
            
            # Calcular duração
            drawdown_duration = len(equity_array) - 1 - last_peak_idx
        else:
            drawdown_duration = 0
        
        return max_drawdown, current_drawdown, drawdown_duration
    
    def get_drawdown_history(self, entity_id: str) -> Dict[str, List]:
        """
        Obtém o histórico de drawdown para uma entidade.
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            
        Returns:
            Dicionário com históricos de equity, picos, drawdowns e timestamps
        """
        if entity_id not in self._equity_history:
            return {
                "equity": [],
                "peaks": [],
                "drawdowns": [],
                "timestamps": []
            }
        
        return {
            "equity": self._equity_history[entity_id],
            "peaks": self._peak_history[entity_id],
            "drawdowns": self._drawdown_history[entity_id],
            "timestamps": self._timestamp_history[entity_id]
        }
    
    def get_max_drawdown_period(self, entity_id: str) -> Tuple[float, datetime, datetime, int]:
        """
        Identifica o período de máximo drawdown.
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            
        Returns:
            Tupla com (max_drawdown, peak_time, valley_time, duration_days)
        """
        if entity_id not in self._equity_history or len(self._equity_history[entity_id]) < 2:
            return 0.0, None, None, 0
        
        equity = np.array(self._equity_history[entity_id])
        timestamps = self._timestamp_history[entity_id]
        
        # Calcular drawdowns
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks * 100
        
        # Encontrar o índice do drawdown máximo
        max_dd_idx = np.argmax(drawdowns)
        max_drawdown = drawdowns[max_dd_idx]
        
        # Encontrar o índice do pico antes do drawdown máximo
        peak_idx = np.where(equity == peaks[max_dd_idx])[0][-1]
        
        # Calcular a duração em dias
        if peak_idx < len(timestamps) and max_dd_idx < len(timestamps):
            peak_time = timestamps[peak_idx]
            valley_time = timestamps[max_dd_idx]
            duration_days = (valley_time - peak_time).days
        else:
            peak_time = None
            valley_time = None
            duration_days = 0
        
        return max_drawdown, peak_time, valley_time, duration_days
    
    def calculate_calmar_ratio(self, entity_id: str, annualized_return: float) -> float:
        """
        Calcula o Calmar Ratio (retorno anualizado / drawdown máximo).
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            annualized_return: Retorno anualizado do portfólio/estratégia
            
        Returns:
            Calmar Ratio
        """
        if entity_id not in self._drawdown_history:
            return 0.0
        
        max_drawdown = max(self._drawdown_history[entity_id]) if self._drawdown_history[entity_id] else 0
        
        if max_drawdown <= 0:
            return float('inf')  # Sem drawdown
        
        return annualized_return / max_drawdown
    
    def predict_recovery_time(self, entity_id: str, expected_return: float) -> float:
        """
        Estima o tempo de recuperação do drawdown atual baseado no retorno esperado.
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            expected_return: Retorno esperado anualizado (em percentual)
            
        Returns:
            Tempo estimado de recuperação em dias
        """
        if entity_id not in self._drawdown_history or not self._drawdown_history[entity_id]:
            return 0.0
        
        current_drawdown = self._drawdown_history[entity_id][-1]
        
        if current_drawdown <= 0 or expected_return <= 0:
            return 0.0
        
        # Fórmula: tempo = ln(1 / (1 - drawdown)) / ln(1 + retorno_diário)
        # Convertendo retorno anual para diário assumindo 252 dias úteis
        daily_return = (1 + expected_return / 100) ** (1 / 252) - 1
        
        # Drawdown em decimal (não percentual)
        drawdown_decimal = current_drawdown / 100
        
        # Calcular tempo estimado
        if drawdown_decimal >= 1:
            return float('inf')  # Drawdown de 100% ou mais
        
        recovery_time = np.log(1 / (1 - drawdown_decimal)) / np.log(1 + daily_return)
        
        return recovery_time
    
    def calculate_recovery_scenarios(
        self, 
        entity_id: str, 
        return_scenarios: List[float]
    ) -> Dict[float, float]:
        """
        Calcula múltiplos cenários de tempo de recuperação com diferentes retornos.
        
        Args:
            entity_id: Identificador do portfólio ou estratégia
            return_scenarios: Lista de cenários de retorno anualizado
            
        Returns:
            Dicionário com {retorno: tempo_recuperação} para cada cenário
        """
        scenarios = {}
        
        for expected_return in return_scenarios:
            recovery_time = self.predict_recovery_time(entity_id, expected_return)
            scenarios[expected_return] = recovery_time
        
        return scenarios