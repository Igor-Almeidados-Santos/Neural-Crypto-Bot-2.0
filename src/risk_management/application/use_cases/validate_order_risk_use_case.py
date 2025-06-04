from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from src.risk_management.application.services.circuit_breaker_service import CircuitBreakerService
from src.risk_management.application.services.exposure_service import ExposureService
from src.risk_management.domain.entities.risk_profile import RiskProfile
from src.risk_management.infrastructure.risk_repository import RiskRepository


@dataclass
class OrderRiskParams:
    """Parâmetros para validação de risco de uma ordem."""
    asset_id: str
    quantity: float
    price: float
    order_type: str
    side: str  # "buy" ou "sell"
    strategy_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    leverage: float = 1.0
    is_reduce_only: bool = False
    metadata: Optional[Dict[str, str]] = None


@dataclass
class OrderRiskValidationResult:
    """Resultado da validação de risco de uma ordem."""
    is_valid: bool
    validation_id: str
    reasons: List[str]
    warnings: List[str]
    metrics: Dict[str, float]


class ValidateOrderRiskUseCase:
    """
    Caso de uso para validar se uma ordem está dentro dos limites de risco.
    """
    
    def __init__(
        self,
        risk_repository: RiskRepository,
        exposure_service: ExposureService,
        circuit_breaker_service: CircuitBreakerService,
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            risk_repository: Repositório de perfis de risco
            exposure_service: Serviço de gerenciamento de exposição
            circuit_breaker_service: Serviço de circuit breaker
        """
        self._risk_repository = risk_repository
        self._exposure_service = exposure_service
        self._circuit_breaker_service = circuit_breaker_service
    
    def execute(self, params: OrderRiskParams) -> OrderRiskValidationResult:
        """
        Executa a validação de risco para uma ordem.
        
        Args:
            params: Parâmetros da ordem a ser validada
            
        Returns:
            Resultado da validação de risco
        """
        reasons = []
        warnings = []
        metrics = {}
        
        # Passo 1: Verificar se há circuit breakers ativos
        entity_ids = [
            params.asset_id,
            params.strategy_id,
            params.portfolio_id
        ]
        
        for entity_id in entity_ids:
            if entity_id and self._circuit_breaker_service.is_open(entity_id):
                reasons.append(f"Circuit breaker ativo para {entity_id}")
        
        # Passo 2: Verificar se a ordem está reduzindo posição (less risk)
        is_reducing_risk = params.is_reduce_only
        
        # Passo 3: Carregar perfis de risco aplicáveis
        profiles = self._risk_repository.get_applicable_profiles(
            portfolio_id=params.portfolio_id,
            strategy_id=params.strategy_id,
            asset_id=params.asset_id
        )
        
        if not profiles:
            warnings.append("Nenhum perfil de risco aplicável encontrado")
        
        # Passo 4: Verificar limites de alavancagem
        if params.leverage > 1.0:
            for profile in profiles:
                if params.leverage > profile.max_leverage:
                    reasons.append(
                        f"Alavancagem de {params.leverage}x excede o limite de {profile.max_leverage}x "
                        f"definido no perfil '{profile.name}'"
                    )
                    break
        
        # Passo 5: Verificar limites de exposição se a ordem aumentar a posição
        if not is_reducing_risk and params.side == "buy":
            order_value = params.quantity * params.price
            
            # Verificar se pode aumentar exposição
            can_increase, exposure_reason = self._exposure_service.can_increase_position(
                params.asset_id, order_value
            )
            
            if not can_increase:
                reasons.append(exposure_reason)
            
            # Verificar tamanho máximo de posição em perfis
            for profile in profiles:
                total_capital = self._exposure_service.get_total_capital()
                max_position_value = total_capital * (profile.max_position_size_pct / 100)
                
                # Estimar novo valor da posição
                current_exposure = self._exposure_service._asset_exposures.get(params.asset_id, 0)
                new_exposure = current_exposure + order_value
                
                if new_exposure > max_position_value:
                    reasons.append(
                        f"Novo valor de posição ({new_exposure:.2f}) excederia o limite máximo "
                        f"({max_position_value:.2f}) definido no perfil '{profile.name}'"
                    )
        
        # Passo 6: Verificar qualquer regra adicional específica para o ativo
        asset_specific_rules = self._risk_repository.get_asset_specific_rules(params.asset_id)
        for rule in asset_specific_rules:
            # Implementar lógica de validação para regras específicas do ativo
            pass
        
        # Passo 7: Coletar métricas relevantes para o resultado
        metrics["order_value"] = params.quantity * params.price
        metrics["leverage"] = params.leverage
        
        # Se estamos aumentando posição, estimar o novo valor
        if not is_reducing_risk and params.side == "buy":
            current_exposure = self._exposure_service._asset_exposures.get(params.asset_id, 0)
            new_exposure = current_exposure + (params.quantity * params.price)
            metrics["current_exposure"] = current_exposure
            metrics["new_exposure"] = new_exposure
            
            # Calcular percentual do capital
            total_capital = self._exposure_service.get_total_capital()
            if total_capital > 0:
                metrics["current_exposure_pct"] = (current_exposure / total_capital) * 100
                metrics["new_exposure_pct"] = (new_exposure / total_capital) * 100
        
        # Gerar resultado final
        is_valid = len(reasons) == 0
        
        return OrderRiskValidationResult(
            is_valid=is_valid,
            validation_id=f"val_{params.asset_id}_{params.order_type}_{params.side}",
            reasons=reasons,
            warnings=warnings,
            metrics=metrics
        )