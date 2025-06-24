import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union

from risk_management.domain.entities.risk_metric import MetricType, RiskMetric
from risk_management.domain.entities.risk_profile import RiskProfile


@dataclass
class ExposureLimit:
    """Configuração de limite de exposição para um ativo ou categoria."""
    max_percentage: float  # Percentual máximo do capital total
    max_absolute: Optional[float] = None  # Valor absoluto máximo (opcional)
    warning_threshold_pct: float = 0.8  # Percentual do limite em que começa o alerta
    include_unrealized_pnl: bool = True  # Se deve incluir lucros/perdas não realizados
    cooldown_minutes: int = 30  # Tempo mínimo entre alertas repetidos


@dataclass
class ExposureInfo:
    """Informações sobre a exposição atual a um ativo ou categoria."""
    current_value: float  # Valor atual da exposição
    max_allowed: float  # Valor máximo permitido
    percentage_used: float  # Percentual do limite utilizado (0-100)
    warning_triggered: bool  # Se atingiu o nível de alerta
    critical_triggered: bool  # Se atingiu o nível crítico
    last_updated: datetime  # Data da última atualização
    metadata: Dict[str, str] = None  # Metadados adicionais


class ExposureService:
    """
    Serviço responsável por gerenciar e monitorar limites de exposição 
    a diferentes ativos, mercados e categorias.
    """
    
    def __init__(self, total_capital: float, logger: Optional[logging.Logger] = None):
        """
        Inicializa o serviço de gerenciamento de exposição.
        
        Args:
            total_capital: Capital total disponível para negociação
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._total_capital = total_capital
        self._asset_limits: Dict[str, ExposureLimit] = {}
        self._category_limits: Dict[str, ExposureLimit] = {}
        self._asset_exposures: Dict[str, float] = {}
        self._category_exposures: Dict[str, float] = {}
        self._asset_categories: Dict[str, Set[str]] = {}
        self._last_alerts: Dict[str, datetime] = {}
    
    def set_total_capital(self, amount: float) -> None:
        """
        Atualiza o valor do capital total disponível.
        
        Args:
            amount: Novo valor do capital total
        """
        if amount <= 0:
            raise ValueError("O capital total deve ser um valor positivo")
        
        old_capital = self._total_capital
        self._total_capital = amount
        self._logger.info(f"Capital total atualizado: {old_capital:.2f} -> {amount:.2f}")
    
    def get_total_capital(self) -> float:
        """
        Obtém o valor atual do capital total.
        
        Returns:
            Valor do capital total
        """
        return self._total_capital
    
    def set_asset_limit(self, asset_id: str, limit: ExposureLimit) -> None:
        """
        Define um limite de exposição para um ativo específico.
        
        Args:
            asset_id: Identificador do ativo
            limit: Configuração do limite de exposição
        """
        self._asset_limits[asset_id] = limit
        self._logger.info(
            f"Limite de exposição definido para ativo {asset_id}: "
            f"{limit.max_percentage:.2f}% (max: {limit.max_absolute or 'N/A'})"
        )
    
    def set_category_limit(self, category: str, limit: ExposureLimit) -> None:
        """
        Define um limite de exposição para uma categoria de ativos.
        
        Args:
            category: Nome da categoria
            limit: Configuração do limite de exposição
        """
        self._category_limits[category] = limit
        self._logger.info(
            f"Limite de exposição definido para categoria {category}: "
            f"{limit.max_percentage:.2f}% (max: {limit.max_absolute or 'N/A'})"
        )
    
    def register_asset_category(self, asset_id: str, categories: List[str]) -> None:
        """
        Registra as categorias a que um ativo pertence.
        
        Args:
            asset_id: Identificador do ativo
            categories: Lista de categorias
        """
        self._asset_categories[asset_id] = set(categories)
        self._logger.debug(f"Ativo {asset_id} registrado nas categorias: {', '.join(categories)}")
    
    def update_exposure(self, asset_id: str, amount: float) -> List[RiskMetric]:
        """
        Atualiza a exposição atual a um ativo e recalcula exposições de categorias.
        
        Args:
            asset_id: Identificador do ativo
            amount: Valor atual da exposição (em termos absolutos)
            
        Returns:
            Lista de métricas de risco geradas pela atualização
        """
        old_exposure = self._asset_exposures.get(asset_id, 0)
        self._asset_exposures[asset_id] = amount
        
        # Calcular a variação na exposição para ajustar as categorias
        exposure_delta = amount - old_exposure
        
        # Atualizar exposição das categorias
        if asset_id in self._asset_categories:
            for category in self._asset_categories[asset_id]:
                self._category_exposures[category] = self._category_exposures.get(category, 0) + exposure_delta
        
        # Gerar métricas de risco de exposição
        risk_metrics = []
        
        # Métrica para o ativo
        asset_metric = self._create_exposure_metric(asset_id, amount, is_asset=True)
        if asset_metric:
            risk_metrics.append(asset_metric)
        
        # Métricas para as categorias
        if asset_id in self._asset_categories:
            for category in self._asset_categories[asset_id]:
                category_exposure = self._category_exposures.get(category, 0)
                category_metric = self._create_exposure_metric(category, category_exposure, is_asset=False)
                if category_metric:
                    risk_metrics.append(category_metric)
        
        return risk_metrics
    
    def _create_exposure_metric(self, entity_id: str, amount: float, is_asset: bool) -> Optional[RiskMetric]:
        """
        Cria uma métrica de risco de exposição para um ativo ou categoria.
        
        Args:
            entity_id: Identificador do ativo ou categoria
            amount: Valor da exposição
            is_asset: True se for um ativo, False se for uma categoria
            
        Returns:
            Métrica de risco criada ou None
        """
        limits = self._asset_limits if is_asset else self._category_limits
        
        if entity_id not in limits:
            return None
        
        exposure_percentage = (amount / self._total_capital) * 100
        
        return RiskMetric.create(
            metric_type=MetricType.MAX_EXPOSURE,
            value=exposure_percentage,
            asset_id=entity_id if is_asset else None,
            metadata={
                "entity_type": "asset" if is_asset else "category",
                "entity_id": entity_id,
                "absolute_value": str(amount),
                "percentage": f"{exposure_percentage:.2f}%",
                "max_percentage": f"{limits[entity_id].max_percentage:.2f}%",
            }
        )
    
    def check_exposure_limits(self, profile: RiskProfile) -> List[Tuple[str, ExposureInfo, bool]]:
        """
        Verifica se algum limite de exposição foi violado.
        
        Args:
            profile: Perfil de risco para consultar limites adicionais
            
        Returns:
            Lista de tuplas (entity_id, exposure_info, is_critical) para limites violados
        """
        violations = []
        now = datetime.utcnow()
        
        # Verificar limites de ativos
        for asset_id, amount in self._asset_exposures.items():
            if asset_id not in self._asset_limits:
                continue
            
            limit = self._asset_limits[asset_id]
            max_allowed = min(
                self._total_capital * (limit.max_percentage / 100),
                limit.max_absolute or float('inf')
            )
            
            percentage_used = (amount / max_allowed) * 100 if max_allowed > 0 else float('inf')
            warning_triggered = percentage_used >= (limit.warning_threshold_pct * 100)
            critical_triggered = percentage_used >= 100
            
            if warning_triggered or critical_triggered:
                # Verificar cooldown para evitar alertas repetidos
                if asset_id in self._last_alerts:
                    minutes_since_last = (now - self._last_alerts[asset_id]).total_seconds() / 60
                    if minutes_since_last < limit.cooldown_minutes:
                        continue
                
                self._last_alerts[asset_id] = now
                
                info = ExposureInfo(
                    current_value=amount,
                    max_allowed=max_allowed,
                    percentage_used=percentage_used,
                    warning_triggered=warning_triggered,
                    critical_triggered=critical_triggered,
                    last_updated=now,
                    metadata={"type": "asset"}
                )
                
                violations.append((asset_id, info, critical_triggered))
        
        # Verificar limites de categorias
        for category, amount in self._category_exposures.items():
            if category not in self._category_limits:
                continue
            
            limit = self._category_limits[category]
            max_allowed = min(
                self._total_capital * (limit.max_percentage / 100),
                limit.max_absolute or float('inf')
            )
            
            percentage_used = (amount / max_allowed) * 100 if max_allowed > 0 else float('inf')
            warning_triggered = percentage_used >= (limit.warning_threshold_pct * 100)
            critical_triggered = percentage_used >= 100
            
            if warning_triggered or critical_triggered:
                # Verificar cooldown para evitar alertas repetidos
                if category in self._last_alerts:
                    minutes_since_last = (now - self._last_alerts[category]).total_seconds() / 60
                    if minutes_since_last < limit.cooldown_minutes:
                        continue
                
                self._last_alerts[category] = now
                
                info = ExposureInfo(
                    current_value=amount,
                    max_allowed=max_allowed,
                    percentage_used=percentage_used,
                    warning_triggered=warning_triggered,
                    critical_triggered=critical_triggered,
                    last_updated=now,
                    metadata={"type": "category"}
                )
                
                violations.append((category, info, critical_triggered))
        
        return violations
    
    def get_exposure_metrics(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """
        Obtém métricas de exposição atuais para todos os ativos e categorias.
        
        Returns:
            Dicionário com métricas de exposição
        """
        result = {
            "total_capital": self._total_capital,
            "assets": {},
            "categories": {},
            "summary": {
                "total_exposure": sum(self._asset_exposures.values()),
                "exposure_percentage": (sum(self._asset_exposures.values()) / self._total_capital) * 100 
                                       if self._total_capital > 0 else 0,
                "highest_asset_exposure": max(self._asset_exposures.values()) if self._asset_exposures else 0,
                "highest_category_exposure": max(self._category_exposures.values()) if self._category_exposures else 0,
            }
        }
        
        # Métricas de ativos
        for asset_id, amount in self._asset_exposures.items():
            percentage = (amount / self._total_capital) * 100 if self._total_capital > 0 else 0
            limit_info = {}
            
            if asset_id in self._asset_limits:
                limit = self._asset_limits[asset_id]
                max_allowed = min(
                    self._total_capital * (limit.max_percentage / 100),
                    limit.max_absolute or float('inf')
                )
                limit_info = {
                    "max_allowed": max_allowed,
                    "max_percentage": limit.max_percentage,
                    "percentage_used": (amount / max_allowed) * 100 if max_allowed > 0 else float('inf'),
                    "status": "critical" if amount > max_allowed else 
                             "warning" if amount > (max_allowed * limit.warning_threshold_pct) else 
                             "normal"
                }
            
            result["assets"][asset_id] = {
                "amount": amount,
                "percentage": percentage,
                "categories": list(self._asset_categories.get(asset_id, set())),
                **limit_info
            }
        
        # Métricas de categorias
        for category, amount in self._category_exposures.items():
            percentage = (amount / self._total_capital) * 100 if self._total_capital > 0 else 0
            limit_info = {}
            
            if category in self._category_limits:
                limit = self._category_limits[category]
                max_allowed = min(
                    self._total_capital * (limit.max_percentage / 100),
                    limit.max_absolute or float('inf')
                )
                limit_info = {
                    "max_allowed": max_allowed,
                    "max_percentage": limit.max_percentage,
                    "percentage_used": (amount / max_allowed) * 100 if max_allowed > 0 else float('inf'),
                    "status": "critical" if amount > max_allowed else 
                             "warning" if amount > (max_allowed * limit.warning_threshold_pct) else 
                             "normal"
                }
            
            result["categories"][category] = {
                "amount": amount,
                "percentage": percentage,
                **limit_info
            }
        
        return result
    
    def can_increase_position(self, asset_id: str, amount: float) -> Tuple[bool, str]:
        """
        Verifica se é possível aumentar a posição em um ativo sem violar limites.
        
        Args:
            asset_id: Identificador do ativo
            amount: Valor adicional de exposição
            
        Returns:
            Tupla (pode_aumentar, motivo)
        """
        # Verificar se o ativo tem limite definido
        if asset_id not in self._asset_limits:
            return True, ""
        
        current_exposure = self._asset_exposures.get(asset_id, 0)
        new_exposure = current_exposure + amount
        
        limit = self._asset_limits[asset_id]
        max_allowed = min(
            self._total_capital * (limit.max_percentage / 100),
            limit.max_absolute or float('inf')
        )
        
        if new_exposure > max_allowed:
            return False, (
                f"Aumento excederia o limite máximo de exposição para {asset_id}: "
                f"{new_exposure:.2f} > {max_allowed:.2f}"
            )
        
        # Verificar limites de categoria
        if asset_id in self._asset_categories:
            for category in self._asset_categories[asset_id]:
                if category not in self._category_limits:
                    continue
                
                current_cat_exposure = self._category_exposures.get(category, 0)
                new_cat_exposure = current_cat_exposure + amount
                
                cat_limit = self._category_limits[category]
                cat_max_allowed = min(
                    self._total_capital * (cat_limit.max_percentage / 100),
                    cat_limit.max_absolute or float('inf')
                )
                
                if new_cat_exposure > cat_max_allowed:
                    return False, (
                        f"Aumento excederia o limite máximo de exposição para categoria {category}: "
                        f"{new_cat_exposure:.2f} > {cat_max_allowed:.2f}"
                    )
        
        return True, ""