import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Table, Text, create_engine, func, select)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from src.risk_management.application.use_cases.generate_risk_report_use_case import RiskReport
from src.risk_management.domain.entities.risk_metric import MetricType, RiskMetric, TimeFrame
from src.risk_management.domain.entities.risk_profile import RiskLevel, RiskProfile
from src.risk_management.domain.events.risk_threshold_breached_event import RiskThresholdBreachedEvent


Base = declarative_base()


class RiskProfileModel(Base):
    """Modelo SQLAlchemy para perfis de risco."""
    __tablename__ = "risk_profiles"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    risk_level = Column(String, nullable=False)
    max_drawdown_pct = Column(Float, nullable=False)
    max_position_size_pct = Column(Float, nullable=False)
    max_leverage = Column(Float, nullable=False)
    var_confidence_level = Column(Float, nullable=False)
    thresholds = Column(JSONB)
    correlation_limits = Column(JSONB)
    position_limits = Column(JSONB)
    risk_categories = Column(JSONB)
    metadata = Column(JSONB)
    enabled = Column(Integer, nullable=False, default=1)
    portfolio_id = Column(String)
    strategy_id = Column(String)
    asset_id = Column(String)


class RiskMetricModel(Base):
    """Modelo SQLAlchemy para métricas de risco."""
    __tablename__ = "risk_metrics"
    
    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    asset_id = Column(String)
    strategy_id = Column(String)
    portfolio_id = Column(String)
    timeframe = Column(String)
    confidence_level = Column(Float)
    metadata = Column(JSONB)


class RiskThresholdBreachModel(Base):
    """Modelo SQLAlchemy para violações de limites de risco."""
    __tablename__ = "risk_threshold_breaches"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    metric_id = Column(String, nullable=False)
    profile_id = Column(String, nullable=False)
    threshold_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    is_critical = Column(Integer, nullable=False)
    portfolio_id = Column(String)
    strategy_id = Column(String)
    asset_id = Column(String)
    metadata = Column(JSONB)


class RiskReportModel(Base):
    """Modelo SQLAlchemy para relatórios de risco."""
    __tablename__ = "risk_reports"
    
    id = Column(String, primary_key=True)
    portfolio_id = Column(String, nullable=False)
    generation_time = Column(DateTime, nullable=False)
    timeframe = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    report_data = Column(JSONB, nullable=False)


class AssetRuleModel(Base):
    """Modelo SQLAlchemy para regras específicas de ativos."""
    __tablename__ = "asset_rules"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(String, nullable=False)
    rule_type = Column(String, nullable=False)
    rule_value = Column(Float)
    rule_params = Column(JSONB)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    enabled = Column(Integer, nullable=False, default=1)


class RiskRepository:
    """
    Repositório para persistência e recuperação de dados de risco.
    """
    
    def __init__(
        self,
        connection_string: str,
        create_tables: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o repositório de risco.
        
        Args:
            connection_string: String de conexão SQLAlchemy
            create_tables: Se deve criar as tabelas no banco de dados
            logger: Logger opcional para registro de eventos
        """
        self._logger = logger or logging.getLogger(__name__)
        self._engine = create_engine(connection_string)
        self._Session = sessionmaker(bind=self._engine)
        
        if create_tables:
            Base.metadata.create_all(self._engine)
    
    def save_risk_profile(self, profile: RiskProfile) -> None:
        """
        Salva um perfil de risco no banco de dados.
        
        Args:
            profile: Perfil de risco a ser salvo
        """
        session = self._Session()
        try:
            # Converter thresholds para formato adequado ao banco
            thresholds_dict = {}
            for metric_type, config in profile.thresholds.items():
                thresholds_dict[metric_type.name] = {
                    "warning_threshold": config.warning_threshold,
                    "critical_threshold": config.critical_threshold,
                    "is_upper_bound": config.is_upper_bound
                }
            
            # Criar modelo para persistência
            db_profile = RiskProfileModel(
                id=str(profile.id),
                name=profile.name,
                description=profile.description,
                created_at=profile.created_at,
                updated_at=profile.updated_at,
                risk_level=profile.risk_level.value,
                max_drawdown_pct=profile.max_drawdown_pct,
                max_position_size_pct=profile.max_position_size_pct,
                max_leverage=profile.max_leverage,
                var_confidence_level=profile.var_confidence_level,
                thresholds=thresholds_dict,
                correlation_limits=profile.correlation_limits,
                position_limits=profile.position_limits,
                risk_categories=[r.value for r in profile.risk_categories],
                metadata=profile.metadata,
                enabled=1 if profile.enabled else 0,
                portfolio_id=profile.portfolio_id,
                strategy_id=profile.strategy_id,
                asset_id=profile.asset_id
            )
            
            # Verificar se já existe
            existing = session.query(RiskProfileModel).filter_by(id=str(profile.id)).first()
            if existing:
                # Atualizar campos
                existing.name = db_profile.name
                existing.description = db_profile.description
                existing.updated_at = db_profile.updated_at
                existing.risk_level = db_profile.risk_level
                existing.max_drawdown_pct = db_profile.max_drawdown_pct
                existing.max_position_size_pct = db_profile.max_position_size_pct
                existing.max_leverage = db_profile.max_leverage
                existing.var_confidence_level = db_profile.var_confidence_level
                existing.thresholds = db_profile.thresholds
                existing.correlation_limits = db_profile.correlation_limits
                existing.position_limits = db_profile.position_limits
                existing.risk_categories = db_profile.risk_categories
                existing.metadata = db_profile.metadata
                existing.enabled = db_profile.enabled
                existing.portfolio_id = db_profile.portfolio_id
                existing.strategy_id = db_profile.strategy_id
                existing.asset_id = db_profile.asset_id
            else:
                # Inserir novo
                session.add(db_profile)
            
            session.commit()
            self._logger.info(f"Perfil de risco salvo: {profile.name} ({profile.id})")
        except Exception as e:
            session.rollback()
            self._logger.error(f"Erro ao salvar perfil de risco: {e}")
            raise
        finally:
            session.close()
    
    def get_risk_profile(self, profile_id: Union[str, UUID]) -> Optional[RiskProfile]:
        """
        Recupera um perfil de risco pelo ID.
        
        Args:
            profile_id: ID do perfil de risco
            
        Returns:
            Perfil de risco ou None se não encontrado
        """
        session = self._Session()
        try:
            profile_model = session.query(RiskProfileModel).filter_by(id=str(profile_id)).first()
            
            if not profile_model:
                return None
            
            # Converter dados do banco para entidade de domínio
            return self._model_to_risk_profile(profile_model)
        except Exception as e:
            self._logger.error(f"Erro ao recuperar perfil de risco: {e}")
            return None
        finally:
            session.close()
    
    def get_risk_profile_by_portfolio(self, portfolio_id: str) -> Optional[RiskProfile]:
        """
        Recupera um perfil de risco para um portfólio específico.
        
        Args:
            portfolio_id: ID do portfólio
            
        Returns:
            Perfil de risco ou None se não encontrado
        """
        session = self._Session()
        try:
            profile_model = session.query(RiskProfileModel).filter_by(
                portfolio_id=portfolio_id, enabled=1
            ).first()
            
            if not profile_model:
                return None
            
            return self._model_to_risk_profile(profile_model)
        except Exception as e:
            self._logger.error(f"Erro ao recuperar perfil de risco para portfólio {portfolio_id}: {e}")
            return None
        finally:
            session.close()
    
    def get_applicable_profiles(
        self,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> List[RiskProfile]:
        """
        Recupera todos os perfis de risco aplicáveis a uma combinação de entidades.
        
        Args:
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            Lista de perfis de risco aplicáveis
        """
        session = self._Session()
        try:
            query = session.query(RiskProfileModel).filter_by(enabled=1)
            
            # Buscar perfis globais (sem IDs específicos)
            global_profiles = query.filter(
                RiskProfileModel.portfolio_id.is_(None),
                RiskProfileModel.strategy_id.is_(None),
                RiskProfileModel.asset_id.is_(None)
            ).all()
            
            result_profiles = [self._model_to_risk_profile(p) for p in global_profiles]
            
            # Buscar perfis específicos para cada entidade fornecida
            if portfolio_id:
                portfolio_profiles = query.filter_by(portfolio_id=portfolio_id).all()
                result_profiles.extend([self._model_to_risk_profile(p) for p in portfolio_profiles])
            
            if strategy_id:
                strategy_profiles = query.filter_by(strategy_id=strategy_id).all()
                result_profiles.extend([self._model_to_risk_profile(p) for p in strategy_profiles])
            
            if asset_id:
                asset_profiles = query.filter_by(asset_id=asset_id).all()
                result_profiles.extend([self._model_to_risk_profile(p) for p in asset_profiles])
            
            return result_profiles
        except Exception as e:
            self._logger.error(f"Erro ao recuperar perfis de risco aplicáveis: {e}")
            return []
        finally:
            session.close()
    
    def save_metric(self, metric: RiskMetric) -> None:
        """
        Salva uma métrica de risco no banco de dados.
        
        Args:
            metric: Métrica de risco a ser salva
        """
        session = self._Session()
        try:
            # Criar modelo para persistência
            db_metric = RiskMetricModel(
                id=str(metric.id),
                type=metric.type.name,
                value=metric.value,
                timestamp=metric.timestamp,
                asset_id=metric.asset_id,
                strategy_id=metric.strategy_id,
                portfolio_id=metric.portfolio_id,
                timeframe=metric.timeframe.value if metric.timeframe else None,
                confidence_level=metric.confidence_level,
                metadata=metric.metadata or {}
            )
            
            session.add(db_metric)
            session.commit()
        except Exception as e:
            session.rollback()
            self._logger.error(f"Erro ao salvar métrica de risco: {e}")
            raise
        finally:
            session.close()
    
    def save_metrics_batch(self, metrics: List[RiskMetric]) -> None:
        """
        Salva um lote de métricas de risco de uma vez.
        
        Args:
            metrics: Lista de métricas a serem salvas
        """
        if not metrics:
            return
        
        session = self._Session()
        try:
            db_metrics = []
            for metric in metrics:
                db_metric = RiskMetricModel(
                    id=str(metric.id),
                    type=metric.type.name,
                    value=metric.value,
                    timestamp=metric.timestamp,
                    asset_id=metric.asset_id,
                    strategy_id=metric.strategy_id,
                    portfolio_id=metric.portfolio_id,
                    timeframe=metric.timeframe.value if metric.timeframe else None,
                    confidence_level=metric.confidence_level,
                    metadata=metric.metadata or {}
                )
                db_metrics.append(db_metric)
            
            session.add_all(db_metrics)
            session.commit()
            self._logger.info(f"Lote de {len(metrics)} métricas salvo com sucesso")
        except Exception as e:
            session.rollback()
            self._logger.error(f"Erro ao salvar lote de métricas: {e}")
            raise
        finally:
            session.close()
    
    def get_latest_metric(
        self,
        metric_type: MetricType,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> Optional[RiskMetric]:
        """
        Recupera a métrica mais recente de um tipo específico.
        
        Args:
            metric_type: Tipo da métrica
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            Métrica mais recente ou None se não encontrada
        """
        session = self._Session()
        try:
            query = session.query(RiskMetricModel).filter_by(type=metric_type.name)
            
            if portfolio_id:
                query = query.filter_by(portfolio_id=portfolio_id)
            if strategy_id:
                query = query.filter_by(strategy_id=strategy_id)
            if asset_id:
                query = query.filter_by(asset_id=asset_id)
            
            metric_model = query.order_by(RiskMetricModel.timestamp.desc()).first()
            
            if not metric_model:
                return None
            
            return self._model_to_risk_metric(metric_model)
        except Exception as e:
            self._logger.error(f"Erro ao recuperar última métrica: {e}")
            return None
        finally:
            session.close()
    
    def get_metrics_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        metric_types: Optional[List[MetricType]] = None,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> List[RiskMetric]:
        """
        Recupera métricas de risco dentro de um intervalo de datas.
        
        Args:
            start_date: Data inicial
            end_date: Data final
            metric_types: Lista de tipos de métrica (opcional)
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            Lista de métricas de risco
        """
        session = self._Session()
        try:
            query = session.query(RiskMetricModel).filter(
                RiskMetricModel.timestamp >= start_date,
                RiskMetricModel.timestamp <= end_date
            )
            
            if metric_types:
                query = query.filter(RiskMetricModel.type.in_([m.name for m in metric_types]))
            
            if portfolio_id:
                query = query.filter_by(portfolio_id=portfolio_id)
            if strategy_id:
                query = query.filter_by(strategy_id=strategy_id)
            if asset_id:
                query = query.filter_by(asset_id=asset_id)
            
            metric_models = query.order_by(RiskMetricModel.timestamp).all()
            
            return [self._model_to_risk_metric(m) for m in metric_models]
        except Exception as e:
            self._logger.error(f"Erro ao recuperar métricas por intervalo de datas: {e}")
            return []
        finally:
            session.close()
    
    def save_threshold_breach(self, event: RiskThresholdBreachedEvent) -> None:
        """
        Salva um evento de violação de limite de risco.
        
        Args:
            event: Evento de violação a ser salvo
        """
        session = self._Session()
        try:
            # Criar modelo para persistência
            db_breach = RiskThresholdBreachModel(
                id=str(event.id),
                timestamp=event.timestamp,
                metric_id=str(event.metric.id),
                profile_id=str(event.profile.id),
                threshold_value=event.threshold_value,
                actual_value=event.actual_value,
                risk_level=event.risk_level.value,
                is_critical=1 if event.is_critical else 0,
                portfolio_id=event.portfolio_id,
                strategy_id=event.strategy_id,
                asset_id=event.asset_id,
                metadata=event.metadata or {}
            )
            
            session.add(db_breach)
            session.commit()
            self._logger.info(f"Violação de limite salva: {event.get_message()}")
        except Exception as e:
            session.rollback()
            self._logger.error(f"Erro ao salvar violação de limite: {e}")
            raise
        finally:
            session.close()
    
    def get_threshold_breaches(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        is_critical: Optional[bool] = None,
        limit: int = 100
    ) -> List[RiskThresholdBreachedEvent]:
        """
        Recupera eventos de violação de limite de risco.
        
        Args:
            start_date: Data inicial (opcional)
            end_date: Data final (opcional)
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            is_critical: Filtrar apenas violações críticas (opcional)
            limit: Limite de resultados
            
        Returns:
            Lista de eventos de violação
        """
        session = self._Session()
        try:
            query = session.query(RiskThresholdBreachModel)
            
            if start_date:
                query = query.filter(RiskThresholdBreachModel.timestamp >= start_date)
            if end_date:
                query = query.filter(RiskThresholdBreachModel.timestamp <= end_date)
            if portfolio_id:
                query = query.filter_by(portfolio_id=portfolio_id)
            if strategy_id:
                query = query.filter_by(strategy_id=strategy_id)
            if asset_id:
                query = query.filter_by(asset_id=asset_id)
            if is_critical is not None:
                query = query.filter_by(is_critical=1 if is_critical else 0)
            
            breach_models = query.order_by(RiskThresholdBreachModel.timestamp.desc()).limit(limit).all()
            
            # Converter para eventos de domínio
            # Nota: Esta implementação simplificada carrega os objetos relacionados separadamente.
            # Uma implementação mais eficiente usaria joins.
            events = []
            for breach in breach_models:
                try:
                    metric = self.get_metric_by_id(breach.metric_id)
                    profile = self.get_risk_profile(breach.profile_id)
                    
                    if metric and profile:
                        event = RiskThresholdBreachedEvent(
                            id=UUID(breach.id),
                            timestamp=breach.timestamp,
                            metric=metric,
                            profile=profile,
                            threshold_value=breach.threshold_value,
                            actual_value=breach.actual_value,
                            risk_level=RiskLevel(breach.risk_level),
                            is_critical=bool(breach.is_critical),
                            portfolio_id=breach.portfolio_id,
                            strategy_id=breach.strategy_id,
                            asset_id=breach.asset_id,
                            metadata=breach.metadata
                        )
                        events.append(event)
                except Exception as e:
                    self._logger.error(f"Erro ao converter violação {breach.id}: {e}")
            
            return events
        except Exception as e:
            self._logger.error(f"Erro ao recuperar violações de limite: {e}")
            return []
        finally:
            session.close()
    
    def get_metric_by_id(self, metric_id: Union[str, UUID]) -> Optional[RiskMetric]:
        """
        Recupera uma métrica de risco pelo ID.
        
        Args:
            metric_id: ID da métrica
            
        Returns:
            Métrica de risco ou None se não encontrada
        """
        session = self._Session()
        try:
            metric_model = session.query(RiskMetricModel).filter_by(id=str(metric_id)).first()
            
            if not metric_model:
                return None
            
            return self._model_to_risk_metric(metric_model)
        except Exception as e:
            self._logger.error(f"Erro ao recuperar métrica por ID: {e}")
            return None
        finally:
            session.close()
    
    def save_risk_report(self, report: RiskReport) -> None:
        """
        Salva um relatório de risco no banco de dados.
        
        Args:
            report: Relatório de risco a ser salvo
        """
        session = self._Session()
        try:
            # Converter dados do relatório para JSON
            report_data = {
                "portfolio_stats": self._portfolio_stats_to_dict(report.portfolio_stats),
                "metrics_summary": [self._metric_summary_to_dict(m) for m in report.metrics_summary],
                "risk_breaches": report.risk_breaches,
                "circuit_breaker_events": report.circuit_breaker_events,
                "exposure_breakdown": report.exposure_breakdown,
                "asset_risk_contribution": report.asset_risk_contribution,
                "correlation_matrix": report.correlation_matrix,
                "recommendations": report.recommendations
            }
            
            # Criar modelo para persistência
            db_report = RiskReportModel(
                id=report.report_id,
                portfolio_id=report.portfolio_id,
                generation_time=report.generation_time,
                timeframe=report.timeframe.value,
                start_date=report.start_date,
                end_date=report.end_date,
                report_data=report_data
            )
            
            session.add(db_report)
            session.commit()
            self._logger.info(f"Relatório de risco salvo: {report.report_id}")
        except Exception as e:
            session.rollback()
            self._logger.error(f"Erro ao salvar relatório de risco: {e}")
            raise
        finally:
            session.close()
    
    def get_asset_specific_rules(self, asset_id: str) -> List[Dict]:
        """
        Recupera regras específicas para um ativo.
        
        Args:
            asset_id: ID do ativo
            
        Returns:
            Lista de regras específicas
        """
        session = self._Session()
        try:
            rules = session.query(AssetRuleModel).filter_by(
                asset_id=asset_id, enabled=1
            ).all()
            
            return [
                {
                    "id": rule.id,
                    "asset_id": rule.asset_id,
                    "rule_type": rule.rule_type,
                    "rule_value": rule.rule_value,
                    "rule_params": rule.rule_params,
                    "created_at": rule.created_at,
                    "updated_at": rule.updated_at
                }
                for rule in rules
            ]
        except Exception as e:
            self._logger.error(f"Erro ao recuperar regras específicas para ativo {asset_id}: {e}")
            return []
        finally:
            session.close()
    
    def _model_to_risk_profile(self, model: RiskProfileModel) -> RiskProfile:
        """
        Converte um modelo de banco de dados para entidade de domínio RiskProfile.
        
        Args:
            model: Modelo de banco de dados
            
        Returns:
            Entidade de domínio RiskProfile
        """
        # Converter thresholds de volta para o formato de domínio
        thresholds = {}
        for metric_name, config in model.thresholds.items():
            try:
                metric_type = MetricType[metric_name]
                thresholds[metric_type] = RiskProfile.ThresholdConfig(
                    metric_type=metric_type,
                    warning_threshold=config["warning_threshold"],
                    critical_threshold=config["critical_threshold"],
                    is_upper_bound=config.get("is_upper_bound", True)
                )
            except (KeyError, ValueError) as e:
                self._logger.warning(f"Erro ao converter threshold {metric_name}: {e}")
        
        # Converter categorias de risco
        risk_categories = []
        for category_name in model.risk_categories:
            try:
                risk_categories.append(RiskProfile.RiskCategory(category_name))
            except ValueError:
                self._logger.warning(f"Categoria de risco desconhecida: {category_name}")
        
        return RiskProfile(
            id=UUID(model.id),
            name=model.name,
            description=model.description,
            created_at=model.created_at,
            updated_at=model.updated_at,
            risk_level=RiskLevel(model.risk_level),
            max_drawdown_pct=model.max_drawdown_pct,
            max_position_size_pct=model.max_position_size_pct,
            max_leverage=model.max_leverage,
            var_confidence_level=model.var_confidence_level,
            thresholds=thresholds,
            correlation_limits=model.correlation_limits,
            position_limits=model.position_limits,
            risk_categories=risk_categories,
            metadata=model.metadata,
            enabled=bool(model.enabled),
            portfolio_id=model.portfolio_id,
            strategy_id=model.strategy_id,
            asset_id=model.asset_id
        )
    
    def _model_to_risk_metric(self, model: RiskMetricModel) -> RiskMetric:
        """
        Converte um modelo de banco de dados para entidade de domínio RiskMetric.
        
        Args:
            model: Modelo de banco de dados
            
        Returns:
            Entidade de domínio RiskMetric
        """
        try:
            metric_type = MetricType[model.type]
        except KeyError:
            self._logger.warning(f"Tipo de métrica desconhecido: {model.type}")
            metric_type = MetricType.CUSTOM
        
        timeframe = None
        if model.timeframe:
            try:
                timeframe = TimeFrame(model.timeframe)
            except ValueError:
                self._logger.warning(f"Timeframe desconhecido: {model.timeframe}")
        
        return RiskMetric(
            id=UUID(model.id),
            type=metric_type,
            value=model.value,
            timestamp=model.timestamp,
            asset_id=model.asset_id,
            strategy_id=model.strategy_id,
            portfolio_id=model.portfolio_id,
            timeframe=timeframe,
            confidence_level=model.confidence_level,
            metadata=model.metadata
        )
    
    def _portfolio_stats_to_dict(self, stats) -> Dict:
        """
        Converte PortfolioStats para dicionário.
        
        Args:
            stats: Objeto PortfolioStats
            
        Returns:
            Dicionário com dados do PortfolioStats
        """
        return {
            "total_value": stats.total_value,
            "unrealized_pnl": stats.unrealized_pnl,
            "unrealized_pnl_pct": stats.unrealized_pnl_pct,
            "num_positions": stats.num_positions,
            "max_position_value": stats.max_position_value,
            "max_position_weight": stats.max_position_weight,
            "max_position_asset": stats.max_position_asset,
            "concentration_index": stats.concentration_index,
            "correlation_avg": stats.correlation_avg,
            "volatility_daily": stats.volatility_daily,
            "sharpe_ratio": stats.sharpe_ratio,
            "sortino_ratio": stats.sortino_ratio,
            "var_95": stats.var_95,
            "var_99": stats.var_99,
            "expected_shortfall_95": stats.expected_shortfall_95,
            "max_drawdown": stats.max_drawdown,
            "drawdown_current": stats.drawdown_current,
            "drawdown_period": stats.drawdown_period,
            "timestamp": stats.timestamp.isoformat()
        }
    
    def _metric_summary_to_dict(self, summary) -> Dict:
        """
        Converte MetricSummary para dicionário.
        
        Args:
            summary: Objeto MetricSummary
            
        Returns:
            Dicionário com dados do MetricSummary
        """
        return {
            "name": summary.name,
            "current_value": summary.current_value,
            "average_value": summary.average_value,
            "min_value": summary.min_value,
            "max_value": summary.max_value,
            "trend": summary.trend,
            "status": summary.status,
            "threshold": summary.threshold
        }
    
    def get_risk_metrics_aggregation(
        self,
        metric_type: MetricType,
        period: str,  # 'day', 'week', 'month'
        start_date: datetime,
        end_date: datetime,
        portfolio_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        asset_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Recupera métricas de risco agregadas por período.
        
        Args:
            metric_type: Tipo da métrica
            period: Período de agregação ('day', 'week', 'month')
            start_date: Data inicial
            end_date: Data final
            portfolio_id: ID do portfólio (opcional)
            strategy_id: ID da estratégia (opcional)
            asset_id: ID do ativo (opcional)
            
        Returns:
            DataFrame pandas com métricas agregadas
        """
        session = self._Session()
        try:
            # Definir formato de data para agrupamento
            if period == 'day':
                date_format = '%Y-%m-%d'
            elif period == 'week':
                date_format = '%Y-%U'  # Ano-Semana
            elif period == 'month':
                date_format = '%Y-%m'
            else:
                raise ValueError(f"Período de agregação inválido: {period}")
            
            # Construir query base
            query = session.query(
                func.strftime(date_format, RiskMetricModel.timestamp).label('period'),
                func.avg(RiskMetricModel.value).label('avg_value'),
                func.min(RiskMetricModel.value).label('min_value'),
                func.max(RiskMetricModel.value).label('max_value'),
                func.count(RiskMetricModel.id).label('count')
            ).filter(
                RiskMetricModel.type == metric_type.name,
                RiskMetricModel.timestamp >= start_date,
                RiskMetricModel.timestamp <= end_date
            )
            
            # Adicionar filtros opcionais
            if portfolio_id:
                query = query.filter(RiskMetricModel.portfolio_id == portfolio_id)
            if strategy_id:
                query = query.filter(RiskMetricModel.strategy_id == strategy_id)
            if asset_id:
                query = query.filter(RiskMetricModel.asset_id == asset_id)
            
            # Agrupar e ordenar
            result = query.group_by('period').order_by('period').all()
            
            # Converter para DataFrame
            df = pd.DataFrame([
                {'period': r.period, 'avg_value': r.avg_value, 'min_value': r.min_value, 
                'max_value': r.max_value, 'count': r.count}
                for r in result
            ])
            
            return df
        except Exception as e:
            self._logger.error(f"Erro ao recuperar agregação de métricas: {e}")
            return pd.DataFrame()
        finally:
            session.close()