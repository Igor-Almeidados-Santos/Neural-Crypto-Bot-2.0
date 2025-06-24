"""
Serviço de análise para o módulo Analytics.

Este serviço fornece funcionalidades de alto nível para obtenção de métricas,
geração de relatórios e análises de dados relacionados a trading.
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import uuid
import asyncio
import json
import numpy as np
import pandas as pd
from decimal import Decimal

from common.application.base_service import BaseService
from analytics.domain.value_objects.performance_metric import (
    PerformanceMetric, PerformanceMetricCollection, PerformanceAnalysis, MetricType
)


class AnalyticsService(BaseService):
    """
    Serviço para análises e métricas de desempenho.
    
    Este serviço fornece métodos para obtenção de métricas de desempenho,
    geração de relatórios, análise de mercado e gerenciamento de alertas.
    """
    
    def __init__(
        self,
        data_warehouse=None,
        dashboard_publisher=None,
        performance_service=None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o serviço de análise.
        
        Args:
            data_warehouse: Repositório de dados opcional
            dashboard_publisher: Publicador de dashboards opcional
            performance_service: Serviço de atribuição de performance opcional
            logger: Logger opcional
        """
        super().__init__(logger)
        self.data_warehouse = data_warehouse
        self.dashboard_publisher = dashboard_publisher
        self.performance_service = performance_service
    
    def set_data_warehouse(self, data_warehouse):
        """Define o data warehouse para o serviço."""
        self.data_warehouse = data_warehouse
    
    def set_dashboard_publisher(self, dashboard_publisher):
        """Define o dashboard publisher para o serviço."""
        self.dashboard_publisher = dashboard_publisher
    
    def set_performance_service(self, performance_service):
        """Define o serviço de performance para o serviço."""
        self.performance_service = performance_service
    
    async def get_performance_metrics(
        self,
        user_id: str,
        strategy_ids: Optional[List[str]] = None,
        time_range: str = "month",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Obtém métricas de desempenho para estratégias.
        
        Args:
            user_id: ID do usuário
            strategy_ids: Lista de IDs de estratégias (opcional)
            time_range: Intervalo de tempo (day, week, month, year, all)
            **kwargs: Parâmetros adicionais
            
        Returns:
            Lista de métricas de desempenho por estratégia
        """
        self.logger.info(f"Obtendo métricas de desempenho para usuário {user_id}")
        
        # Se strategy_ids não for fornecido, obter todas as estratégias ativas do usuário
        if not strategy_ids and self.data_warehouse:
            active_strategies = await self.data_warehouse.get_active_strategies(user_id)
            strategy_ids = [s["id"] for s in active_strategies]
        
        if not strategy_ids:
            return []
        
        # Obter análise de desempenho para cada estratégia
        results = []
        for strategy_id in strategy_ids:
            try:
                if self.data_warehouse:
                    analysis = await self.data_warehouse.get_performance_analysis(
                        strategy_id=strategy_id,
                        time_range=time_range
                    )
                    
                    if analysis:
                        results.append(analysis)
            except Exception as e:
                self.logger.error(f"Erro ao obter métricas para estratégia {strategy_id}: {e}")
        
        return results
    
    async def get_portfolio_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Obtém resumo do portfolio.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Resumo do portfolio com principais métricas
        """
        self.logger.info(f"Obtendo resumo do portfolio para usuário {user_id}")
        
        # Obter estratégias ativas
        active_strategies = []
        if self.data_warehouse:
            active_strategies = await self.data_warehouse.get_active_strategies(user_id)
        
        # Calcular métricas do portfolio
        total_capital = 0.0
        allocated_capital = 0.0
        total_profit_loss = 0.0
        
        for strategy in active_strategies:
            allocation = strategy.get("allocation", 0.0)
            total_capital += allocation
            allocated_capital += allocation
            
            metrics = strategy.get("metrics", {})
            if "total_return" in metrics:
                profit_loss = allocation * metrics["total_return"]
                total_profit_loss += profit_loss
        
        # Adicionar algum capital não alocado para exemplo
        available_capital = total_capital * 0.2  # 20% não alocado para exemplo
        total_capital += available_capital
        
        # Calcular percentual de lucro/prejuízo
        total_profit_loss_percent = (total_profit_loss / total_capital) if total_capital > 0 else 0
        
        return {
            "user_id": user_id,
            "total_capital": float(total_capital),
            "allocated_capital": float(allocated_capital),
            "available_capital": float(available_capital),
            "total_profit_loss": float(total_profit_loss),
            "total_profit_loss_percent": float(total_profit_loss_percent),
            "active_strategies": len(active_strategies),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def get_exposure_analysis(self, user_id: str) -> Dict[str, Any]:
        """
        Obtém análise de exposição do portfolio.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Análise detalhada de exposição
        """
        self.logger.info(f"Obtendo análise de exposição para usuário {user_id}")
        
        # Obter estratégias ativas
        active_strategies = []
        if self.data_warehouse:
            active_strategies = await self.data_warehouse.get_active_strategies(user_id)
        
        # Calcular exposição por ativo
        exposure_by_asset = {}
        total_exposure = 0.0
        
        for strategy in active_strategies:
            allocation = strategy.get("allocation", 0.0)
            assets = strategy.get("details", {}).get("assets", "unknown")
            
            # Dividir exposição por ativo
            if assets == "multi_asset":
                # Para estratégias multi-asset, simular exposição em diferentes ativos
                asset_list = ["BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "DOT/USD"]
                weights = np.random.dirichlet(np.ones(len(asset_list)))
                
                for asset, weight in zip(asset_list, weights):
                    asset_exposure = allocation * weight
                    exposure_by_asset[asset] = exposure_by_asset.get(asset, 0.0) + asset_exposure
                    total_exposure += asset_exposure
            else:
                # Para estratégias de ativo único
                exposure_by_asset[assets] = exposure_by_asset.get(assets, 0.0) + allocation
                total_exposure += allocation
        
        # Calcular exposição por tipo de estratégia
        exposure_by_strategy_type = {}
        
        for strategy in active_strategies:
            allocation = strategy.get("allocation", 0.0)
            strategy_type = strategy.get("details", {}).get("type", "unknown")
            
            exposure_by_strategy_type[strategy_type] = exposure_by_strategy_type.get(strategy_type, 0.0) + allocation
        
        # Calcular exposição por timeframe
        exposure_by_timeframe = {}
        
        for strategy in active_strategies:
            allocation = strategy.get("allocation", 0.0)
            timeframe = strategy.get("details", {}).get("timeframe", "unknown")
            
            exposure_by_timeframe[timeframe] = exposure_by_timeframe.get(timeframe, 0.0) + allocation
        
        return {
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_exposure": float(total_exposure),
            "exposure_by_asset": {k: float(v) for k, v in exposure_by_asset.items()},
            "exposure_by_strategy_type": {k: float(v) for k, v in exposure_by_strategy_type.items()},
            "exposure_by_timeframe": {k: float(v) for k, v in exposure_by_timeframe.items()}
        }
    
    async def get_trade_analysis(
        self,
        user_id: str,
        page: int = 1,
        size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Obtém análise de trades com paginação.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            filters: Filtros adicionais
            
        Returns:
            Análise de trades paginada
        """
        self.logger.info(f"Obtendo análise de trades para usuário {user_id}")
        
        # Inicializar filtros padrão se não fornecidos
        if not filters:
            filters = {}
        
        # Obter estratégias ativas
        active_strategies = []
        if self.data_warehouse:
            active_strategies = await self.data_warehouse.get_active_strategies(user_id)
        
        # Coletar trades de todas as estratégias
        all_trades = []
        
        for strategy in active_strategies:
            strategy_id = strategy.get("id")
            
            # Obter trades para a estratégia
            if self.data_warehouse:
                try:
                    trades_data = await self.data_warehouse.get_trades(
                        strategy_id=strategy_id,
                        status=filters.get("status", "closed")
                    )
                    
                    if trades_data and "trades" in trades_data:
                        all_trades.extend(trades_data["trades"])
                except Exception as e:
                    self.logger.error(f"Erro ao obter trades para estratégia {strategy_id}: {e}")
        
        # Aplicar filtros
        filtered_trades = all_trades
        
        if "symbol" in filters:
            filtered_trades = [t for t in filtered_trades if t.get("symbol") == filters["symbol"]]
        
        if "direction" in filters:
            filtered_trades = [t for t in filtered_trades if t.get("direction") == filters["direction"]]
        
        if "min_profit" in filters:
            filtered_trades = [t for t in filtered_trades if t.get("profit_loss", 0) >= filters["min_profit"]]
        
        if "max_profit" in filters:
            filtered_trades = [t for t in filtered_trades if t.get("profit_loss", 0) <= filters["max_profit"]]
        
        if "start_date" in filters:
            start_date = filters["start_date"]
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            filtered_trades = [t for t in filtered_trades if t.get("exit_time") >= start_date]
        
        if "end_date" in filters:
            end_date = filters["end_date"]
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            filtered_trades = [t for t in filtered_trades if t.get("exit_time") <= end_date]
        
        # Ordenar por data de saída (mais recente primeiro)
        filtered_trades.sort(key=lambda x: x.get("exit_time", datetime.min), reverse=True)
        
        # Aplicar paginação
        total = len(filtered_trades)
        pages = (total + size - 1) // size if size > 0 else 0
        
        start_idx = (page - 1) * size
        end_idx = min(start_idx + size, total)
        
        paginated_trades = filtered_trades[start_idx:end_idx]
        
        # Calcular estatísticas dos trades filtrados
        stats = self._calculate_trade_statistics(filtered_trades)
        
        return {
            "user_id": user_id,
            "trades": paginated_trades,
            "total": total,
            "page": page,
            "size": size,
            "pages": pages,
            "statistics": stats
        }
    
    def _calculate_trade_statistics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula estatísticas de trades.
        
        Args:
            trades: Lista de trades
            
        Returns:
            Estatísticas calculadas
        """
        if not trades:
            return {
                "count": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "total_profit": 0,
                "total_loss": 0,
                "net_profit": 0
            }
        
        # Separar trades vencedores e perdedores
        winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit_loss", 0) <= 0]
        
        # Calcular estatísticas
        count = len(trades)
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        win_rate = win_count / count if count > 0 else 0
        
        total_profit = sum(t.get("profit_loss", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("profit_loss", 0) for t in losing_trades))
        
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        net_profit = total_profit - total_loss
        
        return {
            "count": count,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "avg_profit": float(avg_profit),
            "avg_loss": float(avg_loss),
            "total_profit": float(total_profit),
            "total_loss": float(total_loss),
            "net_profit": float(net_profit)
        }
    
    async def get_market_analysis(
        self,
        trading_pairs: List[str],
        time_range: str = "day",
        include_sentiment: bool = False,
        include_correlations: bool = True,
        include_indicators: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Obtém análise de mercado para pares de trading.
        
        Args:
            trading_pairs: Lista de pares de trading
            time_range: Intervalo de tempo (day, week, month, year)
            include_sentiment: Se deve incluir análise de sentimento
            include_correlations: Se deve incluir correlações
            include_indicators: Se deve incluir indicadores técnicos
            **kwargs: Parâmetros adicionais
            
        Returns:
            Lista de análises por par de trading
        """
        self.logger.info(f"Obtendo análise de mercado para {len(trading_pairs)} pares")
        
        # Determinar datas com base no intervalo
        end_date = datetime.utcnow()
        
        if time_range == "day":
            start_date = end_date - timedelta(days=1)
            interval = "5m"
        elif time_range == "week":
            start_date = end_date - timedelta(days=7)
            interval = "1h"
        elif time_range == "month":
            start_date = end_date - timedelta(days=30)
            interval = "4h"
        elif time_range == "year":
            start_date = end_date - timedelta(days=365)
            interval = "1d"
        else:  # default: day
            start_date = end_date - timedelta(days=1)
            interval = "5m"
        
        # Obter dados de mercado para cada par
        market_data_list = []
        
        for pair in trading_pairs:
            try:
                if self.data_warehouse:
                    market_data = await self.data_warehouse.get_market_data(
                        symbol=pair,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval
                    )
                    
                    if market_data:
                        market_data_list.append(market_data)
            except Exception as e:
                self.logger.error(f"Erro ao obter dados de mercado para {pair}: {e}")
        
        # Obter dados de sentimento se solicitado
        sentiment_data_list = []
        
        if include_sentiment:
            for pair in trading_pairs:
                try:
                    if self.data_warehouse:
                        sentiment_data = await self.data_warehouse.get_sentiment_data(
                            symbol=pair,
                            start_date=start_date,
                            end_date=end_date
                        )
                        
                        if sentiment_data:
                            sentiment_data_list.append(sentiment_data)
                except Exception as e:
                    self.logger.error(f"Erro ao obter dados de sentimento para {pair}: {e}")
        
        # Calcular correlações se solicitado
        correlation_matrix = None
        
        if include_correlations and len(market_data_list) > 1 and self.data_warehouse:
            try:
                correlation_matrix = self.data_warehouse._calculate_correlation_matrix(market_data_list)
            except Exception as e:
                self.logger.error(f"Erro ao calcular correlações: {e}")
        
        # Obter indicadores técnicos se solicitado
        indicators_list = []
        
        if include_indicators:
            indicators = kwargs.get("indicators", ["sma", "ema", "rsi", "bollinger", "macd"])
            
            for market_data in market_data_list:
                try:
                    if self.data_warehouse:
                        indicators_data = await self.data_warehouse.get_technical_indicators(
                            symbol=market_data["symbol"],
                            indicators=indicators,
                            start_date=start_date,
                            end_date=end_date,
                            interval=interval
                        )
                        
                        if indicators_data:
                            indicators_list.append(indicators_data)
                except Exception as e:
                    self.logger.error(f"Erro ao obter indicadores técnicos para {market_data['symbol']}: {e}")
        
        # Preparar resultados
        results = []
        
        for i, market_data in enumerate(market_data_list):
            symbol = market_data["symbol"]
            
            # Encontrar dados de sentimento correspondentes
            sentiment = next((s for s in sentiment_data_list if s["symbol"] == symbol), None)
            
            # Encontrar indicadores correspondentes
            indicators = next((ind for ind in indicators_list if ind["symbol"] == symbol), None)
            
            # Calcular estatísticas básicas
            returns = np.array(market_data["returns"])
            prices = np.array(market_data["prices"])
            
            # Volatilidade (anualizada)
            volatility = np.std(returns) * np.sqrt(252 if interval == "1d" else 365 * 24 / {
                "5m": 5/60, "1h": 1, "4h": 4, "1d": 24
            }.get(interval, 24))
            
            # Variação percentual
            change_percent = (prices[-1] / prices[0] - 1) * 100
            
            # Preparar resultado para este par
            result = {
                "symbol": symbol,
                "interval": interval,
                "time_range": time_range,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "stats": {
                    "open": float(prices[0]),
                    "close": float(prices[-1]),
                    "high": float(np.max(prices)),
                    "low": float(np.min(prices)),
                    "change_percent": float(change_percent),
                    "volatility": float(volatility),
                    "returns_mean": float(np.mean(returns) * 100),  # percentual
                    "returns_median": float(np.median(returns) * 100),  # percentual
                }
            }
            
            # Adicionar dados de sentimento se disponíveis
            if sentiment:
                result["sentiment"] = {
                    "average_score": float(np.mean(sentiment["sentiment_scores"])),
                    "latest_score": float(sentiment["sentiment_scores"][-1]),
                    "bullish_count": sentiment["sentiment_categories"].count("bullish"),
                    "bearish_count": sentiment["sentiment_categories"].count("bearish"),
                    "neutral_count": sentiment["sentiment_categories"].count("neutral"),
                }
            
            # Adicionar informações de indicadores técnicos
            if indicators:
                result["indicators"] = True
            
            results.append(result)
        
        # Adicionar matriz de correlação se calculada
        if correlation_matrix:
            for result in results:
                symbol = result["symbol"]
                correlations = {}
                
                # Extrair correlações para este símbolo
                for other_symbol in correlation_matrix["symbols"]:
                    if other_symbol != symbol:
                        corr_value = correlation_matrix["correlation_matrix"].get(symbol, {}).get(other_symbol)
                        if corr_value is not None:
                            correlations[other_symbol] = float(corr_value)
                
                result["correlations"] = correlations
        
        return results
    
    async def generate_report(
        self,
        user_id: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Registra a geração de um relatório.
        
        Args:
            user_id: ID do usuário
            request: Dados do relatório
            
        Returns:
            Resultado do registro
        """
        self.logger.info(f"Registrando geração de relatório para usuário {user_id}")
        
        # Gerar ID para o relatório
        report_id = str(uuid.uuid4())
        
        # Adicionar dados padrão
        report_data = {
            "id": report_id,
            "user_id": user_id,
            "name": request.get("name", "Untitled Report"),
            "type": request.get("type", "unknown"),
            "parameters": request.get("parameters", {}),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Salvar no data warehouse
        if self.data_warehouse:
            try:
                await self.data_warehouse.save_report(
                    user_id=user_id,
                    report_name=report_data["name"],
                    report_type=report_data["type"],
                    url=request.get("url", ""),
                    parameters=report_data["parameters"]
                )
            except Exception as e:
                self.logger.error(f"Erro ao salvar relatório: {e}")
        
        return report_data
    
    async def list_reports(
        self,
        user_id: str,
        page: int = 1,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Lista relatórios gerados anteriormente.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            
        Returns:
            Lista de relatórios
        """
        self.logger.info(f"Listando relatórios para usuário {user_id}")
        
        if self.data_warehouse:
            try:
                return await self.data_warehouse.get_reports(
                    user_id=user_id,
                    page=page,
                    size=size
                )
            except Exception as e:
                self.logger.error(f"Erro ao listar relatórios: {e}")
        
        # Retornar lista vazia se não foi possível obter relatórios
        return {
            "reports": [],
            "total": 0,
            "page": page,
            "size": size,
            "pages": 0
        }
    
    async def list_dashboards(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Lista configurações de dashboards disponíveis.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de configurações de dashboard
        """
        self.logger.info(f"Listando dashboards para usuário {user_id}")
        
        # Implementação simulada
        # Na implementação real, isso buscaria do banco de dados
        
        # Criar alguns dashboards de exemplo
        dashboards = [
            {
                "id": "dashboard-1",
                "name": "Portfolio Overview",
                "description": "Visão geral do portfolio com principais métricas",
                "is_default": True,
                "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
                "last_updated": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "widgets_count": 8
            },
            {
                "id": "dashboard-2",
                "name": "Market Analysis",
                "description": "Análise de mercado para principais criptomoedas",
                "is_default": False,
                "created_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
                "last_updated": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "widgets_count": 6
            },
            {
                "id": "dashboard-3",
                "name": "Strategy Performance",
                "description": "Desempenho detalhado das estratégias",
                "is_default": False,
                "created_at": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "last_updated": (datetime.utcnow() - timedelta(hours=12)).isoformat(),
                "widgets_count": 10
            }
        ]
        
        return dashboards
    
    async def create_dashboard(
        self,
        user_id: str,
        dashboard: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cria um novo dashboard.
        
        Args:
            user_id: ID do usuário
            dashboard: Configuração do dashboard
            
        Returns:
            Dashboard criado
        """
        self.logger.info(f"Criando dashboard para usuário {user_id}")
        
        # Gerar ID para o dashboard
        dashboard_id = str(uuid.uuid4())
        
        # Obter nome e layout
        dashboard_name = dashboard.get("name", "New Dashboard")
        layout = dashboard.get("layout", {})
        
        # Criar URL para o dashboard
        dashboard_url = ""
        
        if self.dashboard_publisher:
            try:
                dashboard_url = await self.dashboard_publisher.create_dashboard(
                    user_id=user_id,
                    dashboard_id=dashboard_id,
                    dashboard_name=dashboard_name,
                    layout=layout
                )
            except Exception as e:
                self.logger.error(f"Erro ao criar dashboard: {e}")
        
        # Adicionar dados padrão
        dashboard_data = {
            "id": dashboard_id,
            "user_id": user_id,
            "name": dashboard_name,
            "layout": layout,
            "is_default": dashboard.get("is_default", False),
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "url": dashboard_url
        }
        
        return dashboard_data
    
    async def list_alerts(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Lista configurações de alertas.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de configurações de alerta
        """
        self.logger.info(f"Listando alertas para usuário {user_id}")
        
        # Implementação simulada
        # Na implementação real, isso buscaria do banco de dados
        
        # Criar alguns alertas de exemplo
        alerts = [
            {
                "id": "alert-1",
                "name": "Drawdown Alert",
                "metric_type": "MAX_DRAWDOWN",
                "strategy_id": "strategy-123",
                "condition": ">",
                "threshold": 0.10,  # 10%
                "level": "warning",
                "notification_channels": ["email", "app"],
                "created_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
                "is_active": True
            },
            {
                "id": "alert-2",
                "name": "Low Sharpe Ratio",
                "metric_type": "SHARPE_RATIO",
                "strategy_id": "strategy-456",
                "condition": "<",
                "threshold": 0.5,
                "level": "warning",
                "notification_channels": ["email"],
                "created_at": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                "is_active": True
            },
            {
                "id": "alert-3",
                "name": "High Volatility",
                "metric_type": "VOLATILITY",
                "strategy_id": None,  # Todas as estratégias
                "condition": ">",
                "threshold": 0.30,  # 30%
                "level": "error",
                "notification_channels": ["email", "app", "sms"],
                "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
                "is_active": True
            }
        ]
        
        return alerts
    
    async def create_alert(
        self,
        user_id: str,
        alert_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Cria uma nova configuração de alerta.
        
        Args:
            user_id: ID do usuário
            alert_config: Configuração do alerta
            
        Returns:
            Alerta criado
        """
        self.logger.info(f"Criando alerta para usuário {user_id}")
        
        # Validar configuração mínima
        if "metric_type" not in alert_config or "condition" not in alert_config or "threshold" not in alert_config:
            raise ValueError("Configuração de alerta incompleta. Necessário: metric_type, condition, threshold")
        
        # Gerar ID para o alerta
        alert_id = str(uuid.uuid4())
        
        # Adicionar dados padrão
        alert_data = {
            "id": alert_id,
            "user_id": user_id,
            "name": alert_config.get("name", f"Alert {alert_id}"),
            "metric_type": alert_config["metric_type"],
            "strategy_id": alert_config.get("strategy_id"),
            "condition": alert_config["condition"],
            "threshold": float(alert_config["threshold"]),
            "level": alert_config.get("level", "info"),
            "notification_channels": alert_config.get("notification_channels", ["app"]),
            "created_at": datetime.utcnow().isoformat(),
            "is_active": alert_config.get("is_active", True)
        }
        
        return alert_data
    
    async def get_alert_history(
        self,
        user_id: str,
        page: int = 1,
        size: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Obtém histórico de alertas disparados.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            filters: Filtros adicionais
            
        Returns:
            Histórico de alertas paginado
        """
        self.logger.info(f"Obtendo histórico de alertas para usuário {user_id}")
        
        # Inicializar filtros padrão se não fornecidos
        if not filters:
            filters = {}
        
        # Implementação simulada
        # Na implementação real, isso buscaria do banco de dados
        
        # Criar alguns eventos de alerta de exemplo
        alert_events = []
        
        # Alerta de drawdown
        alert_events.append({
            "id": "event-1",
            "alert_id": "alert-1",
            "user_id": user_id,
            "name": "Drawdown Alert",
            "metric_type": "MAX_DRAWDOWN",
            "strategy_id": "strategy-123",
            "strategy_name": "Strategy 1",
            "condition": ">",
            "threshold": 0.10,
            "actual_value": 0.12,
            "level": "warning",
            "message": "Maximum drawdown exceeded threshold: 12% > 10%",
            "timestamp": (datetime.utcnow() - timedelta(days=3, hours=6)).isoformat(),
            "acknowledged": True,
            "acknowledged_at": (datetime.utcnow() - timedelta(days=3, hours=5)).isoformat()
        })
        
        # Alerta de Sharpe Ratio
        alert_events.append({
            "id": "event-2",
            "alert_id": "alert-2",
            "user_id": user_id,
            "name": "Low Sharpe Ratio",
            "metric_type": "SHARPE_RATIO",
            "strategy_id": "strategy-456",
            "strategy_name": "Strategy 2",
            "condition": "<",
            "threshold": 0.5,
            "actual_value": 0.32,
            "level": "warning",
            "message": "Sharpe ratio below threshold: 0.32 < 0.5",
            "timestamp": (datetime.utcnow() - timedelta(days=2, hours=9)).isoformat(),
            "acknowledged": True,
            "acknowledged_at": (datetime.utcnow() - timedelta(days=2, hours=7)).isoformat()
        })
        
        # Alerta de volatilidade
        alert_events.append({
            "id": "event-3",
            "alert_id": "alert-3",
            "user_id": user_id,
            "name": "High Volatility",
            "metric_type": "VOLATILITY",
            "strategy_id": "strategy-789",
            "strategy_name": "Strategy 3",
            "condition": ">",
            "threshold": 0.30,
            "actual_value": 0.37,
            "level": "error",
            "message": "Portfolio volatility exceeded threshold: 37% > 30%",
            "timestamp": (datetime.utcnow() - timedelta(hours=14)).isoformat(),
            "acknowledged": False,
            "acknowledged_at": None
        })
        
        # Outro alerta de volatilidade mais recente
        alert_events.append({
            "id": "event-4",
            "alert_id": "alert-3",
            "user_id": user_id,
            "name": "High Volatility",
            "metric_type": "VOLATILITY",
            "strategy_id": "strategy-789",
            "strategy_name": "Strategy 3",
            "condition": ">",
            "threshold": 0.30,
            "actual_value": 0.42,
            "level": "error",
            "message": "Portfolio volatility exceeded threshold: 42% > 30%",
            "timestamp": (datetime.utcnow() - timedelta(hours=4)).isoformat(),
            "acknowledged": False,
            "acknowledged_at": None
        })
        
        # Aplicar filtros
        filtered_events = alert_events
        
        if "level" in filters:
            filtered_events = [e for e in filtered_events if e["level"] == filters["level"]]
        
        if "strategy_id" in filters:
            filtered_events = [e for e in filtered_events if e["strategy_id"] == filters["strategy_id"]]
        
        if "metric_type" in filters:
            filtered_events = [e for e in filtered_events if e["metric_type"] == filters["metric_type"]]
        
        if "acknowledged" in filters:
            acknowledged = filters["acknowledged"]
            if acknowledged is True:
                filtered_events = [e for e in filtered_events if e["acknowledged"]]
            elif acknowledged is False:
                filtered_events = [e for e in filtered_events if not e["acknowledged"]]
        
        if "start_date" in filters:
            start_date = filters["start_date"]
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date)
            filtered_events = [e for e in filtered_events if datetime.fromisoformat(e["timestamp"]) >= start_date]
        
        if "end_date" in filters:
            end_date = filters["end_date"]
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date)
            filtered_events = [e for e in filtered_events if datetime.fromisoformat(e["timestamp"]) <= end_date]
        
        # Ordenar por data (mais recente primeiro)
        filtered_events.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Aplicar paginação
        total = len(filtered_events)
        pages = (total + size - 1) // size if size > 0 else 0
        
        start_idx = (page - 1) * size
        end_idx = min(start_idx + size, total)
        
        paginated_events = filtered_events[start_idx:end_idx]
        
        return {
            "events": paginated_events,
            "total": total,
            "page": page,
            "size": size,
            "pages": pages,
            "unacknowledged_count": sum(1 for e in filtered_events if not e["acknowledged"])
        }
    
    async def acknowledge_alert(
        self,
        user_id: str,
        event_id: str
    ) -> bool:
        """
        Marca um alerta como reconhecido.
        
        Args:
            user_id: ID do usuário
            event_id: ID do evento de alerta
            
        Returns:
            True se reconhecido com sucesso, False caso contrário
        """
        self.logger.info(f"Reconhecendo alerta {event_id} para usuário {user_id}")
        
        # Implementação simulada
        # Na implementação real, isso atualizaria o banco de dados
        
        # Simulando sucesso
        return True
    
    async def update_metric_thresholds(
        self,
        user_id: str,
        strategy_id: str,
        thresholds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Atualiza limiares de métricas para uma estratégia.
        
        Args:
            user_id: ID do usuário
            strategy_id: ID da estratégia
            thresholds: Dicionário de limiares por tipo de métrica
            
        Returns:
            Configuração atualizada
        """
        self.logger.info(f"Atualizando limiares de métricas para estratégia {strategy_id}")
        
        # Implementação simulada
        # Na implementação real, isso atualizaria o banco de dados
        
        # Validar limiares
        for metric_type, value in thresholds.items():
            if not isinstance(value, (int, float, Decimal)):
                raise ValueError(f"Valor inválido para limiar de {metric_type}: {value}")
        
        # Criar configuração
        config = {
            "user_id": user_id,
            "strategy_id": strategy_id,
            "thresholds": {k: float(v) for k, v in thresholds.items()},
            "updated_at": datetime.utcnow().isoformat()
        }
        
        return config
    
    async def get_asset_correlation_analysis(
        self,
        assets: List[str],
        time_range: str = "month"
    ) -> Dict[str, Any]:
        """
        Obtém análise de correlação entre ativos.
        
        Args:
            assets: Lista de símbolos de ativos
            time_range: Intervalo de tempo (day, week, month, year)
            
        Returns:
            Análise de correlação entre os ativos
        """
        self.logger.info(f"Obtendo análise de correlação para {len(assets)} ativos")
        
        # Determinar datas com base no intervalo
        end_date = datetime.utcnow()
        
        if time_range == "day":
            start_date = end_date - timedelta(days=1)
            interval = "5m"
        elif time_range == "week":
            start_date = end_date - timedelta(days=7)
            interval = "1h"
        elif time_range == "month":
            start_date = end_date - timedelta(days=30)
            interval = "4h"
        elif time_range == "year":
            start_date = end_date - timedelta(days=365)
            interval = "1d"
        else:  # default: month
            start_date = end_date - timedelta(days=30)
            interval = "4h"
        
        # Obter dados de mercado para cada ativo
        market_data_list = []
        
        for asset in assets:
            try:
                if self.data_warehouse:
                    market_data = await self.data_warehouse.get_market_data(
                        symbol=asset,
                        start_date=start_date,
                        end_date=end_date,
                        interval=interval
                    )
                    
                    if market_data:
                        market_data_list.append(market_data)
            except Exception as e:
                self.logger.error(f"Erro ao obter dados de mercado para {asset}: {e}")
        
        # Calcular correlações
        correlation_matrix = None
        
        if len(market_data_list) > 1 and self.data_warehouse:
            try:
                correlation_matrix = self.data_warehouse._calculate_correlation_matrix(market_data_list)
            except Exception as e:
                self.logger.error(f"Erro ao calcular correlações: {e}")
        
        if not correlation_matrix:
            return {
                "assets": assets,
                "time_range": time_range,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "correlation_matrix": {},
                "avg_correlation": 0
            }
        
        # Calcular correlação média (excluindo auto-correlações)
        correlations = []
        
        for i, asset1 in enumerate(correlation_matrix["symbols"]):
            for j, asset2 in enumerate(correlation_matrix["symbols"]):
                if i != j:  # Excluir auto-correlações
                    corr_value = correlation_matrix["correlation_matrix"].get(asset1, {}).get(asset2)
                    if corr_value is not None:
                        correlations.append(corr_value)
        
        avg_correlation = np.mean(correlations) if correlations else 0
        
        return {
            "assets": assets,
            "time_range": time_range,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "correlation_matrix": correlation_matrix["correlation_matrix"],
            "avg_correlation": float(avg_correlation)
        }
    
    async def calculate_portfolio_allocation(
        self,
        user_id: str,
        strategies: List[Dict[str, Any]],
        risk_profile: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Calcula alocação ideal de portfolio com base no perfil de risco.
        
        Args:
            user_id: ID do usuário
            strategies: Lista de estratégias com dados de desempenho
            risk_profile: Perfil de risco (conservative, moderate, aggressive)
            
        Returns:
            Alocação sugerida por estratégia
        """
        self.logger.info(f"Calculando alocação de portfolio para usuário {user_id}")
        
        # Determinar pesos com base no perfil de risco
        if risk_profile == "conservative":
            # Priorizar Sharpe e minimizar drawdown
            sharpe_weight = 0.5
            return_weight = 0.2
            drawdown_weight = -0.3
        elif risk_profile == "aggressive":
            # Priorizar retorno, aceitar mais drawdown
            sharpe_weight = 0.2
            return_weight = 0.6
            drawdown_weight = -0.2
        else:  # moderate
            # Balancear retorno e risco
            sharpe_weight = 0.4
            return_weight = 0.4
            drawdown_weight = -0.2
        
        # Calcular pontuação para cada estratégia
        strategies_with_scores = []
        
        for strategy in strategies:
            metrics = strategy.get("metrics", {})
            
            # Obter métricas (com valores padrão)
            sharpe = metrics.get("sharpe_ratio", 0)
            annualized_return = metrics.get("annualized_return", 0)
            max_drawdown = metrics.get("max_drawdown", 0)
            
            # Calcular pontuação
            score = (
                sharpe_weight * sharpe +
                return_weight * annualized_return +
                drawdown_weight * max_drawdown
            )
            
            strategies_with_scores.append({
                "id": strategy.get("id"),
                "name": strategy.get("name", f"Strategy {strategy.get('id')}"),
                "score": score,
                "metrics": metrics
            })
        
        # Ordenar por pontuação (maior primeiro)
        strategies_with_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Calcular alocação com base na pontuação
        total_score = sum(max(s["score"], 0.01) for s in strategies_with_scores)  # Evitar divisão por zero
        
        allocations = []
        
        for strategy in strategies_with_scores:
            # Garantir pontuação mínima para evitar alocação zero
            score = max(strategy["score"], 0.01)
            
            # Calcular alocação proporcional à pontuação
            allocation = score / total_score if total_score > 0 else 1.0 / len(strategies_with_scores)
            
            allocations.append({
                "id": strategy["id"],
                "name": strategy["name"],
                "allocation": float(allocation),
                "score": float(strategy["score"]),
                "metrics": strategy["metrics"]
            })
        
        return {
            "user_id": user_id,
            "risk_profile": risk_profile,
            "timestamp": datetime.utcnow().isoformat(),
            "allocations": allocations
        }
    
    async def get_metrics_history(
        self,
        strategy_id: str,
        metric_types: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Obtém histórico de métricas para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            metric_types: Lista de tipos de métrica
            start_date: Data de início opcional
            end_date: Data de fim opcional
            
        Returns:
            Histórico de métricas
        """
        self.logger.info(f"Obtendo histórico de métricas para estratégia {strategy_id}")
        
        # Definir datas padrão se não especificadas
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=90)  # 90 dias por padrão
        
        # Implementação simulada
        # Na implementação real, isso buscaria do banco de dados
        
        # Criar série temporal para o período
        current_date = start_date
        dates = []
        
        while current_date <= end_date:
            # Apenas dias úteis
            if current_date.weekday() < 5:  # Segunda a Sexta
                dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Criar séries para cada métrica
        metrics_history = {}
        
        for metric_type in metric_types:
            # Inicializar série para a métrica
            metrics_history[metric_type] = {
                "dates": [d.isoformat() for d in dates],
                "values": []
            }
            
            # Gerar valores com alguma continuidade (modelo AR(1) simples)
            values = []
            
            # Valor inicial baseado no tipo de métrica
            if metric_type == "TOTAL_RETURN":
                value = 0.0  # Começa em 0
                volatility = 0.005  # Volatilidade diária
            elif metric_type == "SHARPE_RATIO":
                value = 1.5  # Valor inicial razoável
                volatility = 0.1
            elif metric_type == "MAX_DRAWDOWN":
                value = 0.05  # 5% inicial
                volatility = 0.003
            elif metric_type == "VOLATILITY":
                value = 0.15  # 15% anualizada
                volatility = 0.01
            else:
                value = 0.0
                volatility = 0.01
            
            # Fator de autocorrelação
            ar_factor = 0.95
            
            for _ in dates:
                # Adicionar valor atual
                values.append(value)
                
                # Gerar próximo valor com alguma autocorrelação
                innovation = np.random.normal(0, volatility)
                value = ar_factor * value + innovation
                
                # Ajustes específicos por tipo de métrica
                if metric_type == "TOTAL_RETURN":
                    # Tendência positiva leve
                    value += 0.0005
                elif metric_type == "MAX_DRAWDOWN":
                    # Garantir valor positivo e máximo razoável
                    value = min(max(value, 0.01), 0.3)
            
            metrics_history[metric_type]["values"] = [float(v) for v in values]
        
        return {
            "strategy_id": strategy_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "metrics": metrics_history
        }