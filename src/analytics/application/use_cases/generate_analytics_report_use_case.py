"""
Caso de uso para geração de relatórios de análise.

Este módulo implementa o caso de uso para gerar diferentes
tipos de relatórios analíticos para estratégias de trading.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import tempfile

from src.common.application.base_use_case import BaseUseCase
from src.analytics.application.services.performance_attribution_service import PerformanceAttributionService
from src.analytics.application.services.analytics_service import AnalyticsService
from src.analytics.infrastructure.dashboard_publisher import DashboardPublisher
from src.analytics.infrastructure.data_warehouse import DataWarehouse


class GenerateAnalyticsReportUseCase(BaseUseCase):
    """Caso de uso para gerar relatórios de análise."""
    
    def __init__(
        self,
        performance_service: PerformanceAttributionService,
        analytics_service: AnalyticsService,
        data_warehouse: DataWarehouse,
        dashboard_publisher: DashboardPublisher,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            performance_service: Serviço de atribuição de performance
            analytics_service: Serviço de analytics
            data_warehouse: Repositório de dados
            dashboard_publisher: Publicador de dashboards
            logger: Logger opcional
        """
        super().__init__(logger)
        self.performance_service = performance_service
        self.analytics_service = analytics_service
        self.data_warehouse = data_warehouse
        self.dashboard_publisher = dashboard_publisher
    
    async def execute(
        self,
        user_id: str,
        report_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa o caso de uso.
        
        Args:
            user_id: ID do usuário
            report_type: Tipo de relatório
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        self.logger.info(f"Gerando relatório do tipo '{report_type}' para usuário {user_id}")
        
        # Verificar o tipo de relatório
        if report_type == "performance_summary":
            return await self._generate_performance_summary(user_id, parameters)
        elif report_type == "strategy_comparison":
            return await self._generate_strategy_comparison(user_id, parameters)
        elif report_type == "market_analysis":
            return await self._generate_market_analysis(user_id, parameters)
        elif report_type == "risk_analysis":
            return await self._generate_risk_analysis(user_id, parameters)
        elif report_type == "trade_analysis":
            return await self._generate_trade_analysis(user_id, parameters)
        elif report_type == "portfolio_overview":
            return await self._generate_portfolio_overview(user_id, parameters)
        else:
            raise ValueError(f"Tipo de relatório não reconhecido: {report_type}")
    
    async def _generate_performance_summary(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de resumo de performance.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        strategy_id = parameters.get("strategy_id")
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        include_benchmark = parameters.get("include_benchmark", False)
        benchmark_symbol = parameters.get("benchmark_symbol", "BTC/USD")
        report_format = parameters.get("format", "pdf")
        
        if not strategy_id:
            raise ValueError("ID da estratégia é obrigatório")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados da estratégia
        strategy_data = await self.data_warehouse.get_strategy_returns(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not strategy_data or "returns" not in strategy_data:
            raise ValueError(f"Não foi possível obter dados para a estratégia {strategy_id}")
        
        # Obter dados de benchmark se solicitado
        benchmark_data = None
        if include_benchmark:
            benchmark_data = await self.data_warehouse.get_market_data(
                symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
        
        # Calcular métricas de performance
        performance_analysis = await self.performance_service.calculate_performance_metrics(
            returns=strategy_data["returns"],
            timestamps=strategy_data["timestamps"],
            benchmark_returns=benchmark_data["returns"] if benchmark_data else None,
            strategy_id=strategy_id,
            strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
            initial_capital=strategy_data.get("initial_capital", 10000.0),
            trades_data=strategy_data.get("trades", [])
        )
        
        # Analisar drawdowns
        drawdowns = await self.performance_service.analyze_drawdowns(
            returns=strategy_data["returns"],
            timestamps=strategy_data["timestamps"],
            threshold=0.01,  # 1%
            max_drawdowns=5
        )
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de retorno cumulativo
            cumulative_return_chart = self._generate_cumulative_return_chart(
                strategy_data["returns"],
                strategy_data["timestamps"],
                benchmark_data["returns"] if benchmark_data else None,
                benchmark_data["timestamps"] if benchmark_data else None,
                strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["cumulative_return"] = cumulative_return_chart
            
            # Gráfico underwater (drawdown)
            underwater_data = await self.performance_service.get_underwater_chart_data(
                strategy_data["returns"],
                strategy_data["timestamps"]
            )
            underwater_chart = self._generate_underwater_chart(underwater_data)
            charts["underwater"] = underwater_chart
            
            # Gráfico de distribuição de retornos
            returns_distribution_chart = self._generate_returns_distribution_chart(
                strategy_data["returns"],
                benchmark_data["returns"] if benchmark_data else None,
                strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["returns_distribution"] = returns_distribution_chart
            
            # Gráfico de retornos mensais
            monthly_returns_chart = self._generate_monthly_returns_chart(
                strategy_data["returns"],
                strategy_data["timestamps"]
            )
            charts["monthly_returns"] = monthly_returns_chart
        
        # Obter dados de trades se disponíveis
        trades_summary = {}
        if "trades" in strategy_data and strategy_data["trades"]:
            trades_summary = self._calculate_trades_summary(strategy_data["trades"])
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"performance_summary_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Performance Summary"),
            performance_analysis=performance_analysis,
            drawdowns=drawdowns,
            charts=charts,
            trades_summary=trades_summary,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Performance Summary"),
                "type": "performance_summary",
                "parameters": parameters,
                "strategy_id": strategy_id,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "include_benchmark": include_benchmark,
                "benchmark_symbol": benchmark_symbol if include_benchmark else None
            }
        )
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Performance Summary"),
            "type": "performance_summary",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "strategy_id": strategy_id,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "summary": {
                "total_return": performance_analysis.metrics.get_metric(MetricType.TOTAL_RETURN).formatted_value,
                "annualized_return": performance_analysis.metrics.get_metric(MetricType.ANNUALIZED_RETURN).formatted_value,
                "sharpe_ratio": performance_analysis.metrics.get_metric(MetricType.SHARPE_RATIO).formatted_value,
                "max_drawdown": performance_analysis.metrics.get_metric(MetricType.MAX_DRAWDOWN).formatted_value,
                "win_rate": f"{performance_analysis.win_rate * 100:.2f}%",
                "profit_loss": f"${performance_analysis.profit_loss:.2f}"
            }
        }
    
    async def _generate_strategy_comparison(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de comparação de estratégias.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        strategy_ids = parameters.get("strategy_ids", [])
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        include_benchmark = parameters.get("include_benchmark", False)
        benchmark_symbol = parameters.get("benchmark_symbol", "BTC/USD")
        report_format = parameters.get("format", "pdf")
        
        if not strategy_ids or len(strategy_ids) < 2:
            raise ValueError("Pelo menos duas estratégias devem ser especificadas para comparação")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados para cada estratégia
        strategies_data = []
        for strategy_id in strategy_ids:
            strategy_data = await self.data_warehouse.get_strategy_returns(
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not strategy_data or "returns" not in strategy_data:
                self.logger.warning(f"Não foi possível obter dados para a estratégia {strategy_id}")
                continue
                
            strategies_data.append(strategy_data)
        
        if not strategies_data:
            raise ValueError("Não foi possível obter dados para nenhuma das estratégias especificadas")
        
        # Obter dados de benchmark se solicitado
        benchmark_data = None
        if include_benchmark:
            benchmark_data = await self.data_warehouse.get_market_data(
                symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
        
        # Calcular métricas de performance para cada estratégia
        performance_analyses = []
        for strategy_data in strategies_data:
            analysis = await self.performance_service.calculate_performance_metrics(
                returns=strategy_data["returns"],
                timestamps=strategy_data["timestamps"],
                benchmark_returns=benchmark_data["returns"] if benchmark_data else None,
                strategy_id=strategy_data["strategy_id"],
                strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_data['strategy_id']}"),
                initial_capital=strategy_data.get("initial_capital", 10000.0),
                trades_data=strategy_data.get("trades", [])
            )
            performance_analyses.append(analysis)
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de comparação de retorno cumulativo
            cumulative_return_chart = self._generate_cumulative_return_comparison_chart(
                strategies_data,
                benchmark_data=benchmark_data if include_benchmark else None,
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["cumulative_return_comparison"] = cumulative_return_chart
            
            # Gráfico de comparação de drawdowns
            drawdown_comparison_chart = self._generate_drawdown_comparison_chart(
                strategies_data,
                benchmark_data=benchmark_data if include_benchmark else None,
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["drawdown_comparison"] = drawdown_comparison_chart
            
            # Gráfico de comparação de métricas
            metrics_comparison_chart = self._generate_metrics_comparison_chart(performance_analyses)
            charts["metrics_comparison"] = metrics_comparison_chart
            
            # Gráfico de correlação entre estratégias
            correlation_chart = self._generate_correlation_chart(strategies_data)
            charts["correlation"] = correlation_chart
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"strategy_comparison_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Strategy Comparison"),
            performance_analyses=performance_analyses,
            charts=charts,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Strategy Comparison"),
                "type": "strategy_comparison",
                "parameters": parameters,
                "strategy_ids": strategy_ids,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "include_benchmark": include_benchmark,
                "benchmark_symbol": benchmark_symbol if include_benchmark else None
            }
        )
        
        # Criar resumo comparativo
        comparison_summary = []
        for analysis in performance_analyses:
            comparison_summary.append({
                "strategy_id": analysis.strategy_id,
                "strategy_name": analysis.strategy_name,
                "total_return": analysis.metrics.get_metric(MetricType.TOTAL_RETURN).formatted_value,
                "annualized_return": analysis.metrics.get_metric(MetricType.ANNUALIZED_RETURN).formatted_value,
                "sharpe_ratio": analysis.metrics.get_metric(MetricType.SHARPE_RATIO).formatted_value,
                "max_drawdown": analysis.metrics.get_metric(MetricType.MAX_DRAWDOWN).formatted_value,
                "win_rate": f"{analysis.win_rate * 100:.2f}%",
                "profit_loss": f"${analysis.profit_loss:.2f}"
            })
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Strategy Comparison"),
            "type": "strategy_comparison",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "strategy_ids": strategy_ids,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "comparison_summary": comparison_summary
        }
    
    async def _generate_market_analysis(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise de mercado.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        symbols = parameters.get("symbols", [])
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_sentiment = parameters.get("include_sentiment", False)
        include_correlations = parameters.get("include_correlations", True)
        include_indicators = parameters.get("include_indicators", True)
        indicators = parameters.get("indicators", ["rsi", "macd", "bollinger"])
        report_format = parameters.get("format", "pdf")
        
        if not symbols:
            raise ValueError("Pelo menos um símbolo deve ser especificado para análise")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados de mercado para cada símbolo
        market_data = []
        for symbol in symbols:
            data = await self.data_warehouse.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            if not data:
                self.logger.warning(f"Não foi possível obter dados para o símbolo {symbol}")
                continue
                
            market_data.append(data)
        
        if not market_data:
            raise ValueError("Não foi possível obter dados para nenhum dos símbolos especificados")
        
        # Obter dados de sentimento se solicitado
        sentiment_data = None
        if include_sentiment:
            sentiment_data = []
            for symbol in symbols:
                data = await self.data_warehouse.get_sentiment_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if data:
                    sentiment_data.append(data)
        
        # Calcular correlações se solicitado
        correlation_matrix = None
        if include_correlations and len(symbols) > 1:
            correlation_matrix = self._calculate_correlation_matrix(market_data)
        
        # Calcular indicadores técnicos se solicitado
        technical_indicators = None
        if include_indicators:
            technical_indicators = []
            for data in market_data:
                indicators_data = self._calculate_technical_indicators(
                    data,
                    indicators=indicators
                )
                technical_indicators.append(indicators_data)
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"market_analysis_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Market Analysis"),
            market_data=market_data,
            sentiment_data=sentiment_data,
            correlation_matrix=correlation_matrix,
            technical_indicators=technical_indicators,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Market Analysis"),
                "type": "market_analysis",
                "parameters": parameters,
                "symbols": symbols,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "include_sentiment": include_sentiment,
                "include_correlations": include_correlations,
                "include_indicators": include_indicators,
                "indicators": indicators
            }
        )
        
        # Criar resumo de mercado
        market_summary = []
        for data in market_data:
            returns = np.array(data["returns"])
            prices = np.array(data["prices"])
            
            market_summary.append({
                "symbol": data["symbol"],
                "start_price": float(prices[0]),
                "end_price": float(prices[-1]),
                "change_percent": float((prices[-1] / prices[0] - 1) * 100),
                "volatility": float(np.std(returns) * np.sqrt(252) * 100),  # anualizada e em percentual
                "volume_avg": float(np.mean(data["volumes"])) if "volumes" in data else None,
                "high": float(np.max(prices)),
                "low": float(np.min(prices)),
                "range_percent": float((np.max(prices) - np.min(prices)) / np.min(prices) * 100)
            })
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Market Analysis"),
            "type": "market_analysis",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "symbols": symbols,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "market_summary": market_summary,
            "has_sentiment_data": bool(sentiment_data),
            "has_correlation_matrix": bool(correlation_matrix),
            "has_technical_indicators": bool(technical_indicators)
        }
    
    async def _generate_risk_analysis(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise de risco.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Implementação básica - pode ser expandida com mais análises de risco
        strategy_id = parameters.get("strategy_id")
        time_range = parameters.get("time_range", "month")
        
        if not strategy_id:
            raise ValueError("ID da estratégia é obrigatório")
        
        # Reutilizar a lógica de análise de performance, mas com foco em métricas de risco
        parameters["include_risk_metrics"] = True
        
        return await self._generate_performance_summary(user_id, parameters)
    
    async def _generate_trade_analysis(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise de trades.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        strategy_id = parameters.get("strategy_id")
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        report_format = parameters.get("format", "pdf")
        
        if not strategy_id:
            raise ValueError("ID da estratégia é obrigatório")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados de trades
        trades_data = await self.data_warehouse.get_trades(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not trades_data or not trades_data["trades"]:
            raise ValueError(f"Não foi possível obter dados de trades para a estratégia {strategy_id}")
        
        # Calcular estatísticas de trades
        trades_summary = self._calculate_trades_summary(trades_data["trades"])
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de distribuição de P&L
            pnl_distribution_chart = self._generate_pnl_distribution_chart(trades_data["trades"])
            charts["pnl_distribution"] = pnl_distribution_chart
            
            # Gráfico de P&L ao longo do tempo
            pnl_over_time_chart = self._generate_pnl_over_time_chart(trades_data["trades"])
            charts["pnl_over_time"] = pnl_over_time_chart
            
            # Gráfico de win/loss por ativo
            win_loss_by_asset_chart = self._generate_win_loss_by_asset_chart(trades_data["trades"])
            charts["win_loss_by_asset"] = win_loss_by_asset_chart
            
            # Gráfico de duração de trades
            trade_duration_chart = self._generate_trade_duration_chart(trades_data["trades"])
            charts["trade_duration"] = trade_duration_chart
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"trade_analysis_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Trade Analysis"),
            trades_data=trades_data,
            trades_summary=trades_summary,
            charts=charts,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Trade Analysis"),
                "type": "trade_analysis",
                "parameters": parameters,
                "strategy_id": strategy_id,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        )
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Trade Analysis"),
            "type": "trade_analysis",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "strategy_id": strategy_id,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "trades_summary": trades_summary
        }
    
    async def _generate_portfolio_overview(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de visão geral do portfolio.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        include_charts = parameters.get("include_charts", True)
        include_active_strategies = parameters.get("include_active_strategies", True)
        include_exposure = parameters.get("include_exposure", True)
        report_format = parameters.get("format", "pdf")
        
        # Obter resumo do portfolio
        portfolio_summary = await self.analytics_service.get_portfolio_summary(user_id)
        
        # Obter estratégias ativas se solicitado
        active_strategies = None
        if include_active_strategies:
            active_strategies = await self.data_warehouse.get_active_strategies(user_id)
        
        # Obter análise de exposição se solicitado
        exposure_analysis = None
        if include_exposure:
            exposure_analysis = await self.analytics_service.get_exposure_analysis(user_id)
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de alocação de capital
            allocation_chart = self._generate_allocation_chart(portfolio_summary)
            charts["allocation"] = allocation_chart
            
            # Gráfico de exposição por ativo
            if exposure_analysis:
                exposure_chart = self._generate_exposure_chart(exposure_analysis)
                charts["exposure"] = exposure_chart
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"portfolio_overview_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Portfolio Overview"),
            portfolio_summary=portfolio_summary,
            active_strategies=active_strategies,
            exposure_analysis=exposure_analysis,
            charts=charts,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Portfolio Overview"),
                "type": "portfolio_overview",
                "parameters": parameters
            }
        )
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Portfolio Overview"),
            "type": "portfolio_overview",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "summary": {
                "total_capital": portfolio_summary["total_capital"],
                "allocated_capital": portfolio_summary["allocated_capital"],
                "available_capital": portfolio_summary["available_capital"],
                "total_profit_loss": portfolio_summary["total_profit_loss"],
                "total_profit_loss_percent": portfolio_summary["total_profit_loss_percent"],
                "active_strategies": portfolio_summary["active_strategies"]
            }
        }
    
    def _calculate_trades_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula estatísticas resumidas de trades."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "avg_trade_pnl": 0,
                "total_pnl": 0
            }
        
        # Separar trades vencedores e perdedores
        winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit_loss", 0) <= 0]
        
        # Calcular estatísticas
        total_trades = len(trades)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get("profit_loss", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("profit_loss", 0) for t in losing_trades))
        
        avg_profit = total_profit / winning_trades_count if winning_trades_count > 0 else 0
        avg_loss = total_loss / losing_trades_count if losing_trades_count > 0 else 0
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        total_pnl = sum(t.get("profit_loss", 0) for t in trades)
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        # Calcular duração média dos trades
        avg_duration = 0
        if all("entry_time" in t and "exit_time" in t for t in trades):
            durations = []
            for trade in trades:
                entry_time = trade["entry_time"]
                exit_time = trade["exit_time"]
                
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time)
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)
                    
                duration = (exit_time - entry_time).total_seconds() / 3600  # em horas
                durations.append(duration)
                
            avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calcular estatísticas por ativo
        assets = {}
        for trade in trades:
            asset = trade.get("symbol", "unknown")
            if asset not in assets:
                assets[asset] = {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pnl": 0
                }
            
            assets[asset]["total_trades"] += 1
            assets[asset]["total_pnl"] += trade.get("profit_loss", 0)
            
            if trade.get("profit_loss", 0) > 0:
                assets[asset]["winning_trades"] += 1
            else:
                assets[asset]["losing_trades"] += 1
        
        # Adicionar win rate para cada ativo
        for asset in assets:
            total = assets[asset]["total_trades"]
            winning = assets[asset]["winning_trades"]
            assets[asset]["win_rate"] = winning / total if total > 0 else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades_count,
            "losing_trades": losing_trades_count,
            "win_rate": win_rate,
            "avg_profit": float(avg_profit),
            "avg_loss": float(avg_loss),
            "profit_factor": float(profit_factor),
            "avg_trade_pnl": float(avg_trade_pnl),
            "total_pnl": float(total_pnl),
            "avg_duration_hours": float(avg_duration),
            "assets": assets
        }
    
    def _generate_cumulative_return_chart(
        self,
        returns: List[float],
        timestamps: List[datetime],
        benchmark_returns: Optional[List[float]] = None,
        benchmark_timestamps: Optional[List[datetime]] = None,
        strategy_name: str = "Strategy",
        benchmark_name: Optional[str] = None
    ) -> str:
        """Gera um gráfico de retorno cumulativo."""
        plt.figure(figsize=(12, 6))
        
        # Calcular retorno cumulativo para a estratégia
        cum_returns = np.cumprod(np.array(returns) + 1) - 1
        plt.plot(timestamps, cum_returns * 100, label=strategy_name, linewidth=2)
        
        # Adicionar benchmark se disponível
        if benchmark_returns and benchmark_timestamps:
            cum_bench_returns = np.cumprod(np.array(benchmark_returns) + 1) - 1
            plt.plot(benchmark_timestamps, cum_bench_returns * 100, label=benchmark_name or "Benchmark", 
                    linewidth=2, linestyle="--")
        
        plt.title("Cumulative Return (%)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Return (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_underwater_chart(self, underwater_data: Dict[str, Any]) -> str:
        """Gera um gráfico underwater (drawdown)."""
        plt.figure(figsize=(12, 6))
        
        # Converter timestamps de string para datetime
        timestamps = [datetime.fromisoformat(ts) for ts in underwater_data["timestamps"]]
        
        # Plotar o drawdown (multiplicar por 100 para percentual)
        plt.fill_between(timestamps, [0] * len(timestamps), 
                         np.array(underwater_data["underwater"]) * 100,
                         color="red", alpha=0.4)
        
        plt.plot(timestamps, np.array(underwater_data["underwater"]) * 100,
                color="darkred", linewidth=1)
        
        plt.title("Drawdown Over Time (%)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Drawdown (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adicionar linha horizontal para máximo drawdown
        max_dd = underwater_data["max_drawdown"] * 100
        plt.axhline(y=max_dd, color='r', linestyle='--', alpha=0.7, 
                   label=f"Max Drawdown: {max_dd:.2f}%")
        
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_returns_distribution_chart(
        self,
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        strategy_name: str = "Strategy",
        benchmark_name: Optional[str] = None
    ) -> str:
        """Gera um gráfico de distribuição de retornos."""
        plt.figure(figsize=(12, 6))
        
        # Converter para percentual
        returns_pct = np.array(returns) * 100
        
        # Plotar histograma da estratégia
        sns.histplot(returns_pct, kde=True, stat="density", label=strategy_name, alpha=0.6)
        
        # Adicionar benchmark se disponível
        if benchmark_returns:
            benchmark_returns_pct = np.array(benchmark_returns) * 100
            sns.histplot(benchmark_returns_pct, kde=True, stat="density", 
                        label=benchmark_name or "Benchmark", alpha=0.4)
        
        # Adicionar linha vertical para retorno médio
        mean_return = np.mean(returns_pct)
        plt.axvline(x=mean_return, color='green', linestyle='--', 
                   label=f"Mean Return: {mean_return:.2f}%")
        
        # Adicionar linha vertical para retorno zero
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        plt.title("Returns Distribution", fontsize=14)
        plt.xlabel("Daily Return (%)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_monthly_returns_chart(
        self,
        returns: List[float],
        timestamps: List[datetime]
    ) -> str:
        """Gera um gráfico de retornos mensais."""
        # Criar DataFrame com retornos
        df = pd.DataFrame({
            'date': timestamps,
            'return': returns
        })
        
        # Definir data como índice e resampling para retornos mensais
        df.set_index('date', inplace=True)
        monthly_returns = df.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Adicionar colunas de ano e mês
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        # Reshape para ano x mês
        pivot_table = monthly_returns.pivot_table(
            values='return', index='year', columns='month', aggfunc='sum'
        )
        
        # Plotar heatmap
        plt.figure(figsize=(12, 8))
        
        # Multiplicar por 100 para percentual
        pivot_table = pivot_table * 100
        
        # Definir mapa de cores - verde para positivo, vermelho para negativo
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        
        # Plotar heatmap
        sns.heatmap(pivot_table, cmap=cmap, center=0, annot=True, fmt=".2f",
                  cbar_kws={'label': 'Monthly Return (%)'})
        
        # Configurar nomes dos meses
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.xticks(np.arange(12) + 0.5, month_names, rotation=0)
        
        plt.title("Monthly Returns (%)", fontsize=14)
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_cumulative_return_comparison_chart(
        self,
        strategies_data: List[Dict[str, Any]],
        benchmark_data: Optional[Dict[str, Any]] = None,
        benchmark_name: Optional[str] = None
    ) -> str:
        """Gera um gráfico de comparação de retorno cumulativo entre estratégias."""
        plt.figure(figsize=(12, 6))
        
        # Plotar cada estratégia
        for strategy_data in strategies_data:
            returns = np.array(strategy_data["returns"])
            timestamps = strategy_data["timestamps"]
            strategy_name = strategy_data.get("strategy_name", f"Strategy {strategy_data['strategy_id']}")
            
            cum_returns = np.cumprod(returns + 1) - 1
            plt.plot(timestamps, cum_returns * 100, label=strategy_name, linewidth=2)
        
        # Adicionar benchmark se disponível
        if benchmark_data:
            returns = np.array(benchmark_data["returns"])
            timestamps = benchmark_data["timestamps"]
            
            cum_returns = np.cumprod(returns + 1) - 1
            plt.plot(timestamps, cum_returns * 100, label=benchmark_name or "Benchmark", 
                    linewidth=2, linestyle="--", color="black")
        
        plt.title("Cumulative Return Comparison (%)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Return (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_drawdown_comparison_chart(
        self,
        strategies_data: List[Dict[str, Any]],
        benchmark_data: Optional[Dict[str, Any]] = None,
        benchmark_name: Optional[str] = None
    ) -> str:
        """Gera um gráfico de comparação de drawdowns entre estratégias."""
        plt.figure(figsize=(12, 6))
        
        # Plotar cada estratégia
        for strategy_data in strategies_data:
            returns = np.array(strategy_data["returns"])
            timestamps = strategy_data["timestamps"]
            strategy_name = strategy_data.get("strategy_name", f"Strategy {strategy_data['strategy_id']}")
            
            # Calcular retorno cumulativo
            cum_returns = np.cumprod(returns + 1) - 1
            
            # Calcular drawdown
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / (1 + running_max)
            
            plt.plot(timestamps, drawdown * 100, label=strategy_name, linewidth=2)
        
        # Adicionar benchmark se disponível
        if benchmark_data:
            returns = np.array(benchmark_data["returns"])
            timestamps = benchmark_data["timestamps"]
            
            # Calcular retorno cumulativo
            cum_returns = np.cumprod(returns + 1) - 1
            
            # Calcular drawdown
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / (1 + running_max)
            
            plt.plot(timestamps, drawdown * 100, label=benchmark_name or "Benchmark", 
                    linewidth=2, linestyle="--", color="black")
        
        plt.title("Drawdown Comparison (%)", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Drawdown (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_metrics_comparison_chart(
        self,
        performance_analyses: List[Any]
    ) -> str:
        """Gera um gráfico de comparação de métricas entre estratégias."""
        # Definir métricas a comparar
        metrics = [
            ("Total Return", "TOTAL_RETURN"),
            ("Sharpe Ratio", "SHARPE_RATIO"),
            ("Max Drawdown", "MAX_DRAWDOWN"),
            ("Volatility", "VOLATILITY"),
            ("Sortino Ratio", "SORTINO_RATIO")
        ]
        
        # Extrair valores para cada estratégia
        strategies = []
        for analysis in performance_analyses:
            strategy_data = {
                "name": analysis.strategy_name,
                "metrics": {}
            }
            
            for metric_name, metric_key in metrics:
                metric_type = getattr(MetricType, metric_key)
                metric = analysis.metrics.get_metric(metric_type)
                
                if metric:
                    value = float(metric.value)
                    # Converter drawdown e volatilidade para positivos para visualização
                    if metric_key in ["MAX_DRAWDOWN", "VOLATILITY"]:
                        value = abs(value)
                    
                    strategy_data["metrics"][metric_name] = value
            
            strategies.append(strategy_data)
        
        # Criar DataFrame para plotagem
        df = pd.DataFrame({
            strategy["name"]: strategy["metrics"] for strategy in strategies
        })
        
        # Plotar gráfico de barras agrupadas
        plt.figure(figsize=(14, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(strategies)
        
        for i, strategy in enumerate(strategies):
            values = [strategy["metrics"].get(m[0], 0) for m in metrics]
            plt.bar(x + i * width - 0.4 + width/2, values, width, label=strategy["name"])
        
        plt.title("Strategy Metrics Comparison", fontsize=14)
        plt.xticks(x, [m[0] for m in metrics])
        plt.ylabel("Value")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_correlation_chart(
        self,
        strategies_data: List[Dict[str, Any]]
    ) -> str:
        """Gera um gráfico de correlação entre estratégias."""
        # Criar DataFrame com retornos de cada estratégia
        returns_data = {}
        
        # Usar a estratégia com mais dados como referência para o índice de tempo
        max_length_strategy = max(strategies_data, key=lambda x: len(x["returns"]))
        reference_timestamps = max_length_strategy["timestamps"]
        
        # Criar DataFrame com índice de tempo
        df = pd.DataFrame(index=reference_timestamps)
        
        # Adicionar retornos de cada estratégia
        for strategy_data in strategies_data:
            strategy_name = strategy_data.get("strategy_name", f"Strategy {strategy_data['strategy_id']}")
            
            # Criar Series com timestamp como índice
            s = pd.Series(strategy_data["returns"], index=strategy_data["timestamps"])
            
            # Reindexar para alinhar com o DataFrame de referência
            s = s.reindex(df.index)
            
            # Adicionar ao DataFrame
            df[strategy_name] = s
        
        # Calcular matriz de correlação
        corr_matrix = df.corr()
        
        # Plotar heatmap de correlação
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        
        plt.title("Strategy Returns Correlation", fontsize=14)
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_pnl_distribution_chart(self, trades: List[Dict[str, Any]]) -> str:
        """Gera um gráfico de distribuição de P&L dos trades."""
        pnl_values = [t.get("profit_loss", 0) for t in trades]
        
        plt.figure(figsize=(12, 6))
        
        # Plotar histograma de P&L
        sns.histplot(pnl_values, kde=True, stat="density", color="blue", alpha=0.6)
        
        # Adicionar linha vertical para P&L médio
        mean_pnl = np.mean(pnl_values)
        plt.axvline(x=mean_pnl, color='green', linestyle='--', 
                   label=f"Mean P&L: ${mean_pnl:.2f}")
        
        # Adicionar linha vertical para P&L zero
        plt.axvline(x=0, color='red', linestyle='-', alpha=0.5)
        
        plt.title("Trade P&L Distribution", fontsize=14)
        plt.xlabel("Profit/Loss ($)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_pnl_over_time_chart(self, trades: List[Dict[str, Any]]) -> str:
        """Gera um gráfico de P&L ao longo do tempo."""
        # Ordenar trades por data de fechamento
        sorted_trades = sorted(trades, key=lambda x: x.get("exit_time", x.get("entry_time", 0)))
        
        # Extrair timestamps e P&L
        timestamps = []
        pnl_values = []
        cumulative_pnl = []
        running_total = 0
        
        for trade in sorted_trades:
            exit_time = trade.get("exit_time")
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time)
                
            pnl = trade.get("profit_loss", 0)
            
            timestamps.append(exit_time)
            pnl_values.append(pnl)
            
            running_total += pnl
            cumulative_pnl.append(running_total)
        
        plt.figure(figsize=(12, 8))
        
        # Criar dois subplots
        ax1 = plt.subplot(2, 1, 1)  # P&L individual
        ax2 = plt.subplot(2, 1, 2)  # P&L cumulativo
        
        # Plotar P&L individual
        ax1.bar(timestamps, pnl_values, color=["green" if p > 0 else "red" for p in pnl_values],
               alpha=0.7)
        ax1.set_title("Individual Trade P&L", fontsize=12)
        ax1.set_ylabel("P&L ($)", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plotar P&L cumulativo
        ax2.plot(timestamps, cumulative_pnl, color="blue", linewidth=2)
        ax2.set_title("Cumulative P&L", fontsize=12)
        ax2.set_xlabel("Date", fontsize=10)
        ax2.set_ylabel("Cumulative P&L ($)", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_win_loss_by_asset_chart(self, trades: List[Dict[str, Any]]) -> str:
        """Gera um gráfico de win/loss por ativo."""
        # Agrupar trades por ativo
        assets = {}
        for trade in trades:
            asset = trade.get("symbol", "unknown")
            if asset not in assets:
                assets[asset] = {"win": 0, "loss": 0}
                
            if trade.get("profit_loss", 0) > 0:
                assets[asset]["win"] += 1
            else:
                assets[asset]["loss"] += 1
        
        # Converter para DataFrame
        df = pd.DataFrame({
            "Asset": list(assets.keys()),
            "Win": [assets[a]["win"] for a in assets],
            "Loss": [assets[a]["loss"] for a in assets]
        })
        
        # Ordenar por número total de trades
        df["Total"] = df["Win"] + df["Loss"]
        df = df.sort_values("Total", ascending=False).head(10)  # top 10 ativos por volume
        
        plt.figure(figsize=(12, 6))
        
        # Criar gráfico de barras empilhadas
        width = 0.8
        ax = plt.subplot(1, 1, 1)
        
        ax.bar(df["Asset"], df["Win"], width, label="Win", color="green", alpha=0.7)
        ax.bar(df["Asset"], df["Loss"], width, bottom=df["Win"], label="Loss", color="red", alpha=0.7)
        
        # Adicionar win rate como texto
        for i, asset in enumerate(df["Asset"]):
            win = df.iloc[i]["Win"]
            total = df.iloc[i]["Total"]
            win_rate = win / total if total > 0 else 0
            
            ax.text(i, win + df.iloc[i]["Loss"] + 0.5, f"{win_rate*100:.1f}%",
                  ha="center", va="bottom", fontsize=9)
        
        ax.set_title("Win/Loss by Asset", fontsize=14)
        ax.set_xlabel("Asset", fontsize=12)
        ax.set_ylabel("Number of Trades", fontsize=12)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_trade_duration_chart(self, trades: List[Dict[str, Any]]) -> str:
        """Gera um gráfico de duração dos trades."""
        durations = []
        win_durations = []
        loss_durations = []
        
        for trade in trades:
            entry_time = trade.get("entry_time")
            exit_time = trade.get("exit_time")
            
            if not entry_time or not exit_time:
                continue
                
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time)
            if isinstance(exit_time, str):
                exit_time = datetime.fromisoformat(exit_time)
                
            # Duração em horas
            duration = (exit_time - entry_time).total_seconds() / 3600
            durations.append(duration)
            
            if trade.get("profit_loss", 0) > 0:
                win_durations.append(duration)
            else:
                loss_durations.append(duration)
        
        plt.figure(figsize=(12, 6))
        
        # Plotar histograma de duração
        bins = np.linspace(0, max(durations) * 1.1, 20)
        
        plt.hist(win_durations, bins=bins, alpha=0.6, color="green", label="Winning Trades")
        plt.hist(loss_durations, bins=bins, alpha=0.6, color="red", label="Losing Trades")
        
        plt.title("Trade Duration Distribution", fontsize=14)
        plt.xlabel("Duration (hours)", fontsize=12)
        plt.ylabel("Number of Trades", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_allocation_chart(self, portfolio_summary: Dict[str, Any]) -> str:
        """Gera um gráfico de alocação de capital."""
        plt.figure(figsize=(10, 6))
        
        # Dados de alocação
        allocated = portfolio_summary["allocated_capital"]
        available = portfolio_summary["available_capital"]
        
        # Criar gráfico de pizza
        plt.pie([allocated, available], 
               labels=["Allocated", "Available"],
               autopct='%1.1f%%',
               startangle=90,
               colors=["#3498db", "#95a5a6"],
               explode=(0.1, 0))
        
        plt.title("Capital Allocation", fontsize=14)
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _generate_exposure_chart(self, exposure_analysis: Dict[str, Any]) -> str:
        """Gera um gráfico de exposição por ativo."""
        plt.figure(figsize=(12, 6))
        
        # Dados de exposição por ativo
        assets = exposure_analysis.get("exposure_by_asset", {})
        asset_names = list(assets.keys())
        exposures = list(assets.values())
        
        # Ordenar por exposição (decrescente)
        sorted_data = sorted(zip(asset_names, exposures), key=lambda x: x[1], reverse=True)
        asset_names, exposures = zip(*sorted_data) if sorted_data else ([], [])
        
        # Criar gráfico de barras
        plt.bar(asset_names, exposures, color=plt.cm.viridis(np.linspace(0, 0.9, len(asset_names))))
        
        plt.title("Exposure by Asset", fontsize=14)
        plt.xlabel("Asset", fontsize=12)
        plt.ylabel("Exposure ($)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        
        # Converter para base64
        buffer = BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        plt.close()
        
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return f"data:image/png;base64,{image_base64}"
    
    def _calculate_correlation_matrix(self, market_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula a matriz de correlação entre diferentes símbolos."""
        # Criar DataFrame com retornos para cada símbolo
        symbols = []
        all_returns = {}
        all_timestamps = []
        
        # Coletar timestamps e retornos
        for data in market_data:
            symbol = data["symbol"]
            symbols.append(symbol)
            all_returns[symbol] = data["returns"]
            
            if not all_timestamps:
                all_timestamps = data["timestamps"]
        
        # Criar DataFrame
        df = pd.DataFrame(index=all_timestamps)
        
        # Adicionar retornos para cada símbolo
        for symbol in symbols:
            df[symbol] = all_returns[symbol]
        
        # Calcular matriz de correlação
        correlation = df.corr().round(3).to_dict()
        
        return {
            "correlation_matrix": correlation,
            "symbols": symbols
        }
    
    def _calculate_technical_indicators(
        self,
        market_data: Dict[str, Any],
        indicators: List[str]
    ) -> Dict[str, Any]:
        """Calcula indicadores técnicos para os dados de mercado."""
        result = {
            "symbol": market_data["symbol"],
            "indicators": {}
        }
        
        # Obter dados OHLCV
        prices = np.array(market_data["prices"])
        timestamps = market_data["timestamps"]
        
        # Obter high/low/open/close/volume se disponíveis
        high = np.array(market_data.get("highs", prices))
        low = np.array(market_data.get("lows", prices))
        open_prices = np.array(market_data.get("opens", prices))
        close_prices = np.array(market_data.get("closes", prices))
        volumes = np.array(market_data.get("volumes", [0] * len(prices)))
        
        # Calcular indicadores solicitados
        for indicator in indicators:
            if indicator.lower() == "sma":
                # Simple Moving Average - 20 períodos
                window = 20
                if len(prices) >= window:
                    sma = np.convolve(prices, np.ones(window)/window, mode='valid')
                    # Preencher valores iniciais com NaN
                    sma_full = np.full_like(prices, np.nan)
                    sma_full[window-1:] = sma
                    
                    result["indicators"]["sma_20"] = {
                        "values": sma_full.tolist(),
                        "timestamps": timestamps
                    }
                    
            elif indicator.lower() == "ema":
                # Exponential Moving Average - 20 períodos
                window = 20
                if len(prices) >= window:
                    # Fórmula simplificada de EMA
                    alpha = 2 / (window + 1)
                    ema = np.zeros_like(prices)
                    ema[0] = prices[0]
                    
                    for i in range(1, len(prices)):
                        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                    
                    result["indicators"]["ema_20"] = {
                        "values": ema.tolist(),
                        "timestamps": timestamps
                    }
                    
            elif indicator.lower() == "rsi":
                # Relative Strength Index - 14 períodos
                window = 14
                if len(prices) > window:
                    # Calcular mudanças nos preços
                    delta = np.zeros_like(prices)
                    delta[1:] = prices[1:] - prices[:-1]
                    
                    # Separar ganhos e perdas
                    gain = delta.copy()
                    loss = delta.copy()
                    gain[gain < 0] = 0
                    loss[loss > 0] = 0
                    loss = abs(loss)
                    
                    # Calcular médias móveis
                    avg_gain = np.zeros_like(prices)
                    avg_loss = np.zeros_like(prices)
                    
                    # Inicializar primeiras médias
                    avg_gain[window] = np.mean(gain[1:window+1])
                    avg_loss[window] = np.mean(loss[1:window+1])
                    
                    # Calcular médias móveis subsequentes
                    for i in range(window+1, len(prices)):
                        avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
                        avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
                    
                    # Calcular RS e RSI
                    rs = np.zeros_like(prices)
                    rsi = np.zeros_like(prices)
                    
                    for i in range(window, len(prices)):
                        if avg_loss[i] == 0:
                            rsi[i] = 100
                        else:
                            rs[i] = avg_gain[i] / avg_loss[i]
                            rsi[i] = 100 - (100 / (1 + rs[i]))
                    
                    result["indicators"]["rsi_14"] = {
                        "values": rsi.tolist(),
                        "timestamps": timestamps
                    }
                    
            elif indicator.lower() == "bollinger":
                # Bollinger Bands - 20 períodos, 2 desvios padrão
                window = 20
                std_dev = 2
                if len(prices) >= window:
                    # Calcular SMA
                    sma = np.zeros_like(prices)
                    
                    for i in range(window-1, len(prices)):
                        sma[i] = np.mean(prices[i-(window-1):i+1])
                    
                    # Calcular desvio padrão móvel
                    mstd = np.zeros_like(prices)
                    
                    for i in range(window-1, len(prices)):
                        mstd[i] = np.std(prices[i-(window-1):i+1], ddof=1)
                    
                    # Calcular bandas
                    upper_band = sma + std_dev * mstd
                    lower_band = sma - std_dev * mstd
                    
                    result["indicators"]["bollinger_bands"] = {
                        "middle": sma.tolist(),
                        "upper": upper_band.tolist(),
                        "lower": lower_band.tolist(),
                        "timestamps": timestamps
                    }
                    
            elif indicator.lower() == "macd":
                # MACD - 12, 26, 9
                fast = 12
                slow = 26
                signal = 9
                
                if len(prices) >= slow + signal:
                    # Calcular EMA rápido
                    alpha_fast = 2 / (fast + 1)
                    ema_fast = np.zeros_like(prices)
                    ema_fast[0] = prices[0]
                    
                    for i in range(1, len(prices)):
                        ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
                    
                    # Calcular EMA lento
                    alpha_slow = 2 / (slow + 1)
                    ema_slow = np.zeros_like(prices)
                    ema_slow[0] = prices[0]
                    
                    for i in range(1, len(prices)):
                        ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
                    
                    # Calcular MACD
                    macd = ema_fast - ema_slow
                    
                    # Calcular linha de sinal
                    alpha_signal = 2 / (signal + 1)
                    signal_line = np.zeros_like(prices)
                    signal_line[slow-1] = macd[slow-1]
                    
                    for i in range(slow, len(prices)):
                        signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i-1]
                    
                    # Calcular histograma
                    histogram = macd - signal_line
                    
                    result["indicators"]["macd"] = {
                        "macd": macd.tolist(),
                        "signal": signal_line.tolist(),
                        "histogram": histogram.tolist(),
                        "timestamps": timestamps
                    }
        
        return result
    
    async def _publish_report(
        self,
        user_id: str,
        report_id: str,
        report_name: str,
        format: str = "pdf",
        filename: str = "report",
        **report_data
    ) -> str:
        """
        Publica um relatório no formato especificado.
        
        Args:
            user_id: ID do usuário
            report_id: ID do relatório
            report_name: Nome do relatório
            format: Formato do relatório (pdf, html, json)
            filename: Nome do arquivo
            **report_data: Dados adicionais para o relatório
            
        Returns:
            URL do relatório
        """
        # Esta é uma implementação simplificada que simula a geração de um relatório
        # Em uma implementação real, aqui seria feita a geração do relatório no formato
        # solicitado e seu armazenamento em um local acessível por URL
        
        # Publicar o relatório usando o dashboard publisher
        report_url = await self.dashboard_publisher.publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=report_name,
            report_data=report_data,
            format=format,
            filename=filename
        )
        
        return report_url
        """
Caso de uso para geração de relatórios de análise.

Este módulo implementa o caso de uso para gerar diferentes
tipos de relatórios analíticos para estratégias de trading.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import os
import tempfile

from src.common.application.base_use_case import BaseUseCase
from src.analytics.application.services.performance_attribution_service import PerformanceAttributionService
from src.analytics.application.services.analytics_service import AnalyticsService
from src.analytics.infrastructure.dashboard_publisher import DashboardPublisher
from src.analytics.infrastructure.data_warehouse import DataWarehouse


class GenerateAnalyticsReportUseCase(BaseUseCase):
    """Caso de uso para gerar relatórios de análise."""
    
    def __init__(
        self,
        performance_service: PerformanceAttributionService,
        analytics_service: AnalyticsService,
        data_warehouse: DataWarehouse,
        dashboard_publisher: DashboardPublisher,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o caso de uso.
        
        Args:
            performance_service: Serviço de atribuição de performance
            analytics_service: Serviço de analytics
            data_warehouse: Repositório de dados
            dashboard_publisher: Publicador de dashboards
            logger: Logger opcional
        """
        super().__init__(logger)
        self.performance_service = performance_service
        self.analytics_service = analytics_service
        self.data_warehouse = data_warehouse
        self.dashboard_publisher = dashboard_publisher
    
    async def execute(
        self,
        user_id: str,
        report_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executa o caso de uso.
        
        Args:
            user_id: ID do usuário
            report_type: Tipo de relatório
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        self.logger.info(f"Gerando relatório do tipo '{report_type}' para usuário {user_id}")
        
        # Verificar o tipo de relatório
        if report_type == "performance_summary":
            return await self._generate_performance_summary(user_id, parameters)
        elif report_type == "strategy_comparison":
            return await self._generate_strategy_comparison(user_id, parameters)
        elif report_type == "market_analysis":
            return await self._generate_market_analysis(user_id, parameters)
        elif report_type == "risk_analysis":
            return await self._generate_risk_analysis(user_id, parameters)
        elif report_type == "trade_analysis":
            return await self._generate_trade_analysis(user_id, parameters)
        elif report_type == "portfolio_overview":
            return await self._generate_portfolio_overview(user_id, parameters)
        else:
            raise ValueError(f"Tipo de relatório não reconhecido: {report_type}")
    
    async def _generate_performance_summary(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de resumo de performance.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        strategy_id = parameters.get("strategy_id")
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        include_benchmark = parameters.get("include_benchmark", False)
        benchmark_symbol = parameters.get("benchmark_symbol", "BTC/USD")
        report_format = parameters.get("format", "pdf")
        
        if not strategy_id:
            raise ValueError("ID da estratégia é obrigatório")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados da estratégia
        strategy_data = await self.data_warehouse.get_strategy_returns(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not strategy_data or "returns" not in strategy_data:
            raise ValueError(f"Não foi possível obter dados para a estratégia {strategy_id}")
        
        # Obter dados de benchmark se solicitado
        benchmark_data = None
        if include_benchmark:
            benchmark_data = await self.data_warehouse.get_market_data(
                symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
        
        # Calcular métricas de performance
        performance_analysis = await self.performance_service.calculate_performance_metrics(
            returns=strategy_data["returns"],
            timestamps=strategy_data["timestamps"],
            benchmark_returns=benchmark_data["returns"] if benchmark_data else None,
            strategy_id=strategy_id,
            strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
            initial_capital=strategy_data.get("initial_capital", 10000.0),
            trades_data=strategy_data.get("trades", [])
        )
        
        # Analisar drawdowns
        drawdowns = await self.performance_service.analyze_drawdowns(
            returns=strategy_data["returns"],
            timestamps=strategy_data["timestamps"],
            threshold=0.01,  # 1%
            max_drawdowns=5
        )
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de retorno cumulativo
            cumulative_return_chart = self._generate_cumulative_return_chart(
                strategy_data["returns"],
                strategy_data["timestamps"],
                benchmark_data["returns"] if benchmark_data else None,
                benchmark_data["timestamps"] if benchmark_data else None,
                strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["cumulative_return"] = cumulative_return_chart
            
            # Gráfico underwater (drawdown)
            underwater_data = await self.performance_service.get_underwater_chart_data(
                strategy_data["returns"],
                strategy_data["timestamps"]
            )
            underwater_chart = self._generate_underwater_chart(underwater_data)
            charts["underwater"] = underwater_chart
            
            # Gráfico de distribuição de retornos
            returns_distribution_chart = self._generate_returns_distribution_chart(
                strategy_data["returns"],
                benchmark_data["returns"] if benchmark_data else None,
                strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["returns_distribution"] = returns_distribution_chart
            
            # Gráfico de retornos mensais
            monthly_returns_chart = self._generate_monthly_returns_chart(
                strategy_data["returns"],
                strategy_data["timestamps"]
            )
            charts["monthly_returns"] = monthly_returns_chart
        
        # Obter dados de trades se disponíveis
        trades_summary = {}
        if "trades" in strategy_data and strategy_data["trades"]:
            trades_summary = self._calculate_trades_summary(strategy_data["trades"])
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"performance_summary_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Performance Summary"),
            performance_analysis=performance_analysis,
            drawdowns=drawdowns,
            charts=charts,
            trades_summary=trades_summary,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Performance Summary"),
                "type": "performance_summary",
                "parameters": parameters,
                "strategy_id": strategy_id,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "include_benchmark": include_benchmark,
                "benchmark_symbol": benchmark_symbol if include_benchmark else None
            }
        )
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Performance Summary"),
            "type": "performance_summary",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "strategy_id": strategy_id,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "summary": {
                "total_return": performance_analysis.metrics.get_metric(MetricType.TOTAL_RETURN).formatted_value,
                "annualized_return": performance_analysis.metrics.get_metric(MetricType.ANNUALIZED_RETURN).formatted_value,
                "sharpe_ratio": performance_analysis.metrics.get_metric(MetricType.SHARPE_RATIO).formatted_value,
                "max_drawdown": performance_analysis.metrics.get_metric(MetricType.MAX_DRAWDOWN).formatted_value,
                "win_rate": f"{performance_analysis.win_rate * 100:.2f}%",
                "profit_loss": f"${performance_analysis.profit_loss:.2f}"
            }
        }
    
    async def _generate_strategy_comparison(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de comparação de estratégias.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        strategy_ids = parameters.get("strategy_ids", [])
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        include_benchmark = parameters.get("include_benchmark", False)
        benchmark_symbol = parameters.get("benchmark_symbol", "BTC/USD")
        report_format = parameters.get("format", "pdf")
        
        if not strategy_ids or len(strategy_ids) < 2:
            raise ValueError("Pelo menos duas estratégias devem ser especificadas para comparação")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados para cada estratégia
        strategies_data = []
        for strategy_id in strategy_ids:
            strategy_data = await self.data_warehouse.get_strategy_returns(
                strategy_id=strategy_id,
                start_date=start_date,
                end_date=end_date
            )
            
            if not strategy_data or "returns" not in strategy_data:
                self.logger.warning(f"Não foi possível obter dados para a estratégia {strategy_id}")
                continue
                
            strategies_data.append(strategy_data)
        
        if not strategies_data:
            raise ValueError("Não foi possível obter dados para nenhuma das estratégias especificadas")
        
        # Obter dados de benchmark se solicitado
        benchmark_data = None
        if include_benchmark:
            benchmark_data = await self.data_warehouse.get_market_data(
                symbol=benchmark_symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
        
        # Calcular métricas de performance para cada estratégia
        performance_analyses = []
        for strategy_data in strategies_data:
            analysis = await self.performance_service.calculate_performance_metrics(
                returns=strategy_data["returns"],
                timestamps=strategy_data["timestamps"],
                benchmark_returns=benchmark_data["returns"] if benchmark_data else None,
                strategy_id=strategy_data["strategy_id"],
                strategy_name=strategy_data.get("strategy_name", f"Strategy {strategy_data['strategy_id']}"),
                initial_capital=strategy_data.get("initial_capital", 10000.0),
                trades_data=strategy_data.get("trades", [])
            )
            performance_analyses.append(analysis)
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de comparação de retorno cumulativo
            cumulative_return_chart = self._generate_cumulative_return_comparison_chart(
                strategies_data,
                benchmark_data=benchmark_data if include_benchmark else None,
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["cumulative_return_comparison"] = cumulative_return_chart
            
            # Gráfico de comparação de drawdowns
            drawdown_comparison_chart = self._generate_drawdown_comparison_chart(
                strategies_data,
                benchmark_data=benchmark_data if include_benchmark else None,
                benchmark_name=benchmark_symbol if include_benchmark else None
            )
            charts["drawdown_comparison"] = drawdown_comparison_chart
            
            # Gráfico de comparação de métricas
            metrics_comparison_chart = self._generate_metrics_comparison_chart(performance_analyses)
            charts["metrics_comparison"] = metrics_comparison_chart
            
            # Gráfico de correlação entre estratégias
            correlation_chart = self._generate_correlation_chart(strategies_data)
            charts["correlation"] = correlation_chart
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"strategy_comparison_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Strategy Comparison"),
            performance_analyses=performance_analyses,
            charts=charts,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Strategy Comparison"),
                "type": "strategy_comparison",
                "parameters": parameters,
                "strategy_ids": strategy_ids,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "include_benchmark": include_benchmark,
                "benchmark_symbol": benchmark_symbol if include_benchmark else None
            }
        )
        
        # Criar resumo comparativo
        comparison_summary = []
        for analysis in performance_analyses:
            comparison_summary.append({
                "strategy_id": analysis.strategy_id,
                "strategy_name": analysis.strategy_name,
                "total_return": analysis.metrics.get_metric(MetricType.TOTAL_RETURN).formatted_value,
                "annualized_return": analysis.metrics.get_metric(MetricType.ANNUALIZED_RETURN).formatted_value,
                "sharpe_ratio": analysis.metrics.get_metric(MetricType.SHARPE_RATIO).formatted_value,
                "max_drawdown": analysis.metrics.get_metric(MetricType.MAX_DRAWDOWN).formatted_value,
                "win_rate": f"{analysis.win_rate * 100:.2f}%",
                "profit_loss": f"${analysis.profit_loss:.2f}"
            })
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Strategy Comparison"),
            "type": "strategy_comparison",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "strategy_ids": strategy_ids,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "comparison_summary": comparison_summary
        }
    
    async def _generate_market_analysis(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise de mercado.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        symbols = parameters.get("symbols", [])
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_sentiment = parameters.get("include_sentiment", False)
        include_correlations = parameters.get("include_correlations", True)
        include_indicators = parameters.get("include_indicators", True)
        indicators = parameters.get("indicators", ["rsi", "macd", "bollinger"])
        report_format = parameters.get("format", "pdf")
        
        if not symbols:
            raise ValueError("Pelo menos um símbolo deve ser especificado para análise")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados de mercado para cada símbolo
        market_data = []
        for symbol in symbols:
            data = await self.data_warehouse.get_market_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval="1d"
            )
            
            if not data:
                self.logger.warning(f"Não foi possível obter dados para o símbolo {symbol}")
                continue
                
            market_data.append(data)
        
        if not market_data:
            raise ValueError("Não foi possível obter dados para nenhum dos símbolos especificados")
        
        # Obter dados de sentimento se solicitado
        sentiment_data = None
        if include_sentiment:
            sentiment_data = []
            for symbol in symbols:
                data = await self.data_warehouse.get_sentiment_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if data:
                    sentiment_data.append(data)
        
        # Calcular correlações se solicitado
        correlation_matrix = None
        if include_correlations and len(symbols) > 1:
            correlation_matrix = self._calculate_correlation_matrix(market_data)
        
        # Calcular indicadores técnicos se solicitado
        technical_indicators = None
        if include_indicators:
            technical_indicators = []
            for data in market_data:
                indicators_data = self._calculate_technical_indicators(
                    data,
                    indicators=indicators
                )
                technical_indicators.append(indicators_data)
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"market_analysis_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Market Analysis"),
            market_data=market_data,
            sentiment_data=sentiment_data,
            correlation_matrix=correlation_matrix,
            technical_indicators=technical_indicators,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Market Analysis"),
                "type": "market_analysis",
                "parameters": parameters,
                "symbols": symbols,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "include_sentiment": include_sentiment,
                "include_correlations": include_correlations,
                "include_indicators": include_indicators,
                "indicators": indicators
            }
        )
        
        # Criar resumo de mercado
        market_summary = []
        for data in market_data:
            returns = np.array(data["returns"])
            prices = np.array(data["prices"])
            
            market_summary.append({
                "symbol": data["symbol"],
                "start_price": float(prices[0]),
                "end_price": float(prices[-1]),
                "change_percent": float((prices[-1] / prices[0] - 1) * 100),
                "volatility": float(np.std(returns) * np.sqrt(252) * 100),  # anualizada e em percentual
                "volume_avg": float(np.mean(data["volumes"])) if "volumes" in data else None,
                "high": float(np.max(prices)),
                "low": float(np.min(prices)),
                "range_percent": float((np.max(prices) - np.min(prices)) / np.min(prices) * 100)
            })
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Market Analysis"),
            "type": "market_analysis",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "symbols": symbols,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "market_summary": market_summary,
            "has_sentiment_data": bool(sentiment_data),
            "has_correlation_matrix": bool(correlation_matrix),
            "has_technical_indicators": bool(technical_indicators)
        }
    
    async def _generate_risk_analysis(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise de risco.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Implementação básica - pode ser expandida com mais análises de risco
        strategy_id = parameters.get("strategy_id")
        time_range = parameters.get("time_range", "month")
        
        if not strategy_id:
            raise ValueError("ID da estratégia é obrigatório")
        
        # Reutilizar a lógica de análise de performance, mas com foco em métricas de risco
        parameters["include_risk_metrics"] = True
        
        return await self._generate_performance_summary(user_id, parameters)
    
    async def _generate_trade_analysis(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de análise de trades.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        strategy_id = parameters.get("strategy_id")
        time_range = parameters.get("time_range", "month")
        start_date = parameters.get("start_date")
        end_date = parameters.get("end_date")
        include_charts = parameters.get("include_charts", True)
        report_format = parameters.get("format", "pdf")
        
        if not strategy_id:
            raise ValueError("ID da estratégia é obrigatório")
        
        # Converter datas se fornecidas como strings
        if start_date and isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if end_date and isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Determinar período de análise se não especificado
        if not start_date:
            if time_range == "day":
                start_date = datetime.utcnow() - timedelta(days=1)
            elif time_range == "week":
                start_date = datetime.utcnow() - timedelta(days=7)
            elif time_range == "month":
                start_date = datetime.utcnow() - timedelta(days=30)
            elif time_range == "quarter":
                start_date = datetime.utcnow() - timedelta(days=90)
            elif time_range == "year":
                start_date = datetime.utcnow() - timedelta(days=365)
            else:
                start_date = datetime.utcnow() - timedelta(days=30)  # padrão: 30 dias
        
        if not end_date:
            end_date = datetime.utcnow()
        
        # Obter dados de trades
        trades_data = await self.data_warehouse.get_trades(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not trades_data or not trades_data["trades"]:
            raise ValueError(f"Não foi possível obter dados de trades para a estratégia {strategy_id}")
        
        # Calcular estatísticas de trades
        trades_summary = self._calculate_trades_summary(trades_data["trades"])
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de distribuição de P&L
            pnl_distribution_chart = self._generate_pnl_distribution_chart(trades_data["trades"])
            charts["pnl_distribution"] = pnl_distribution_chart
            
            # Gráfico de P&L ao longo do tempo
            pnl_over_time_chart = self._generate_pnl_over_time_chart(trades_data["trades"])
            charts["pnl_over_time"] = pnl_over_time_chart
            
            # Gráfico de win/loss por ativo
            win_loss_by_asset_chart = self._generate_win_loss_by_asset_chart(trades_data["trades"])
            charts["win_loss_by_asset"] = win_loss_by_asset_chart
            
            # Gráfico de duração de trades
            trade_duration_chart = self._generate_trade_duration_chart(trades_data["trades"])
            charts["trade_duration"] = trade_duration_chart
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"trade_analysis_{strategy_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Trade Analysis"),
            trades_data=trades_data,
            trades_summary=trades_summary,
            charts=charts,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Trade Analysis"),
                "type": "trade_analysis",
                "parameters": parameters,
                "strategy_id": strategy_id,
                "time_range": time_range,
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None
            }
        )
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Trade Analysis"),
            "type": "trade_analysis",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "strategy_id": strategy_id,
            "time_range": time_range,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "trades_summary": trades_summary
        }
    
    async def _generate_portfolio_overview(
        self,
        user_id: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Gera um relatório de visão geral do portfolio.
        
        Args:
            user_id: ID do usuário
            parameters: Parâmetros para geração do relatório
            
        Returns:
            Resultado da geração do relatório
        """
        # Extrair parâmetros
        include_charts = parameters.get("include_charts", True)
        include_active_strategies = parameters.get("include_active_strategies", True)
        include_exposure = parameters.get("include_exposure", True)
        report_format = parameters.get("format", "pdf")
        
        # Obter resumo do portfolio
        portfolio_summary = await self.analytics_service.get_portfolio_summary(user_id)
        
        # Obter estratégias ativas se solicitado
        active_strategies = None
        if include_active_strategies:
            active_strategies = await self.data_warehouse.get_active_strategies(user_id)
        
        # Obter análise de exposição se solicitado
        exposure_analysis = None
        if include_exposure:
            exposure_analysis = await self.analytics_service.get_exposure_analysis(user_id)
        
        # Gerar gráficos se solicitado
        charts = {}
        if include_charts:
            # Gráfico de alocação de capital
            allocation_chart = self._generate_allocation_chart(portfolio_summary)
            charts["allocation"] = allocation_chart
            
            # Gráfico de exposição por ativo
            if exposure_analysis:
                exposure_chart = self._generate_exposure_chart(exposure_analysis)
                charts["exposure"] = exposure_chart
        
        # Gerar o relatório no formato solicitado
        report_id = str(uuid.uuid4())
        report_filename = f"portfolio_overview_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        report_url = await self._publish_report(
            user_id=user_id,
            report_id=report_id,
            report_name=parameters.get("report_name", "Portfolio Overview"),
            portfolio_summary=portfolio_summary,
            active_strategies=active_strategies,
            exposure_analysis=exposure_analysis,
            charts=charts,
            parameters=parameters,
            format=report_format,
            filename=report_filename
        )
        
        # Salvar o relatório no banco de dados
        await self.analytics_service.generate_report(
            user_id=user_id,
            request={
                "name": parameters.get("report_name", "Portfolio Overview"),
                "type": "portfolio_overview",
                "parameters": parameters
            }
        )
        
        return {
            "id": report_id,
            "name": parameters.get("report_name", "Portfolio Overview"),
            "type": "portfolio_overview",
            "created_at": datetime.utcnow().isoformat(),
            "url": report_url,
            "summary": {
                "total_capital": portfolio_summary["total_capital"],
                "allocated_capital": portfolio_summary["allocated_capital"],
                "available_capital": portfolio_summary["available_capital"],
                "total_profit_loss": portfolio_summary["total_profit_loss"],
                "total_profit_loss_percent": portfolio_summary["total_profit_loss_percent"],
                "active_strategies": portfolio_summary["active_strategies"]
            }
        }
    
    def _calculate_trades_summary(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula estatísticas resumidas de trades."""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_profit": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "avg_trade_pnl": 0,
                "total_pnl": 0
            }
        
        # Separar trades vencedores e perdedores
        winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
        losing_trades = [t for t in trades if t.get("profit_loss", 0) <= 0]
        
        # Calcular estatísticas
        total_trades = len(trades)
        winning_trades_count = len(winning_trades)
        losing_trades_count = len(losing_trades)
        
        win_rate = winning_trades_count / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t.get("profit_loss", 0) for t in winning_trades)
        total_loss = abs(sum(t.get("profit_loss", 0) for t in losing_trades))
        
        avg_profit = total_profit / winning_trades_count if winning_trades_count > 0