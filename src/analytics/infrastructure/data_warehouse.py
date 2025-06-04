"""
Componente de infraestrutura para acesso ao data warehouse.

Este módulo fornece acesso aos dados armazenados no data warehouse,
incluindo dados de mercado, estratégias, trades e métricas.
"""
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import numpy as np
import pandas as pd
import json
import uuid
from decimal import Decimal

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
import arctic
from arctic import Arctic, CHUNK_STORE, VERSION_STORE
from pymongo import MongoClient
import motor.motor_asyncio
import redis.asyncio as redis

from src.common.infrastructure.base_repository import BaseRepository
from src.common.infrastructure.config import DatabaseConfig
from src.analytics.domain.value_objects.performance_metric import (
    PerformanceMetric, PerformanceMetricCollection, PerformanceAnalysis, MetricType
)


class DataWarehouse(BaseRepository):
    """
    Repositório para acesso ao data warehouse.
    
    Este componente centraliza o acesso a diferentes fontes de dados,
    como SQL, Arctic/MongoDB e Redis, fornecendo uma interface unificada
    para consultas analíticas.
    """
    
    def __init__(
        self,
        config: DatabaseConfig,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o data warehouse.
        
        Args:
            config: Configuração de conexão com banco de dados
            logger: Logger opcional
        """
        super().__init__(logger)
        self.config = config
        self._sql_engine = None
        self._async_sql_engine = None
        self._arctic_store = None
        self._mongo_client = None
        self._async_mongo_client = None
        self._redis_client = None
        
        # Inicializar conexões
        self._init_connections()
    
    def _init_connections(self) -> None:
        """Inicializa as conexões com os bancos de dados."""
        # Conexão SQL para PostgreSQL
        if self.config.sql_uri:
            try:
                self._sql_engine = create_engine(self.config.sql_uri)
                self._async_sql_engine = create_async_engine(
                    self.config.sql_uri.replace("postgresql://", "postgresql+asyncpg://")
                )
                self.logger.info("Conexão SQL inicializada com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar conexão SQL: {e}")
        
        # Conexão Arctic para séries temporais
        if self.config.arctic_uri:
            try:
                self._mongo_client = MongoClient(self.config.arctic_uri)
                self._arctic_store = Arctic(self._mongo_client)
                
                # Inicializar bibliotecas Arctic se não existirem
                self._init_arctic_libraries()
                
                self.logger.info("Conexão Arctic inicializada com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar conexão Arctic: {e}")
        
        # Conexão MongoDB assíncrona
        if self.config.mongo_uri:
            try:
                self._async_mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
                    self.config.mongo_uri
                )
                self.logger.info("Conexão MongoDB assíncrona inicializada com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar conexão MongoDB assíncrona: {e}")
        
        # Conexão Redis
        if self.config.redis_uri:
            try:
                self._redis_client = redis.from_url(self.config.redis_uri)
                asyncio.create_task(self._test_redis_connection())
                self.logger.info("Conexão Redis inicializada com sucesso")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar conexão Redis: {e}")
    
    async def _test_redis_connection(self) -> None:
        """Testa a conexão com o Redis."""
        if self._redis_client:
            try:
                await self._redis_client.ping()
                self.logger.info("Teste de conexão Redis bem-sucedido")
            except Exception as e:
                self.logger.error(f"Erro ao testar conexão Redis: {e}")
    
    def _init_arctic_libraries(self) -> None:
        """Inicializa as bibliotecas Arctic necessárias."""
        if not self._arctic_store:
            return
            
        # Bibliotecas para dados de mercado
        if not self._arctic_store.library_exists('MARKET_DATA'):
            self._arctic_store.initialize_library('MARKET_DATA', lib_type=CHUNK_STORE)
            self.logger.info("Biblioteca Arctic 'MARKET_DATA' inicializada")
            
        # Biblioteca para dados de estratégias
        if not self._arctic_store.library_exists('STRATEGY_DATA'):
            self._arctic_store.initialize_library('STRATEGY_DATA', lib_type=CHUNK_STORE)
            self.logger.info("Biblioteca Arctic 'STRATEGY_DATA' inicializada")
            
        # Biblioteca para métricas de performance
        if not self._arctic_store.library_exists('PERFORMANCE_METRICS'):
            self._arctic_store.initialize_library('PERFORMANCE_METRICS', lib_type=VERSION_STORE)
            self.logger.info("Biblioteca Arctic 'PERFORMANCE_METRICS' inicializada")
            
        # Biblioteca para indicadores técnicos
        if not self._arctic_store.library_exists('TECHNICAL_INDICATORS'):
            self._arctic_store.initialize_library('TECHNICAL_INDICATORS', lib_type=CHUNK_STORE)
            self.logger.info("Biblioteca Arctic 'TECHNICAL_INDICATORS' inicializada")
    
    async def get_market_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém dados de mercado para um símbolo.
        
        Args:
            symbol: Símbolo (par de trading)
            start_date: Data de início
            end_date: Data de fim
            interval: Intervalo de tempo (1m, 5m, 15m, 1h, 4h, 1d)
            
        Returns:
            Dicionário com dados de mercado ou None se não encontrado
        """
        self.logger.info(f"Obtendo dados de mercado para {symbol} no intervalo {interval}")
        
        # Na implementação real, os dados seriam obtidos do Arctic ou TimescaleDB
        # Esta é uma implementação simulada que gera dados aleatórios
        
        # Determinar número de pontos com base no intervalo
        if interval == "1m":
            delta = timedelta(minutes=1)
        elif interval == "5m":
            delta = timedelta(minutes=5)
        elif interval == "15m":
            delta = timedelta(minutes=15)
        elif interval == "1h":
            delta = timedelta(hours=1)
        elif interval == "4h":
            delta = timedelta(hours=4)
        else:  # 1d
            delta = timedelta(days=1)
        
        # Gerar série temporal
        current_date = start_date
        timestamps = []
        prices = []
        returns = []
        volumes = []
        
        # Preço inicial aleatório entre 1000 e 50000 para criptos
        price = np.random.uniform(1000, 50000)
        
        while current_date <= end_date:
            # Apenas adicionar pontos para dias úteis se for intervalo diário
            if interval == "1d" and current_date.weekday() >= 5:
                current_date += delta
                continue
                
            timestamps.append(current_date)
            prices.append(price)
            
            # Gerar retorno diário com distribuição normal (média 0, desvio 0.02)
            daily_return = np.random.normal(0, 0.02)
            returns.append(daily_return)
            
            # Atualizar preço
            price *= (1 + daily_return)
            
            # Gerar volume aleatório
            volume = np.random.uniform(100, 1000) * price
            volumes.append(volume)
            
            current_date += delta
        
        # Gerar dados OHLC
        opens = prices.copy()
        highs = [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices]
        closes = prices.copy()
        
        return {
            "symbol": symbol,
            "interval": interval,
            "timestamps": timestamps,
            "prices": prices,
            "returns": returns,
            "volumes": volumes,
            "opens": opens,
            "highs": highs,
            "lows": lows,
            "closes": closes
        }
    
    async def get_strategy_returns(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém retornos de uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            start_date: Data de início opcional
            end_date: Data de fim opcional
            
        Returns:
            Dicionário com dados de retornos ou None se não encontrado
        """
        self.logger.info(f"Obtendo retornos para estratégia {strategy_id}")
        
        # Na implementação real, os dados seriam obtidos do Arctic ou banco de dados
        # Esta é uma implementação simulada que gera dados aleatórios
        
        # Usar datas padrão se não especificadas
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=365)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Gerar série temporal diária
        current_date = start_date
        timestamps = []
        returns = []
        
        # Parâmetros para simulação mais realista
        # Média de retorno diário de 0.001 (cerca de 30% ao ano)
        # Desvio padrão de 0.015 (volatilidade anualizada de cerca de 24%)
        mean_return = 0.001
        std_return = 0.015
        
        while current_date <= end_date:
            # Apenas adicionar pontos para dias úteis
            if current_date.weekday() < 5:  # Segunda a Sexta
                timestamps.append(current_date)
                
                # Gerar retorno diário com distribuição normal
                daily_return = np.random.normal(mean_return, std_return)
                returns.append(daily_return)
            
            current_date += timedelta(days=1)
        
        # Calcular preços a partir dos retornos
        initial_price = 100.0  # Valor inicial arbitrário
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remover o valor inicial
        
        # Simular alguns trades
        num_trades = len(returns) // 10  # Aproximadamente um trade a cada 10 dias
        trades = []
        
        for i in range(num_trades):
            # Índice aleatório para entrada
            entry_idx = np.random.randint(0, len(timestamps) - 2)
            # Duração aleatória do trade (1 a 5 dias)
            duration = np.random.randint(1, 6)
            exit_idx = min(entry_idx + duration, len(timestamps) - 1)
            
            # Calcular P&L
            entry_price = prices[entry_idx]
            exit_price = prices[exit_idx]
            
            # Tamanho aleatório da posição
            position_size = np.random.uniform(0.1, 1.0)
            
            # Direção aleatória (long ou short)
            direction = np.random.choice(["long", "short"])
            
            # Calcular P&L
            if direction == "long":
                profit_loss = position_size * (exit_price - entry_price)
            else:
                profit_loss = position_size * (entry_price - exit_price)
            
            trades.append({
                "id": f"trade-{uuid.uuid4()}",
                "strategy_id": strategy_id,
                "symbol": "BTC/USD",  # Símbolo padrão para exemplo
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "position_size": position_size,
                "profit_loss": profit_loss,
                "entry_time": timestamps[entry_idx],
                "exit_time": timestamps[exit_idx],
                "duration_hours": duration * 24,  # Duração em horas
                "status": "closed"
            })
        
        return {
            "strategy_id": strategy_id,
            "strategy_name": f"Strategy {strategy_id}",
            "timestamps": timestamps,
            "returns": returns,
            "prices": prices,
            "trades": trades,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": 10000.0,  # Valor arbitrário para exemplo
            "final_capital": 10000.0 * np.prod(np.array(returns) + 1)  # Capital final
        }
    
    async def get_sentiment_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém dados de sentimento para um símbolo.
        
        Args:
            symbol: Símbolo (par de trading)
            start_date: Data de início
            end_date: Data de fim
            
        Returns:
            Dicionário com dados de sentimento ou None se não encontrado
        """
        self.logger.info(f"Obtendo dados de sentimento para {symbol}")
        
        # Na implementação real, os dados seriam obtidos de um serviço de sentimento
        # Esta é uma implementação simulada que gera dados aleatórios
        
        # Gerar série temporal diária
        current_date = start_date
        timestamps = []
        sentiment_scores = []
        
        while current_date <= end_date:
            # Apenas adicionar pontos para dias úteis
            if current_date.weekday() < 5:  # Segunda a Sexta
                timestamps.append(current_date)
                
                # Gerar score de sentimento entre -1 e 1
                sentiment = np.random.uniform(-1, 1)
                sentiment_scores.append(sentiment)
            
            current_date += timedelta(days=1)
        
        # Categorizar sentimento
        sentiment_categories = []
        for score in sentiment_scores:
            if score < -0.3:
                category = "bearish"
            elif score > 0.3:
                category = "bullish"
            else:
                category = "neutral"
            sentiment_categories.append(category)
        
        # Simular fontes de sentimento
        sources = ["twitter", "news", "reddit", "forums"]
        source_weights = {
            "twitter": np.random.uniform(0.2, 0.4),
            "news": np.random.uniform(0.3, 0.5),
            "reddit": np.random.uniform(0.1, 0.3),
            "forums": np.random.uniform(0.1, 0.2)
        }
        
        # Normalizar pesos
        total_weight = sum(source_weights.values())
        for source in source_weights:
            source_weights[source] /= total_weight
        
        return {
            "symbol": symbol,
            "timestamps": timestamps,
            "sentiment_scores": sentiment_scores,
            "sentiment_categories": sentiment_categories,
            "source_weights": source_weights
        }
    
    async def get_trades(
        self,
        strategy_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        status: Optional[str] = "closed"
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém dados de trades para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            start_date: Data de início opcional
            end_date: Data de fim opcional
            status: Status dos trades (open, closed, all)
            
        Returns:
            Dicionário com dados de trades ou None se não encontrado
        """
        self.logger.info(f"Obtendo trades para estratégia {strategy_id}")
        
        # Reutilizar a função get_strategy_returns que já inclui trades simulados
        strategy_data = await self.get_strategy_returns(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not strategy_data:
            return None
        
        # Filtrar trades por status se necessário
        trades = strategy_data["trades"]
        if status != "all":
            trades = [t for t in trades if t["status"] == status]
        
        # Filtrar por data se especificado
        if start_date:
            trades = [t for t in trades if t["exit_time"] >= start_date]
        if end_date:
            trades = [t for t in trades if t["exit_time"] <= end_date]
        
        return {
            "strategy_id": strategy_id,
            "strategy_name": strategy_data["strategy_name"],
            "trades": trades,
            "count": len(trades)
        }
    
    async def get_active_strategies(
        self,
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Obtém estratégias ativas para um usuário.
        
        Args:
            user_id: ID do usuário
            
        Returns:
            Lista de estratégias ativas
        """
        self.logger.info(f"Obtendo estratégias ativas para usuário {user_id}")
        
        # Na implementação real, os dados seriam obtidos do banco de dados
        # Esta é uma implementação simulada que gera dados aleatórios
        
        # Simular algumas estratégias
        num_strategies = np.random.randint(2, 6)  # 2 a 5 estratégias ativas
        strategies = []
        
        for i in range(num_strategies):
            strategy_id = f"strategy-{uuid.uuid4()}"
            
            # Gerar algumas métricas básicas
            total_return = np.random.uniform(-0.1, 0.5)  # -10% a +50%
            sharpe_ratio = np.random.uniform(0.5, 3.0)
            max_drawdown = np.random.uniform(0.05, 0.3)  # 5% a 30%
            win_rate = np.random.uniform(0.4, 0.7)  # 40% a 70%
            
            # Gerar alocação aleatória
            allocation = np.random.uniform(1000, 10000)
            
            # Gerar alguns detalhes adicionais
            details = {
                "type": np.random.choice(["trend_following", "mean_reversion", "momentum", "ml_based"]),
                "assets": np.random.choice(["BTC/USD", "ETH/USD", "SOL/USD", "multi_asset"], p=[0.3, 0.2, 0.1, 0.4]),
                "timeframe": np.random.choice(["1h", "4h", "1d"], p=[0.2, 0.3, 0.5])
            }
            
            strategies.append({
                "id": strategy_id,
                "name": f"Strategy {i+1}",
                "user_id": user_id,
                "status": "active",
                "allocation": allocation,
                "metrics": {
                    "total_return": total_return,
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown": max_drawdown,
                    "win_rate": win_rate
                },
                "details": details,
                "created_at": datetime.utcnow() - timedelta(days=np.random.randint(30, 365)),
                "last_updated": datetime.utcnow() - timedelta(minutes=np.random.randint(10, 1440))
            })
        
        return strategies
    
    async def save_performance_metrics(
        self,
        metrics: Union[PerformanceMetric, PerformanceMetricCollection, PerformanceAnalysis]
    ) -> str:
        """
        Salva métricas de performance no data warehouse.
        
        Args:
            metrics: Métricas a serem salvas
            
        Returns:
            ID das métricas salvas
        """
        metric_id = str(uuid.uuid4())
        
        if isinstance(metrics, PerformanceMetric):
            # Salvar métrica individual
            metric_data = metrics.to_dict()
            metric_data["id"] = metric_id
            
            if self._async_mongo_client:
                try:
                    db = self._async_mongo_client.crypto_trading
                    collection = db.performance_metrics
                    
                    result = await collection.insert_one(metric_data)
                    self.logger.info(f"Métrica de performance salva com ID {result.inserted_id}")
                    
                    return str(result.inserted_id)
                except Exception as e:
                    self.logger.error(f"Erro ao salvar métrica no MongoDB: {e}")
            
        elif isinstance(metrics, PerformanceMetricCollection):
            # Salvar coleção de métricas
            collection_data = metrics.to_dict()
            collection_data["id"] = metric_id
            
            if self._async_mongo_client:
                try:
                    db = self._async_mongo_client.crypto_trading
                    collection = db.performance_metric_collections
                    
                    result = await collection.insert_one(collection_data)
                    self.logger.info(f"Coleção de métricas salva com ID {result.inserted_id}")
                    
                    return str(result.inserted_id)
                except Exception as e:
                    self.logger.error(f"Erro ao salvar coleção de métricas no MongoDB: {e}")
            
        elif isinstance(metrics, PerformanceAnalysis):
            # Salvar análise completa
            analysis_data = metrics.to_dict()
            analysis_data["id"] = metric_id
            
            if self._async_mongo_client:
                try:
                    db = self._async_mongo_client.crypto_trading
                    collection = db.performance_analyses
                    
                    result = await collection.insert_one(analysis_data)
                    self.logger.info(f"Análise de performance salva com ID {result.inserted_id}")
                    
                    return str(result.inserted_id)
                except Exception as e:
                    self.logger.error(f"Erro ao salvar análise no MongoDB: {e}")
        
        # Fallback para retornar o ID gerado mesmo se não salvou
        return metric_id
    
    async def get_performance_analysis(
        self,
        strategy_id: str,
        time_range: str = "all"
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém uma análise de performance para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            time_range: Intervalo de tempo (day, week, month, year, all)
            
        Returns:
            Análise de performance ou None se não encontrado
        """
        self.logger.info(f"Obtendo análise de performance para estratégia {strategy_id}")
        
        if self._async_mongo_client:
            try:
                db = self._async_mongo_client.crypto_trading
                collection = db.performance_analyses
                
                # Buscar análise mais recente para o intervalo especificado
                query = {
                    "strategy_id": strategy_id,
                    "time_range": time_range
                }
                
                # Ordenar por timestamp decrescente para obter a mais recente
                cursor = collection.find(query).sort("timestamp", -1).limit(1)
                
                async for document in cursor:
                    # Remover _id do MongoDB
                    if "_id" in document:
                        del document["_id"]
                    
                    return document
                
                # Se não encontrou, tentar gerar uma análise sob demanda
                # Na implementação real, isso poderia chamar um serviço para calcular
                return await self._generate_performance_analysis(strategy_id, time_range)
                
            except Exception as e:
                self.logger.error(f"Erro ao buscar análise de performance no MongoDB: {e}")
                # Em caso de erro, gerar análise sob demanda
                return await self._generate_performance_analysis(strategy_id, time_range)
        
        # Se não tem MongoDB, gerar análise sob demanda
        return await self._generate_performance_analysis(strategy_id, time_range)
    
    async def _generate_performance_analysis(
        self,
        strategy_id: str,
        time_range: str = "all"
    ) -> Dict[str, Any]:
        """
        Gera uma análise de performance sob demanda para uma estratégia.
        
        Args:
            strategy_id: ID da estratégia
            time_range: Intervalo de tempo
            
        Returns:
            Análise de performance gerada
        """
        # Determinar datas com base no intervalo
        end_date = datetime.utcnow()
        
        if time_range == "day":
            start_date = end_date - timedelta(days=1)
        elif time_range == "week":
            start_date = end_date - timedelta(days=7)
        elif time_range == "month":
            start_date = end_date - timedelta(days=30)
        elif time_range == "year":
            start_date = end_date - timedelta(days=365)
        else:  # all
            start_date = end_date - timedelta(days=365 * 2)  # 2 anos por padrão
        
        # Obter dados da estratégia
        strategy_data = await self.get_strategy_returns(
            strategy_id=strategy_id,
            start_date=start_date,
            end_date=end_date
        )
        
        if not strategy_data:
            # Retornar análise vazia se não há dados
            return {
                "strategy_id": strategy_id,
                "strategy_name": f"Strategy {strategy_id}",
                "time_range": time_range,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "timestamp": datetime.utcnow().isoformat(),
                "initial_capital": 10000.0,
                "final_capital": 10000.0,
                "metrics": {},
                "trades_count": 0,
                "winning_trades_count": 0,
                "losing_trades_count": 0,
                "win_rate": 0
            }
        
        # Calcular algumas métricas básicas
        returns = np.array(strategy_data["returns"])
        timestamps = strategy_data["timestamps"]
        trades = strategy_data.get("trades", [])
        
        # Retorno total
        total_return = np.prod(returns + 1) - 1
        
        # Retorno anualizado
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatilidade
        volatility = np.std(returns) * np.sqrt(252)  # anualizada assumindo 252 dias de trading
        
        # Máximo drawdown
        cumulative = np.cumprod(returns + 1) - 1
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / (1 + running_max)
        max_drawdown = abs(np.min(drawdown))
        
        # Sharpe ratio (assumindo taxa livre de risco zero para simplificar)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (volatilidade apenas de retornos negativos)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Estatísticas de trades
        trades_count = len(trades)
        winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
        winning_trades_count = len(winning_trades)
        losing_trades_count = trades_count - winning_trades_count
        win_rate = winning_trades_count / trades_count if trades_count > 0 else 0
        
        # Montar resultado
        return {
            "strategy_id": strategy_id,
            "strategy_name": strategy_data.get("strategy_name", f"Strategy {strategy_id}"),
            "time_range": time_range,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
            "initial_capital": float(strategy_data.get("initial_capital", 10000.0)),
            "final_capital": float(strategy_data.get("final_capital", 10000.0)),
            "metrics": {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "max_drawdown": float(max_drawdown),
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio)
            },
            "trades_count": trades_count,
            "winning_trades_count": winning_trades_count,
            "losing_trades_count": losing_trades_count,
            "win_rate": float(win_rate)
        }
    
    async def get_technical_indicators(
        self,
        symbol: str,
        indicators: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> Optional[Dict[str, Any]]:
        """
        Obtém indicadores técnicos para um símbolo.
        
        Args:
            symbol: Símbolo (par de trading)
            indicators: Lista de indicadores a obter
            start_date: Data de início
            end_date: Data de fim
            interval: Intervalo de tempo
            
        Returns:
            Dicionário com indicadores técnicos ou None se não encontrado
        """
        self.logger.info(f"Obtendo indicadores técnicos para {symbol}")
        
        # Obter dados de mercado primeiro
        market_data = await self.get_market_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if not market_data:
            return None
        
        # Na implementação real, os indicadores seriam calculados ou obtidos de um serviço
        # Esta é uma implementação simulada que gera dados aleatórios
        
        result = {
            "symbol": symbol,
            "interval": interval,
            "timestamps": market_data["timestamps"],
            "indicators": {}
        }
        
        prices = np.array(market_data["prices"])
        
        for indicator in indicators:
            if indicator.lower() == "sma":
                # Simple Moving Average (20, 50, 200)
                for period in [20, 50, 200]:
                    if len(prices) >= period:
                        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
                        # Preencher valores iniciais com NaN
                        sma_full = np.full_like(prices, np.nan)
                        sma_full[period-1:] = sma
                        
                        result["indicators"][f"sma_{period}"] = sma_full.tolist()
            
            elif indicator.lower() == "ema":
                # Exponential Moving Average (20, 50, 200)
                for period in [20, 50, 200]:
                    if len(prices) >= period:
                        # Fórmula simplificada de EMA
                        alpha = 2 / (period + 1)
                        ema = np.zeros_like(prices)
                        ema[0] = prices[0]
                        
                        for i in range(1, len(prices)):
                            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
                        
                        result["indicators"][f"ema_{period}"] = ema.tolist()
            
            elif indicator.lower() == "rsi":
                # RSI (14)
                period = 14
                if len(prices) > period:
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
                    avg_gain[period] = np.mean(gain[1:period+1])
                    avg_loss[period] = np.mean(loss[1:period+1])
                    
                    # Calcular médias móveis subsequentes
                    for i in range(period+1, len(prices)):
                        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
                    
                    # Calcular RS e RSI
                    rs = np.zeros_like(prices)
                    rsi = np.zeros_like(prices)
                    
                    for i in range(period, len(prices)):
                        if avg_loss[i] == 0:
                            rsi[i] = 100
                        else:
                            rs[i] = avg_gain[i] / avg_loss[i]
                            rsi[i] = 100 - (100 / (1 + rs[i]))
                    
                    result["indicators"]["rsi_14"] = rsi.tolist()
            
            elif indicator.lower() == "bollinger":
                # Bollinger Bands (20, 2)
                period = 20
                num_std = 2
                
                if len(prices) >= period:
                    # Calcular SMA
                    sma = np.zeros_like(prices)
                    
                    for i in range(period-1, len(prices)):
                        sma[i] = np.mean(prices[i-(period-1):i+1])
                    
                    # Calcular desvio padrão móvel
                    mstd = np.zeros_like(prices)
                    
                    for i in range(period-1, len(prices)):
                        mstd[i] = np.std(prices[i-(period-1):i+1], ddof=1)
                    
                    # Calcular bandas
                    upper_band = sma + num_std * mstd
                    lower_band = sma - num_std * mstd
                    
                    result["indicators"]["bollinger_middle"] = sma.tolist()
                    result["indicators"]["bollinger_upper"] = upper_band.tolist()
                    result["indicators"]["bollinger_lower"] = lower_band.tolist()
            
            elif indicator.lower() == "macd":
                # MACD (12, 26, 9)
                fast_period = 12
                slow_period = 26
                signal_period = 9
                
                if len(prices) >= slow_period + signal_period:
                    # Calcular EMA rápido
                    alpha_fast = 2 / (fast_period + 1)
                    ema_fast = np.zeros_like(prices)
                    ema_fast[0] = prices[0]
                    
                    for i in range(1, len(prices)):
                        ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
                    
                    # Calcular EMA lento
                    alpha_slow = 2 / (slow_period + 1)
                    ema_slow = np.zeros_like(prices)
                    ema_slow[0] = prices[0]
                    
                    for i in range(1, len(prices)):
                        ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
                    
                    # Calcular MACD
                    macd = ema_fast - ema_slow
                    
                    # Calcular linha de sinal
                    alpha_signal = 2 / (signal_period + 1)
                    signal_line = np.zeros_like(prices)
                    signal_line[slow_period-1] = macd[slow_period-1]
                    
                    for i in range(slow_period, len(prices)):
                        signal_line[i] = alpha_signal * macd[i] + (1 - alpha_signal) * signal_line[i-1]
                    
                    # Calcular histograma
                    histogram = macd - signal_line
                    
                    result["indicators"]["macd"] = macd.tolist()
                    result["indicators"]["macd_signal"] = signal_line.tolist()
                    result["indicators"]["macd_histogram"] = histogram.tolist()
        
        return result
    
    async def save_report(
        self,
        user_id: str,
        report_name: str,
        report_type: str,
        url: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Salva um relatório.
        
        Args:
            user_id: ID do usuário
            report_name: Nome do relatório
            report_type: Tipo do relatório
            url: URL para acessar o relatório
            parameters: Parâmetros usados para gerar o relatório
            
        Returns:
            ID do relatório salvo
        """
        report_id = str(uuid.uuid4())
        
        if self._async_mongo_client:
            try:
                db = self._async_mongo_client.crypto_trading
                collection = db.reports
                
                # Preparar documento
                report_data = {
                    "id": report_id,
                    "user_id": user_id,
                    "name": report_name,
                    "type": report_type,
                    "url": url,
                    "parameters": parameters,
                    "created_at": datetime.utcnow()
                }
                
                result = await collection.insert_one(report_data)
                self.logger.info(f"Relatório salvo com ID {result.inserted_id}")
                
                return str(result.inserted_id)
            except Exception as e:
                self.logger.error(f"Erro ao salvar relatório no MongoDB: {e}")
        
        # Fallback para retornar o ID gerado mesmo se não salvou
        return report_id
    
    async def get_reports(
        self,
        user_id: str,
        page: int = 1,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Obtém relatórios para um usuário com paginação.
        
        Args:
            user_id: ID do usuário
            page: Número da página
            size: Itens por página
            
        Returns:
            Dicionário com os relatórios e informações de paginação
        """
        self.logger.info(f"Obtendo relatórios para usuário {user_id}")
        
        if self._async_mongo_client:
            try:
                db = self._async_mongo_client.crypto_trading
                collection = db.reports
                
                # Calcular skip para paginação
                skip = (page - 1) * size
                
                # Consulta
                query = {"user_id": user_id}
                
                # Contar total
                total = await collection.count_documents(query)
                
                # Buscar resultados paginados
                cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(size)
                
                reports = []
                async for document in cursor:
                    # Remover _id do MongoDB
                    if "_id" in document:
                        del document["_id"]
                    
                    reports.append(document)
                
                return {
                    "reports": reports,
                    "total": total,
                    "page": page,
                    "size": size,
                    "pages": (total + size - 1) // size
                }
            except Exception as e:
                self.logger.error(f"Erro ao buscar relatórios no MongoDB: {e}")
        
        # Caso não tenha MongoDB ou ocorra erro, retornar lista vazia
        return {
            "reports": [],
            "total": 0,
            "page": page,
            "size": size,
            "pages": 0
        }
    
    async def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtém um relatório pelo ID.
        
        Args:
            report_id: ID do relatório
            
        Returns:
            Dicionário com os dados do relatório ou None se não existir
        """
        self.logger.info(f"Obtendo relatório com ID {report_id}")
        
        if self._async_mongo_client:
            try:
                db = self._async_mongo_client.crypto_trading
                collection = db.reports
                
                # Buscar relatório
                document = await collection.find_one({"id": report_id})
                
                if document:
                    # Remover _id do MongoDB
                    if "_id" in document:
                        del document["_id"]
                    
                    return document
            except Exception as e:
                self.logger.error(f"Erro ao buscar relatório no MongoDB: {e}")
        
        return None