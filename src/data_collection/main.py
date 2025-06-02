"""
Módulo principal para coleta de dados de mercado - Versão Simplificada.

Este módulo implementa uma versão mais simples do serviço de coleta de dados,
ideal para desenvolvimento, testes e ambientes com recursos limitados.
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from src.data_collection.adapters.binance_adapter import BinanceAdapter
from src.data_collection.adapters.coinbase_adapter import CoinbaseAdapter
from src.data_collection.adapters.kraken_adapter import KrakenAdapter
from src.data_collection.adapters.bybit_adapter import BybitAdapter

from src.data_collection.application.services.data_validation_service import DataValidationService
from src.data_collection.domain.entities.candle import Candle, TimeFrame
from src.data_collection.domain.entities.orderbook import OrderBook
from src.data_collection.domain.entities.trade import Trade

from src.data_collection.infrastructure.database import DatabaseManager

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_collection_simple.log')
    ]
)
logger = logging.getLogger(__name__)


class SimpleDataCollectionService:
    """
    Serviço simplificado para coleta de dados de mercado.
    
    Versão mais simples focada em funcionalidade básica sem
    complexidades adicionais como balanceamento de carga,
    múltiplas exchanges simultâneas ou otimizações avançadas.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o serviço simplificado.
        
        Args:
            config: Configurações básicas do serviço
        """
        self.config = config
        
        # Serviços básicos
        self.validation_service = DataValidationService()
        
        # Exchange adapter (apenas uma por vez nesta versão)
        self.exchange = None
        self.exchange_name = config.get('exchange', 'binance')
        
        # Banco de dados opcional
        self.db_manager = None
        self.enable_database = config.get('enable_database', False)
        
        # Configurações de coleta
        self.trading_pairs = config.get('trading_pairs', ['BTC/USDT', 'ETH/USDT'])
        self.timeframes = config.get('timeframes', ['1m', '5m', '1h'])
        self.collect_orderbook = config.get('collect_orderbook', True)
        self.collect_trades = config.get('collect_trades', True)
        self.collect_candles = config.get('collect_candles', True)
        
        # Controle de estado
        self._running = False
        self._tasks = []
        
        # Contadores simples
        self.stats = {
            'orderbooks_processed': 0,
            'trades_processed': 0,
            'candles_processed': 0,
            'errors': 0,
            'start_time': None
        }
    
    async def initialize(self) -> None:
        """Inicializa o serviço e suas dependências."""
        logger.info("Inicializando serviço simplificado de coleta de dados")
        
        # Inicializa a exchange
        await self._initialize_exchange()
        
        # Inicializa o banco de dados se habilitado
        if self.enable_database:
            await self._initialize_database()
        
        self.stats['start_time'] = datetime.utcnow()
        logger.info("Serviço inicializado com sucesso")
    
    async def _initialize_exchange(self) -> None:
        """Inicializa o adapter da exchange."""
        exchange_config = self.config.get('exchange_config', {})
        
        logger.info(f"Inicializando exchange: {self.exchange_name}")
        
        if self.exchange_name.lower() == 'binance':
            self.exchange = BinanceAdapter(
                api_key=exchange_config.get('api_key'),
                api_secret=exchange_config.get('api_secret'),
                testnet=exchange_config.get('testnet', False)
            )
        elif self.exchange_name.lower() in ['coinbase', 'coinbasepro']:
            self.exchange = CoinbaseAdapter(
                api_key=exchange_config.get('api_key'),
                api_secret=exchange_config.get('api_secret'),
                api_passphrase=exchange_config.get('api_passphrase'),
                sandbox=exchange_config.get('sandbox', False)
            )
        elif self.exchange_name.lower() == 'kraken':
            self.exchange = KrakenAdapter(
                api_key=exchange_config.get('api_key'),
                api_secret=exchange_config.get('api_secret'),
                testnet=exchange_config.get('testnet', False)
            )
        elif self.exchange_name.lower() == 'bybit':
            self.exchange = BybitAdapter(
                api_key=exchange_config.get('api_key'),
                api_secret=exchange_config.get('api_secret'),
                testnet=exchange_config.get('testnet', False)
            )
        else:
            raise ValueError(f"Exchange não suportada: {self.exchange_name}")
        
        await self.exchange.initialize()
        logger.info(f"Exchange {self.exchange_name} inicializada")
    
    async def _initialize_database(self) -> None:
        """Inicializa a conexão com o banco de dados."""
        db_config = self.config.get('database', {})
        
        self.db_manager = DatabaseManager(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'crypto_data'),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', 'postgres'),
            min_connections=db_config.get('min_connections', 2),
            max_connections=db_config.get('max_connections', 10)
        )
        
        await self.db_manager.initialize()
        logger.info("Banco de dados inicializado")
    
    async def start(self) -> None:
        """Inicia a coleta de dados."""
        if self._running:
            logger.warning("Serviço já está rodando")
            return
        
        self._running = True
        logger.info("Iniciando coleta de dados")
        
        # Valida pares de negociação
        await self._validate_trading_pairs()
        
        # Inicia coleta para cada par de negociação
        for trading_pair in self.trading_pairs:
            if self.collect_orderbook:
                task = asyncio.create_task(self._collect_orderbook(trading_pair))
                self._tasks.append(task)
            
            if self.collect_trades:
                task = asyncio.create_task(self._collect_trades(trading_pair))
                self._tasks.append(task)
            
            if self.collect_candles:
                for timeframe_str in self.timeframes:
                    try:
                        timeframe = TimeFrame(timeframe_str)
                        task = asyncio.create_task(self._collect_candles(trading_pair, timeframe))
                        self._tasks.append(task)
                    except ValueError:
                        logger.warning(f"Timeframe inválido: {timeframe_str}")
        
        # Task de monitoramento
        self._tasks.append(asyncio.create_task(self._monitor_stats()))
        
        logger.info(f"Coleta iniciada para {len(self.trading_pairs)} pares em {len(self._tasks)} tasks")
    
    async def stop(self) -> None:
        """Para a coleta de dados."""
        if not self._running:
            return
        
        logger.info("Parando coleta de dados")
        self._running = False
        
        # Cancela todas as tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Aguarda cancelamento
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks = []
        
        # Fecha conexões
        if self.exchange:
            await self.exchange.shutdown()
        
        if self.db_manager:
            await self.db_manager.close()
        
        logger.info("Coleta de dados parada")
    
    async def _validate_trading_pairs(self) -> None:
        """Valida se os pares de negociação são suportados."""
        logger.info("Validando pares de negociação")
        
        available_pairs = await self.exchange.fetch_trading_pairs()
        invalid_pairs = []
        
        for pair in self.trading_pairs:
            if not self.exchange.validate_trading_pair(pair):
                invalid_pairs.append(pair)
        
        if invalid_pairs:
            logger.warning(f"Pares inválidos removidos: {invalid_pairs}")
            self.trading_pairs = [p for p in self.trading_pairs if p not in invalid_pairs]
        
        logger.info(f"Pares válidos: {self.trading_pairs}")
    
    async def _collect_orderbook(self, trading_pair: str) -> None:
        """Coleta dados de orderbook para um par."""
        logger.info(f"Iniciando coleta de orderbook para {trading_pair}")
        
        try:
            async def on_orderbook(orderbook: OrderBook) -> None:
                try:
                    # Valida o orderbook
                    if self.validation_service.validate_orderbook(orderbook):
                        # Salva no banco se habilitado
                        if self.db_manager:
                            await self.db_manager.save_orderbook(orderbook)
                        
                        self.stats['orderbooks_processed'] += 1
                        
                        # Log ocasional
                        if self.stats['orderbooks_processed'] % 100 == 0:
                            logger.info(f"Orderbooks processados: {self.stats['orderbooks_processed']}")
                    
                except Exception as e:
                    logger.error(f"Erro ao processar orderbook: {e}")
                    self.stats['errors'] += 1
            
            # Subscreve para atualizações
            await self.exchange.subscribe_orderbook(trading_pair, on_orderbook)
            
            # Mantém a subscrição ativa
            while self._running:
                await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            logger.debug(f"Coleta de orderbook cancelada para {trading_pair}")
        except Exception as e:
            logger.error(f"Erro na coleta de orderbook para {trading_pair}: {e}")
            self.stats['errors'] += 1
        finally:
            try:
                await self.exchange.unsubscribe_orderbook(trading_pair)
            except Exception:
                pass
    
    async def _collect_trades(self, trading_pair: str) -> None:
        """Coleta dados de trades para um par."""
        logger.info(f"Iniciando coleta de trades para {trading_pair}")
        
        try:
            async def on_trade(trade: Trade) -> None:
                try:
                    # Valida o trade
                    if self.validation_service.validate_trade(trade):
                        # Salva no banco se habilitado
                        if self.db_manager:
                            await self.db_manager.save_trade(trade)
                        
                        self.stats['trades_processed'] += 1
                        
                        # Log ocasional
                        if self.stats['trades_processed'] % 1000 == 0:
                            logger.info(f"Trades processados: {self.stats['trades_processed']}")
                
                except Exception as e:
                    logger.error(f"Erro ao processar trade: {e}")
                    self.stats['errors'] += 1
            
            # Subscreve para atualizações
            await self.exchange.subscribe_trades(trading_pair, on_trade)
            
            # Mantém a subscrição ativa
            while self._running:
                await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            logger.debug(f"Coleta de trades cancelada para {trading_pair}")
        except Exception as e:
            logger.error(f"Erro na coleta de trades para {trading_pair}: {e}")
            self.stats['errors'] += 1
        finally:
            try:
                await self.exchange.unsubscribe_trades(trading_pair)
            except Exception:
                pass
    
    async def _collect_candles(self, trading_pair: str, timeframe: TimeFrame) -> None:
        """Coleta dados de candles para um par e timeframe."""
        logger.info(f"Iniciando coleta de candles para {trading_pair} - {timeframe.value}")
        
        try:
            async def on_candle(candle: Candle) -> None:
                try:
                    # Valida a candle
                    if self.validation_service.validate_candle(candle):
                        # Salva no banco se habilitado
                        if self.db_manager:
                            await self.db_manager.save_candle(candle)
                        
                        self.stats['candles_processed'] += 1
                        
                        # Log ocasional
                        if self.stats['candles_processed'] % 100 == 0:
                            logger.info(f"Candles processados: {self.stats['candles_processed']}")
                
                except Exception as e:
                    logger.error(f"Erro ao processar candle: {e}")
                    self.stats['errors'] += 1
            
            # Subscreve para atualizações
            await self.exchange.subscribe_candles(trading_pair, timeframe, on_candle)
            
            # Mantém a subscrição ativa
            while self._running:
                await asyncio.sleep(1)
            
        except asyncio.CancelledError:
            logger.debug(f"Coleta de candles cancelada para {trading_pair} - {timeframe.value}")
        except Exception as e:
            logger.error(f"Erro na coleta de candles para {trading_pair} - {timeframe.value}: {e}")
            self.stats['errors'] += 1
        finally:
            try:
                await self.exchange.unsubscribe_candles(trading_pair, timeframe)
            except Exception:
                pass
    
    async def _monitor_stats(self) -> None:
        """Monitora e reporta estatísticas do sistema."""
        while self._running:
            try:
                # Calcula uptime
                uptime = datetime.utcnow() - self.stats['start_time']
                
                # Calcula taxas
                total_processed = (self.stats['orderbooks_processed'] + 
                                 self.stats['trades_processed'] + 
                                 self.stats['candles_processed'])
                
                rate_per_second = total_processed / max(1, uptime.total_seconds())
                
                # Log das estatísticas
                logger.info(
                    f"Stats - Uptime: {uptime}, "
                    f"Orderbooks: {self.stats['orderbooks_processed']}, "
                    f"Trades: {self.stats['trades_processed']}, "
                    f"Candles: {self.stats['candles_processed']}, "
                    f"Errors: {self.stats['errors']}, "
                    f"Rate: {rate_per_second:.2f}/s"
                )
                
                await asyncio.sleep(60)  # Reporta a cada minuto
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                await asyncio.sleep(30)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais."""
        uptime = datetime.utcnow() - self.stats['start_time'] if self.stats['start_time'] else timedelta(0)
        total_processed = (self.stats['orderbooks_processed'] + 
                         self.stats['trades_processed'] + 
                         self.stats['candles_processed'])
        
        return {
            **self.stats,
            'uptime_seconds': uptime.total_seconds(),
            'total_processed': total_processed,
            'rate_per_second': total_processed / max(1, uptime.total_seconds())
        }


async def run_simple_collection():
    """Executa o serviço simplificado com configuração padrão."""
    
    # Configuração simples padrão
    config = {
        'exchange': 'binance',
        'exchange_config': {
            'testnet': True  # Usa testnet por padrão para segurança
        },
        'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['1m', '5m'],
        'collect_orderbook': True,
        'collect_trades': True,
        'collect_candles': True,
        'enable_database': False,  # Desabilitado por padrão na versão simples
    }
    
    # Cria e inicializa o serviço
    service = SimpleDataCollectionService(config)
    
    try:
        await service.initialize()
        await service.start()
        
        logger.info("Pressione Ctrl+C para parar o serviço")
        
        # Aguarda indefinidamente
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Parando serviço...")
    except Exception as e:
        logger.error(f"Erro no serviço: {e}")
    finally:
        await service.stop()


async def run_with_database():
    """Executa o serviço com banco de dados habilitado."""
    
    config = {
        'exchange': 'binance',
        'exchange_config': {
            'testnet': True
        },
        'trading_pairs': ['BTC/USDT', 'ETH/USDT'],
        'timeframes': ['1m', '5m', '1h'],
        'collect_orderbook': True,
        'collect_trades': True,
        'collect_candles': True,
        'enable_database': True,
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'crypto_data',
            'user': 'postgres',
            'password': 'postgres'
        }
    }
    
    service = SimpleDataCollectionService(config)
    
    try:
        await service.initialize()
        await service.start()
        
        logger.info("Pressione Ctrl+C para parar o serviço")
        
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Parando serviço...")
    except Exception as e:
        logger.error(f"Erro no serviço: {e}")
    finally:
        await service.stop()


if __name__ == "__main__":
    """
    Ponto de entrada principal.
    
    Exemplos de uso:
    
    1. Execução simples (sem banco de dados):
       python main.py
    
    2. Execução com banco de dados:
       python main.py --with-database
    
    3. Execução com exchange específica:
       python main.py --exchange kraken
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Coleta de Dados Simplificada')
    parser.add_argument('--with-database', action='store_true',
                       help='Habilita armazenamento em banco de dados')
    parser.add_argument('--exchange', default='binance',
                       choices=['binance', 'coinbase', 'kraken', 'bybit'],
                       help='Exchange a ser utilizada')
    parser.add_argument('--pairs', nargs='+', default=['BTC/USDT', 'ETH/USDT'],
                       help='Pares de negociação a coletar')
    parser.add_argument('--timeframes', nargs='+', default=['1m', '5m'],
                       help='Timeframes para candles')
    
    args = parser.parse_args()
    
    # Configura baseado nos argumentos
    config = {
        'exchange': args.exchange,
        'exchange_config': {
            'testnet': True  # Sempre usa testnet na versão simples
        },
        'trading_pairs': args.pairs,
        'timeframes': args.timeframes,
        'collect_orderbook': True,
        'collect_trades': True,
        'collect_candles': True,
        'enable_database': args.with_database,
    }
    
    if args.with_database:
        config['database'] = {
            'host': 'localhost',
            'port': 5432,
            'database': 'crypto_data',
            'user': 'postgres',
            'password': 'postgres'
        }
    
    async def main():
        service = SimpleDataCollectionService(config)
        
        try:
            await service.initialize()
            await service.start()
            
            logger.info("Serviço iniciado. Pressione Ctrl+C para parar")
            
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Parando serviço...")
        except Exception as e:
            logger.error(f"Erro no serviço: {e}")
        finally:
            await service.stop()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Programa interrompido")
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)