"""
Sistema de configuração centralizada para o bot de trading.

Este módulo implementa um sistema robusto de configuração que suporta
múltiplos ambientes, validação de esquemas, criptografia de dados sensíveis
e hot-reload de configurações.
"""
import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from datetime import datetime
import hashlib
import copy

from pydantic import BaseModel, ValidationError, Field, validator
from pydantic.env_settings import BaseSettings

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Ambientes suportados."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Formatos de configuração suportados."""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


@dataclass
class ConfigSource:
    """Representa uma fonte de configuração."""
    name: str
    format: ConfigFormat
    path: str
    priority: int = 0
    required: bool = True
    environment_specific: bool = False


class ExchangeConfig(BaseModel):
    """Configuração de uma exchange."""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    testnet: bool = False
    sandbox: bool = False
    rate_limit: int = 1200  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3
    
    class Config:
        extra = "allow"  # Permite campos adicionais específicos da exchange


class DatabaseConfig(BaseModel):
    """Configuração do banco de dados."""
    host: str = "localhost"
    port: int = 5432
    database: str = "neural_crypto_bot"
    user: str = "postgres"
    password: str = "postgres"
    schema: str = "public"
    min_connections: int = 5
    max_connections: int = 20
    enable_ssl: bool = False
    pool_timeout: int = 30
    command_timeout: int = 60
    
    @validator('port')
    def validate_port(cls, v):
        if not (1 <= v <= 65535):
            raise ValueError('Port must be between 1 and 65535')
        return v


class KafkaConfig(BaseModel):
    """Configuração do Kafka."""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "neural-crypto-bot"
    topic_prefix: str = "market-data"
    acks: str = "all"
    retries: int = 3
    batch_size: int = 16384
    linger_ms: int = 5
    compression_type: str = "snappy"
    enable_idempotence: bool = True


class RedisConfig(BaseModel):
    """Configuração do Redis."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    max_connections: int = 20
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    health_check_interval: int = 30


class LoggingConfig(BaseModel):
    """Configuração de logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    enable_json: bool = False
    enable_structured: bool = True
    log_file: Optional[str] = None
    max_file_size: str = "100MB"
    backup_count: int = 5
    enable_console: bool = True


class DataCollectionConfig(BaseModel):
    """Configuração específica para coleta de dados."""
    enabled_exchanges: List[str] = Field(default_factory=lambda: ["binance", "coinbase"])
    trading_pairs: List[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframes: List[str] = Field(default_factory=lambda: ["1m", "5m", "1h", "1d"])
    enable_orderbook: bool = True
    enable_trades: bool = True
    enable_funding_rates: bool = True
    enable_liquidations: bool = True
    orderbook_depth: int = 20
    data_retention_days: int = 30
    enable_compression: bool = True
    compression_algorithm: str = "zstd"


class RiskManagementConfig(BaseModel):
    """Configuração de gestão de risco."""
    max_position_size: float = 10000.0  # USD
    max_daily_loss: float = 1000.0  # USD
    max_drawdown_percent: float = 10.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: float = 5.0  # %
    position_size_percent: float = 2.0  # % of portfolio
    stop_loss_percent: float = 2.0
    take_profit_percent: float = 6.0


class SecurityConfig(BaseModel):
    """Configuração de segurança."""
    enable_encryption: bool = True
    encryption_algorithm: str = "fernet"
    key_rotation_days: int = 30
    max_key_age_days: int = 90
    enable_api_authentication: bool = True
    api_key_header: str = "X-API-Key"
    jwt_secret_key: Optional[str] = None
    jwt_expiration_hours: int = 24
    enable_rate_limiting: bool = True
    rate_limit_requests_per_minute: int = 60


class MonitoringConfig(BaseModel):
    """Configuração de monitoramento."""
    enable_metrics: bool = True
    metrics_port: int = 8000
    enable_health_checks: bool = True
    health_check_interval: int = 30
    enable_alerting: bool = True
    alert_webhook_url: Optional[str] = None
    enable_prometheus: bool = True
    enable_grafana: bool = False


class TradingBotConfig(BaseSettings):
    """
    Configuração principal do bot de trading.
    
    Utiliza Pydantic BaseSettings para integração com variáveis de ambiente.
    """
    
    # Configurações gerais
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    version: str = "1.0.0"
    timezone: str = "UTC"
    
    # Configurações de componentes
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    class Config:
        env_file = ".env"
        env_prefix = "BOT_"
        env_nested_delimiter = "__"
        case_sensitive = False
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            """Parse personalizado para variáveis de ambiente."""
            if field_name == 'exchanges':
                try:
                    return json.loads(raw_val)
                except json.JSONDecodeError:
                    return {}
            return cls.json_loads(raw_val)


class ConfigManager:
    """
    Gerenciador principal de configurações.
    
    Responsável por carregar, validar, mesclar e distribuir configurações
    para todos os componentes do sistema.
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        environment: Optional[Environment] = None,
        enable_hot_reload: bool = False
    ):
        """
        Inicializa o gerenciador de configurações.
        
        Args:
            config_dir: Diretório de configurações
            environment: Ambiente específico
            enable_hot_reload: Habilita hot-reload de configurações
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or self._detect_environment()
        self.enable_hot_reload = enable_hot_reload
        
        # Estado interno
        self._config: Optional[TradingBotConfig] = None
        self._config_sources: List[ConfigSource] = []
        self._file_checksums: Dict[str, str] = {}
        self._change_callbacks: List[Callable[[TradingBotConfig], None]] = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"ConfigManager inicializado - Ambiente: {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Detecta o ambiente atual."""
        env_var = os.getenv("ENVIRONMENT", os.getenv("ENV", "development"))
        
        try:
            return Environment(env_var.lower())
        except ValueError:
            logger.warning(f"Ambiente desconhecido: {env_var}, usando development")
            return Environment.DEVELOPMENT
    
    def _setup_logging(self) -> None:
        """Configura logging básico."""
        level = logging.DEBUG if self.environment == Environment.DEVELOPMENT else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def add_config_source(
        self,
        name: str,
        path: str,
        format: ConfigFormat,
        priority: int = 0,
        required: bool = True,
        environment_specific: bool = False
    ) -> None:
        """
        Adiciona uma fonte de configuração.
        
        Args:
            name: Nome da fonte
            path: Caminho do arquivo
            format: Formato do arquivo
            priority: Prioridade (maior = mais importante)
            required: Se o arquivo é obrigatório
            environment_specific: Se é específico do ambiente
        """
        source = ConfigSource(
            name=name,
            format=format,
            path=path,
            priority=priority,
            required=required,
            environment_specific=environment_specific
        )
        
        self._config_sources.append(source)
        
        # Ordena por prioridade (maior primeiro)
        self._config_sources.sort(key=lambda x: x.priority, reverse=True)
        
        logger.debug(f"Fonte de configuração adicionada: {name} ({path})")
    
    def load_config(self) -> TradingBotConfig:
        """
        Carrega e valida todas as configurações.
        
        Returns:
            Configuração validada e mesclada
        """
        logger.info("Carregando configurações...")
        
        # Se não há fontes configuradas, usa padrões
        if not self._config_sources:
            self._setup_default_sources()
        
        # Carrega dados de todas as fontes
        merged_data = self._load_and_merge_sources()
        
        # Processa variáveis de ambiente
        env_data = self._load_environment_variables()
        merged_data.update(env_data)
        
        # Valida e cria configuração
        try:
            self._config = TradingBotConfig(**merged_data)
            logger.info("Configurações carregadas e validadas com sucesso")
            
            # Inicia hot-reload se habilitado
            if self.enable_hot_reload:
                asyncio.create_task(self._start_hot_reload())
            
            return self._config
            
        except ValidationError as e:
            logger.error(f"Erro de validação na configuração: {e}")
            raise ConfigurationError(f"Configuração inválida: {e}")
    
    def get_config(self) -> TradingBotConfig:
        """
        Obtém a configuração atual.
        
        Returns:
            Configuração atual
        """
        if self._config is None:
            self._config = self.load_config()
        
        return self._config
    
    def reload_config(self) -> TradingBotConfig:
        """
        Recarrega as configurações.
        
        Returns:
            Nova configuração carregada
        """
        logger.info("Recarregando configurações...")
        
        old_config = copy.deepcopy(self._config)
        self._config = self.load_config()
        
        # Notifica callbacks de mudança
        if old_config and self._config != old_config:
            self._notify_config_change(self._config)
        
        return self._config
    
    def add_change_callback(self, callback: Callable[[TradingBotConfig], None]) -> None:
        """
        Adiciona callback para mudanças de configuração.
        
        Args:
            callback: Função a ser chamada quando configuração mudar
        """
        self._change_callbacks.append(callback)
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """
        Obtém configuração de uma exchange específica.
        
        Args:
            exchange_name: Nome da exchange
            
        Returns:
            Configuração da exchange ou None
        """
        config = self.get_config()
        return config.exchanges.get(exchange_name.lower())
    
    def is_exchange_enabled(self, exchange_name: str) -> bool:
        """
        Verifica se uma exchange está habilitada.
        
        Args:
            exchange_name: Nome da exchange
            
        Returns:
            True se habilitada
        """
        config = self.get_exchange_config(exchange_name)
        return config.enabled if config else False
    
    def get_trading_pairs(self) -> List[str]:
        """
        Obtém lista de pares de trading configurados.
        
        Returns:
            Lista de pares de trading
        """
        config = self.get_config()
        return config.data_collection.trading_pairs
    
    def get_enabled_exchanges(self) -> List[str]:
        """
        Obtém lista de exchanges habilitadas.
        
        Returns:
            Lista de exchanges habilitadas
        """
        config = self.get_config()
        enabled = []
        
        for name, exchange_config in config.exchanges.items():
            if exchange_config.enabled:
                enabled.append(name)
        
        return enabled
    
    def _setup_default_sources(self) -> None:
        """Configura fontes padrão de configuração."""
        base_dir = self.config_dir
        env_name = self.environment.value
        
        # Arquivo base
        self.add_config_source(
            name="base_config",
            path=str(base_dir / "config.yaml"),
            format=ConfigFormat.YAML,
            priority=1,
            required=True
        )
        
        # Arquivo específico do ambiente
        self.add_config_source(
            name=f"{env_name}_config",
            path=str(base_dir / f"config.{env_name}.yaml"),
            format=ConfigFormat.YAML,
            priority=2,
            required=False,
            environment_specific=True
        )
        
        # Arquivo local (para overrides)
        self.add_config_source(
            name="local_config",
            path=str(base_dir / "config.local.yaml"),
            format=ConfigFormat.YAML,
            priority=3,
            required=False
        )
        
        # Secrets (se existir)
        self.add_config_source(
            name="secrets",
            path=str(base_dir / "secrets.yaml"),
            format=ConfigFormat.YAML,
            priority=4,
            required=False
        )
    
    def _load_and_merge_sources(self) -> Dict[str, Any]:
        """Carrega e mescla dados de todas as fontes."""
        merged_data = {}
        
        for source in reversed(self._config_sources):  # Prioridade crescente
            if not self._should_load_source(source):
                continue
            
            try:
                data = self._load_source_data(source)
                if data:
                    merged_data = self._deep_merge(merged_data, data)
                    logger.debug(f"Dados carregados de: {source.name}")
                    
            except FileNotFoundError:
                if source.required:
                    raise ConfigurationError(f"Arquivo de configuração obrigatório não encontrado: {source.path}")
                else:
                    logger.debug(f"Arquivo opcional não encontrado: {source.path}")
            
            except Exception as e:
                logger.error(f"Erro ao carregar {source.name}: {e}")
                if source.required:
                    raise ConfigurationError(f"Erro ao carregar configuração obrigatória: {e}")
        
        return merged_data
    
    def _should_load_source(self, source: ConfigSource) -> bool:
        """Verifica se uma fonte deve ser carregada."""
        if source.environment_specific:
            # Carrega apenas se for do ambiente atual
            env_indicator = f".{self.environment.value}."
            return env_indicator in source.path
        
        return True
    
    def _load_source_data(self, source: ConfigSource) -> Dict[str, Any]:
        """Carrega dados de uma fonte específica."""
        path = Path(source.path)
        
        if not path.exists():
            if source.required:
                raise FileNotFoundError(f"Arquivo não encontrado: {source.path}")
            return {}
        
        # Calcula checksum para hot-reload
        if self.enable_hot_reload:
            self._file_checksums[source.path] = self._calculate_file_checksum(path)
        
        # Carrega dados baseado no formato
        with open(path, 'r', encoding='utf-8') as file:
            if source.format == ConfigFormat.YAML:
                return yaml.safe_load(file) or {}
            elif source.format == ConfigFormat.JSON:
                return json.load(file) or {}
            else:
                raise ConfigurationError(f"Formato não suportado: {source.format}")
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Carrega variáveis de ambiente relevantes."""
        env_data = {}
        
        # Prefixos para buscar
        prefixes = ["BOT_", "NEURAL_", "TRADING_"]
        
        for key, value in os.environ.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    # Converte nome da variável para estrutura aninhada
                    config_key = key[len(prefix):].lower()
                    
                    # Tenta converter valores
                    converted_value = self._convert_env_value(value)
                    env_data[config_key] = converted_value
                    break
        
        return env_data
    
    def _convert_env_value(self, value: str) -> Any:
        """Converte valor de variável de ambiente para tipo apropriado."""
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        if value.isdigit():
            return int(value)
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value
    
    def _deep_merge(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Mescla dicionários recursivamente."""
        result = base.copy()
        
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calcula checksum MD5 de um arquivo."""
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    async def _start_hot_reload(self) -> None:
        """Inicia monitoramento para hot-reload."""
        logger.info("Hot-reload habilitado")
        
        while True:
            try:
                await asyncio.sleep(5)  # Verifica a cada 5 segundos
                
                # Verifica se algum arquivo mudou
                changed = False
                for source in self._config_sources:
                    if not Path(source.path).exists():
                        continue
                    
                    old_checksum = self._file_checksums.get(source.path)
                    new_checksum = self._calculate_file_checksum(Path(source.path))
                    
                    if old_checksum and old_checksum != new_checksum:
                        logger.info(f"Mudança detectada em: {source.path}")
                        changed = True
                        break
                
                # Recarrega se houver mudanças
                if changed:
                    self.reload_config()
                    
            except Exception as e:
                logger.error(f"Erro no hot-reload: {e}")
                await asyncio.sleep(10)  # Espera mais em caso de erro
    
    def _notify_config_change(self, new_config: TradingBotConfig) -> None:
        """Notifica callbacks sobre mudança de configuração."""
        logger.info("Notificando mudanças de configuração")
        
        for callback in self._change_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                logger.error(f"Erro em callback de configuração: {e}")
    
    def export_config(self, format: ConfigFormat = ConfigFormat.YAML) -> str:
        """
        Exporta configuração atual.
        
        Args:
            format: Formato de exportação
            
        Returns:
            Configuração serializada
        """
        config = self.get_config()
        config_dict = config.dict()
        
        if format == ConfigFormat.YAML:
            return yaml.dump(config_dict, default_flow_style=False)
        elif format == ConfigFormat.JSON:
            return json.dumps(config_dict, indent=2)
        else:
            raise ConfigurationError(f"Formato de exportação não suportado: {format}")
    
    def validate_config(self, config_data: Dict[str, Any]) -> bool:
        """
        Valida dados de configuração.
        
        Args:
            config_data: Dados a serem validados
            
        Returns:
            True se válido
        """
        try:
            TradingBotConfig(**config_data)
            return True
        except ValidationError as e:
            logger.error(f"Configuração inválida: {e}")
            return False


class ConfigurationError(Exception):
    """Exceção para erros de configuração."""
    pass


# Instância global do gerenciador
_config_manager: Optional[ConfigManager] = None


def get_config_manager(
    config_dir: str = "config",
    environment: Optional[Environment] = None,
    enable_hot_reload: bool = False
) -> ConfigManager:
    """
    Obtém instância global do gerenciador de configurações.
    
    Args:
        config_dir: Diretório de configurações
        environment: Ambiente específico
        enable_hot_reload: Habilita hot-reload
        
    Returns:
        Instância do ConfigManager
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(
            config_dir=config_dir,
            environment=environment,
            enable_hot_reload=enable_hot_reload
        )
    
    return _config_manager


def get_config() -> TradingBotConfig:
    """
    Obtém configuração atual.
    
    Returns:
        Configuração atual
    """
    return get_config_manager().get_config()


def reload_config() -> TradingBotConfig:
    """
    Recarrega configurações.
    
    Returns:
        Nova configuração
    """
    return get_config_manager().reload_config()


# Utility functions para acesso rápido a configurações
def get_exchange_config(exchange_name: str) -> Optional[ExchangeConfig]:
    """Obtém configuração de exchange."""
    return get_config_manager().get_exchange_config(exchange_name)


def get_database_config() -> DatabaseConfig:
    """Obtém configuração do banco de dados."""
    return get_config().database


def get_kafka_config() -> KafkaConfig:
    """Obtém configuração do Kafka."""
    return get_config().kafka


def get_redis_config() -> RedisConfig:
    """Obtém configuração do Redis."""
    return get_config().redis


def is_development_environment() -> bool:
    """Verifica se está em ambiente de desenvolvimento."""
    return get_config().environment == Environment.DEVELOPMENT


def is_production_environment() -> bool:
    """Verifica se está em ambiente de produção."""
    return get_config().environment == Environment.PRODUCTION