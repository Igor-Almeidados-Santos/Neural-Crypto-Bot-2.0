import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseSettings(BaseSettings):
    """Configurações do Banco de Dados"""
    DB_USER: str = "oasis"
    DB_PASSWORD: str = "yoursecurepassword"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "oasis_db"
    
    # Formata a URL de conexão
    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(env_file='.env', env_prefix='DB_', extra='ignore')

class RedisSettings(BaseSettings):
    """Configurações do Redis"""
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    model_config = SettingsConfigDict(env_file='.env', env_prefix='REDIS_', extra='ignore')

class ApiSettings(BaseSettings):
    """Configurações da API e do Bot"""
    API_KEY_BINANCE: str
    API_SECRET_BINANCE: str
    ENVIRONMENT: str = "development" # development ou production
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


class Settings(BaseSettings):
    """Agregador central de todas as configurações."""
    db: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    api: ApiSettings = ApiSettings()

# Instância global para ser importada por outras partes do sistema
settings = Settings()