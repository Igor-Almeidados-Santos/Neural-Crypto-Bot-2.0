"""
Neural Crypto Bot - API Main Module

Este é o ponto de entrada para a API do Neural Crypto Bot, um bot de trading de criptomoedas 
avançado utilizando arquitetura moderna e práticas de engenharia de elite.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
from prometheus_client import Counter, Histogram
from pydantic_settings import BaseSettings

from src.api.controllers import admin_controller, analytics_controller, strategy_controller
from src.api.middlewares.auth_middleware import AuthMiddleware
from src.api.middlewares.logging_middleware import LoggingMiddleware
from src.common.infrastructure.logging.logger import setup_logger
from src.common.utils.config import load_config

# Métricas para Prometheus
REQUEST_COUNT = Counter(
    "api_request_total", 
    "Total number of requests to the API",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", 
    "API Request latency in seconds",
    ["method", "endpoint"]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação.
    
    Esta função é responsável por inicializar recursos necessários 
    antes da inicialização do servidor e liberá-los quando o servidor é encerrado.
    """
    # Configuração inicial - carrega antes do startup
    config = load_config()
    setup_logger(config.LOG_LEVEL)
    
    # Inicializa conexões de banco de dados, clientes de cache, etc.
    logging.info("Inicializando recursos da API...")
    
    # Fornece o contexto para a aplicação
    yield
    
    # Cleanup ao encerrar
    logging.info("Encerrando recursos da API...")

# Criação da aplicação FastAPI
app = FastAPI(
    title="Neural Crypto Bot API",
    description="API para o Neural Crypto Bot, um bot de trading de criptomoedas avançado.",
    version="1.0.0",
    lifespan=lifespan
)

# Adiciona middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, defina origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(AuthMiddleware)

# Instrumentação para OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Registra as rotas dos controllers
app.include_router(strategy_controller.router, prefix="/api/v1/strategies", tags=["Strategies"])
app.include_router(analytics_controller.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(admin_controller.router, prefix="/api/v1/admin", tags=["Admin"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Manipulador global de exceções.
    
    Captura todas as exceções não tratadas e retorna uma resposta JSON apropriada.
    """
    logging.error(f"Exceção não tratada: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Erro interno do servidor. Por favor, contate o suporte."}
    )

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Endpoint para verificação de saúde da API.
    
    Retorna um status OK se a API estiver funcionando corretamente.
    """
    return {"status": "ok", "service": "Neural Crypto Bot API"}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)