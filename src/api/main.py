from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

# Corrigindo os imports com base na nova estrutura
from api.controllers import strategy_controller, analytics_controller, admin_controller
from common.utils.settings import settings
from common.infrastructure.logging.logger import setup_logging

# Lifespan: Gerencia o que acontece na inicialização e no desligamento da aplicação
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup do logging na inicialização
    setup_logging(level=settings.api.LOG_LEVEL)
    logging.info("Oasis Trading System API inicializando...")
    # Aqui iriam pools de conexão, etc.
    yield
    # Código de limpeza ao desligar
    logging.info("Oasis Trading System API desligando...")


app = FastAPI(
    title="Oasis Trading System API",
    description="API de alta performance para o sistema de trading algorítmico.",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware de tratamento de erro global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Captura todas as exceções não tratadas e retorna uma resposta 500 padronizada.
    Previne que o sistema quebre por erros inesperados.
    """
    logging.critical(f"Erro não tratado na requisição: {request.method} {request.url}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "detail": "Ocorreu um erro inesperado no servidor."},
    )

# Inclusão dos roteadores de forma limpa
app.include_router(strategy_controller.router, prefix="/api/v1", tags=["Strategies"])
app.include_router(analytics_controller.router, prefix="/api/v1", tags=["Analytics"])
app.include_router(admin_controller.router, prefix="/api/v1", tags=["Admin"])


@app.get("/health", tags=["System"])
def health_check():
    """Endpoint de health check para monitoramento."""
    return {"status": "ok", "version": app.version}