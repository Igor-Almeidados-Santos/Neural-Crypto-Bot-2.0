"""
Neural Crypto Bot - API Main Module

Este é o ponto de entrada para a API do Neural Crypto Bot, um bot de trading de criptomoedas 
avançado utilizando arquitetura moderna e práticas de engenharia de elite.
"""
import logging
from contextlib import asynccontextmanager
import json
import redis

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
from prometheus_client import Counter, Histogram
from pydantic_settings import BaseSettings

from src.api.controllers import admin_controller, analytics_controller, strategy_controller
from src.api.controllers.v2 import strategy_controller_v2, analytics_controller_v2
from src.api.middlewares.auth_middleware import AuthMiddleware
from src.api.middlewares.logging_middleware import LoggingMiddleware
from src.api.middlewares.rate_limit_middleware import RateLimitMiddleware
from src.api.middlewares.cache_middleware import CacheMiddleware
from src.api.middlewares.version_middleware import VersionMiddleware
from src.api.version import APIVersion, VersionedEndpoint
from src.common.infrastructure.logging.logger import get_logger
from src.common.infrastructure.cache.cache_manager import get_cache
from src.common.utils.config import get_settings

# Configure o logger principal
logger = get_logger("api")

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
    config = get_settings()
    
    # Inicializa conexões de banco de dados, clientes de cache, etc.
    logger.info("Inicializando recursos da API...")
    
    # Inicializa o sistema de cache
    cache = get_cache()
    logger.info("Sistema de cache inicializado")
    
    # Fornece o contexto para a aplicação
    yield
    
    # Cleanup ao encerrar
    logger.info("Encerrando recursos da API...")

# Lista de endpoints obsoletos
DEPRECATED_ENDPOINTS = [
    VersionedEndpoint(
        path="/api/v1/strategies/{strategy_id}/execute",
        min_version="v2",
        deprecated=True,
        redirect_to="/api/v2/strategies/{strategy_id}/execute"
    ),
    VersionedEndpoint(
        path="/api/v1/analytics/dashboard",
        min_version="v2",
        deprecated=True,
        redirect_to="/api/v2/analytics/dashboards"
    ),
]

# Configurações de tags para documentação
API_TAGS_METADATA = [
    {
        "name": "Auth",
        "description": "Operações relacionadas a autenticação e autorização."
    },
    {
        "name": "Strategies",
        "description": "Gerenciamento e execução de estratégias de trading."
    },
    {
        "name": "Analytics",
        "description": "Análises e métricas de performance."
    },
    {
        "name": "Admin",
        "description": "Funcionalidades administrativas e monitoramento."
    },
    {
        "name": "Health",
        "description": "Verificação de saúde da API."
    },
]

# Criação da aplicação FastAPI
app = FastAPI(
    title="Neural Crypto Bot API",
    description="""
    API para o Neural Crypto Bot, um bot de trading de criptomoedas avançado.
    
    Esta API fornece endpoints para gerenciar estratégias de trading, 
    executar análises e monitorar o desempenho do sistema.
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url=None,  # Desativamos o docs_url padrão para personalizar
    redoc_url=None, # Desativamos o redoc_url padrão para personalizar
    openapi_url="/api/openapi.json"
)

# Adiciona middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, defina origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adiciona o middleware de versionamento (deve ser o primeiro para lidar com rotas obsoletas)
app.add_middleware(
    VersionMiddleware,
    deprecated_endpoints=DEPRECATED_ENDPOINTS,
    default_version=APIVersion.V2
)

# Adiciona middleware de cache
app.add_middleware(
    CacheMiddleware,
    cache_ttl=60,  # 60 segundos por padrão
    include_paths=[
        "/api/*/strategies",
        "/api/*/analytics/performance",
        "/api/*/analytics/market",
    ],
    exclude_paths=[
        "/health",
        "/api/*/auth/*",
        "/api/*/admin/*",
    ]
)

# Adiciona middleware de rate limiting
settings = get_settings()
redis_client = None

# Inicializa o cliente Redis para rate limiting se estiver em produção
if settings.ENVIRONMENT in ("production", "staging"):
    try:
        redis_client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            username=settings.REDIS_USERNAME,
            password=settings.REDIS_PASSWORD,
        )
        logger.info("Cliente Redis inicializado para rate limiting")
    except Exception as e:
        logger.warning(f"Não foi possível inicializar o cliente Redis para rate limiting: {str(e)}")

app.add_middleware(
    RateLimitMiddleware,
    redis_client=redis_client,
    rate_limits={
        "*": (120, 60),  # 120 requisições por minuto por padrão
        "/api/*/strategies/*/backtest": (10, 60),  # 10 backtests por minuto
        "/api/*/analytics/performance": (20, 60),  # 20 análises de performance por minuto
        "/api/*/analytics/reports": (5, 60),       # 5 relatórios por minuto
    },
    exclude_paths=[
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    ]
)

# Adiciona middleware de logging
app.add_middleware(LoggingMiddleware)

# Adiciona middleware de autenticação
app.add_middleware(
    AuthMiddleware,
    exclude_paths=[
        "/health",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/auth/login",
        "/api/v1/auth/refresh",
        "/api/v2/auth/login",
        "/api/v2/auth/refresh",
    ]
)

# Instrumentação para OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Rotas v1 - Compatibilidade com versões anteriores
app.include_router(strategy_controller.router, prefix="/api/v1/strategies", tags=["Strategies"])
app.include_router(analytics_controller.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(admin_controller.router, prefix="/api/v1/admin", tags=["Admin"])

# Rotas v2 - Nova versão
app.include_router(strategy_controller_v2.router, prefix="/api/v2/strategies", tags=["Strategies"])
app.include_router(analytics_controller_v2.router, prefix="/api/v2/analytics", tags=["Analytics"])
app.include_router(admin_controller.router, prefix="/api/v2/admin", tags=["Admin"])  # Reutilizamos o controller admin por enquanto

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Manipulador global de exceções.
    
    Captura todas as exceções não tratadas e retorna uma resposta JSON apropriada.
    """
    logger.error(f"Exceção não tratada: {str(exc)}", exc_info=True)
    
    # Adiciona o ID de requisição nos logs para facilitar troubleshooting
    request_id = getattr(request.state, "request_id", "unknown")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Erro interno do servidor. Por favor, contate o suporte.",
            "request_id": request_id
        }
    )

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Endpoint para verificação de saúde da API.
    
    Retorna um status OK se a API estiver funcionando corretamente,
    juntamente com informações sobre os componentes do sistema.
    """
    # Verifica status de componentes críticos
    components_status = {
        "api": "operational",
        "database": _check_database_health(),
        "cache": _check_cache_health(),
        "message_broker": _check_message_broker_health()
    }
    
    # Calcula o status geral com base nos componentes
    overall_status = "degraded" if any(s != "operational" for s in components_status.values()) else "operational"
    
    return {
        "status": overall_status,
        "service": "Neural Crypto Bot API",
        "version": "2.0.0",
        "components": components_status
    }

def _check_database_health() -> str:
    """Verifica a saúde da conexão com o banco de dados."""
    try:
        # Aqui você implementaria uma verificação real com o banco de dados
        # Por enquanto, simulamos como funcional
        return "operational"
    except Exception as e:
        logger.error(f"Erro ao verificar saúde do banco de dados: {str(e)}")
        return "degraded"

def _check_cache_health() -> str:
    """Verifica a saúde do sistema de cache."""
    try:
        cache = get_cache()
        # Uma verificação simples seria definir e ler um valor
        asyncio.run(cache.set("health_check", "ok", ttl=10))
        value = asyncio.run(cache.get("health_check"))
        return "operational" if value == "ok" else "degraded"
    except Exception as e:
        logger.error(f"Erro ao verificar saúde do cache: {str(e)}")
        return "degraded"

def _check_message_broker_health() -> str:
    """Verifica a saúde do message broker."""
    try:
        # Aqui você implementaria uma verificação real com o message broker
        # Por enquanto, simulamos como funcional
        return "operational"
    except Exception as e:
        logger.error(f"Erro ao verificar saúde do message broker: {str(e)}")
        return "degraded"

# Endpoints personalizados para documentação da API
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Endpoint personalizado para a documentação Swagger."""
    return get_swagger_ui_html(
        openapi_url="/api/openapi.json",
        title="Neural Crypto Bot API - Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_favicon_url="/static/favicon.ico",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Endpoint personalizado para a documentação ReDoc."""
    return get_redoc_html(
        openapi_url="/api/openapi.json",
        title="Neural Crypto Bot API - Documentation",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
        redoc_favicon_url="/static/favicon.ico",
    )

# Customização do esquema OpenAPI para incluir versões
@app.get("/api/openapi.json", include_in_schema=False)
async def get_api_schema():
    """Endpoint personalizado para fornecer o esquema OpenAPI."""
    # Obtém o esquema base
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=API_TAGS_METADATA
    )
    
    # Adiciona informações de versionamento
    openapi_schema["info"]["x-api-versions"] = APIVersion.all()
    openapi_schema["info"]["x-api-latest-version"] = APIVersion.latest()
    
    # Adiciona detalhes sobre as APIs descontinuadas
    deprecated_info = []
    for endpoint in DEPRECATED_ENDPOINTS:
        deprecated_info.append({
            "path": endpoint.path,
            "min_version": endpoint.min_version,
            "redirect_to": endpoint.redirect_to
        })
    
    openapi_schema["info"]["x-api-deprecated"] = deprecated_info
    
    return openapi_schema

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    # Certifica-se de que as bibliotecas assíncronas funcionam corretamente
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Inicia o servidor com hot-reload em desenvolvimento
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)