# Dockerfile - Multi-stage build para Neural Crypto Bot 2.0
# Otimizado para produção com segurança e performance

# ================================
# Stage 1: Base Python com Poetry
# ================================
FROM python:3.11-slim as python-base

# Definir variáveis de ambiente para otimização
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"

# Configurar PATH para Poetry e venv
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Instalar dependências do sistema necessárias
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # Dependências para compilação
        build-essential \
        curl \
        # Dependências para PostgreSQL
        libpq-dev \
        # Dependências para computação científica
        libopenblas-dev \
        liblapack-dev \
        gfortran \
        # Dependências para processamento de imagem/gráficos
        libffi-dev \
        # Dependências para protobuf/grpc
        libprotobuf-dev \
        protobuf-compiler \
        # Ferramentas de rede
        netcat-openbsd \
        # Limpeza
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ================================
# Stage 2: Instalar Poetry e dependências
# ================================
FROM python-base as poetry-base

# Instalar Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Configurar diretório de trabalho
WORKDIR $PYSETUP_PATH

# Copiar arquivos de configuração do Poetry
COPY pyproject.toml poetry.lock* ./

# Instalar dependências (apenas produção)
RUN poetry install --only=main --no-root \
    && rm -rf $POETRY_CACHE_DIR

# ================================
# Stage 3: Imagem de desenvolvimento
# ================================
FROM poetry-base as development

# Instalar dependências de desenvolvimento
RUN poetry install --no-root

# Copiar código fonte
WORKDIR /app
COPY . .

# Criar usuário não-root para desenvolvimento
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expor porta padrão
EXPOSE 8000

# Comando padrão para desenvolvimento
CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ================================
# Stage 4: Imagem de produção (padrão)
# ================================
FROM python:3.11-slim as production

# Reusar variáveis de ambiente otimizadas
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    PYTHONPATH=/app \
    ENVIRONMENT=production

# Configurar PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Instalar apenas dependências de runtime necessárias
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # Runtime essencial
        libpq5 \
        libopenblas0 \
        liblapack3 \
        libgfortran5 \
        # Networking e curl para health checks
        curl \
        netcat-openbsd \
        # Timezone data
        tzdata \
        # CA certificates para HTTPS
        ca-certificates \
        # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copiar ambiente virtual da etapa anterior
COPY --from=poetry-base $VENV_PATH $VENV_PATH

# Configurar timezone padrão
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Criar usuário não-root para segurança
RUN groupadd -r appuser --gid=1000 \
    && useradd -r -g appuser --uid=1000 --home-dir=/app --shell=/bin/bash appuser \
    && mkdir -p /app /app/logs /app/data \
    && chown -R appuser:appuser /app

# Configurar diretório de trabalho
WORKDIR /app

# Copiar código fonte com ownership correto
COPY --chown=appuser:appuser . .

# Criar diretórios necessários
RUN mkdir -p logs data models feature_store \
    && chown -R appuser:appuser logs data models feature_store

# Mudar para usuário não-root
USER appuser

# Configurar variáveis de ambiente da aplicação
ENV LOG_LEVEL=INFO \
    MAX_WORKERS=4 \
    TIMEOUT_KEEP_ALIVE=2 \
    ACCESS_LOG=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expor porta
EXPOSE 8000

# Comando padrão - pode ser sobrescrito
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# ================================
# Stage 5: Worker/Collector específico
# ================================
FROM production as collector

# Configurações específicas para o collector
ENV SERVICE_TYPE=collector \
    LOG_LEVEL=INFO \
    WORKER_CONCURRENCY=8

# Health check específico para collector
HEALTHCHECK --interval=60s --timeout=15s --start-period=10s --retries=5 \
    CMD python -c "import src.data_collection.health_check; src.data_collection.health_check.check()" || exit 1

# Comando específico para collector
CMD ["python", "-m", "src.data_collection.main"]

# ================================
# Stage 6: Execution Engine específico
# ================================
FROM production as execution

# Configurações específicas para execution
ENV SERVICE_TYPE=execution \
    LOG_LEVEL=INFO \
    EXECUTION_TIMEOUT=30 \
    MAX_RETRY_ATTEMPTS=3

# Health check específico para execution
HEALTHCHECK --interval=45s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import src.execution_service.health_check; src.execution_service.health_check.check()" || exit 1

# Comando específico para execution
CMD ["python", "-m", "src.execution_service.main"]

# ================================
# Stage 7: Training Service específico
# ================================
FROM production as training

# Configurações específicas para training
ENV SERVICE_TYPE=training \
    LOG_LEVEL=INFO \
    MODEL_CACHE_SIZE=512MB \
    TRAINING_BATCH_SIZE=128

# Instalar dependências extras para ML (se necessário)
USER root
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # Dependências extras para ML
        libblas3 \
        libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Health check específico para training
HEALTHCHECK --interval=120s --timeout=20s --start-period=30s --retries=3 \
    CMD python -c "import src.model_training.health_check; src.model_training.health_check.check()" || exit 1

# Comando específico para training
CMD ["python", "-m", "src.model_training.main"]

# ================================
# Stage 8: Analytics Service específico  
# ================================
FROM production as analytics

# Configurações específicas para analytics
ENV SERVICE_TYPE=analytics \
    LOG_LEVEL=INFO \
    ANALYTICS_CACHE_TTL=300

# Health check específico para analytics
HEALTHCHECK --interval=90s --timeout=15s --start-period=20s --retries=3 \
    CMD python -c "import src.analytics.health_check; src.analytics.health_check.check()" || exit 1

# Comando específico para analytics
CMD ["python", "-m", "src.analytics.main"]

# ================================
# Stage 9: Testing específico
# ================================
FROM poetry-base as testing

# Instalar todas as dependências (incluindo dev)
RUN poetry install --no-root

# Configurar ambiente de teste
ENV ENVIRONMENT=testing \
    TESTING=true \
    LOG_LEVEL=DEBUG

# Copiar código fonte
WORKDIR /app
COPY . .

# Criar usuário para testes
RUN groupadd -r testuser && useradd -r -g testuser testuser \
    && chown -R testuser:testuser /app
USER testuser

# Comando para testes
CMD ["poetry", "run", "pytest", "tests/", "-v", "--cov=src"]

# ================================
# Metadata e Labels
# ================================
LABEL maintainer="Igor Almeida <igor.almeidasantos2020@gmail.com>" \
      version="2.0.0" \
      description="Neural Crypto Bot 2.0 - Advanced Trading System" \
      vendor="Neural Crypto Bot" \
      licenses="MIT" \
      url="https://github.com/your-username/Neural-Crypto-Bot-2.0" \
      documentation="https://github.com/your-username/Neural-Crypto-Bot-2.0/blob/main/README.md" \
      source="https://github.com/your-username/Neural-Crypto-Bot-2.0" \
      created="2024-01-01T00:00:00Z" \
      revision="${BUILD_REVISION:-unknown}" \
      build-date="${BUILD_DATE:-unknown}"

# Build args para informações de build
ARG BUILD_DATE
ARG BUILD_REVISION
ARG BUILD_VERSION=2.0.0

# Labels dinâmicos
LABEL build-date=$BUILD_DATE \
      revision=$BUILD_REVISION \
      version=$BUILD_VERSION