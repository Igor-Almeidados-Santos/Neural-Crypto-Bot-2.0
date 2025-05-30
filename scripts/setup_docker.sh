#!/bin/bash
# setup_docker.sh

echo "=== Configurando Docker para o Neural Crypto Bot ==="

# Detectar sistema operacional
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    MINGW*|MSYS*|CYGWIN*)    OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Sistema operacional detectado: $OS_TYPE"

# Cria diretório para Dockerfiles
mkdir -p deployment/docker

# Cria Dockerfile.api
echo "Criando Dockerfile.api..."
cat > deployment/docker/Dockerfile.api << EOF
# Dockerfile.api
FROM python:3.11-slim as python-base

# Configuração de ambiente não interativo
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on \\
    PIP_DEFAULT_TIMEOUT=100 \\
    POETRY_HOME="/opt/poetry" \\
    POETRY_VIRTUALENVS_IN_PROJECT=true \\
    POETRY_NO_INTERACTION=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$POETRY_HOME/bin:\$VENV_PATH/bin:\$PATH"

# Instalação de dependências do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    build-essential \\
    curl \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Instalação do Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Cópia dos arquivos de configuração do Poetry
WORKDIR \$PYSETUP_PATH
COPY pyproject.toml poetry.lock* ./

# Instalação das dependências do projeto
RUN poetry install --without dev --no-root

# Imagem final
FROM python:3.11-slim as production

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$VENV_PATH/bin:\$PATH"

# Instalação de dependências mínimas do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    libpq5 \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Cópia do ambiente virtual da etapa anterior
COPY --from=python-base \$VENV_PATH \$VENV_PATH

# Cópia do código fonte
WORKDIR /app
COPY . .

# Exposição da porta da API
EXPOSE 8000

# Configuração de variáveis de ambiente
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Comando para iniciar a API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Cria Dockerfile.collector
echo "Criando Dockerfile.collector..."
cat > deployment/docker/Dockerfile.collector << EOF
# Dockerfile.collector
FROM python:3.11-slim as python-base

# Configuração de ambiente não interativo
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on \\
    PIP_DEFAULT_TIMEOUT=100 \\
    POETRY_HOME="/opt/poetry" \\
    POETRY_VIRTUALENVS_IN_PROJECT=true \\
    POETRY_NO_INTERACTION=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$POETRY_HOME/bin:\$VENV_PATH/bin:\$PATH"

# Instalação de dependências do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    build-essential \\
    curl \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Instalação do Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Cópia dos arquivos de configuração do Poetry
WORKDIR \$PYSETUP_PATH
COPY pyproject.toml poetry.lock* ./

# Instalação das dependências do projeto
RUN poetry install --without dev --no-root

# Imagem final
FROM python:3.11-slim as production

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$VENV_PATH/bin:\$PATH"

# Instalação de dependências mínimas do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    libpq5 \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Cópia do ambiente virtual da etapa anterior
COPY --from=python-base \$VENV_PATH \$VENV_PATH

# Cópia do código fonte
WORKDIR /app
COPY . .

# Configuração de variáveis de ambiente
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Comando para iniciar o serviço de coleta
CMD ["python", "-m", "src.data_collection.main"]
EOF

# Cria Dockerfile.execution
echo "Criando Dockerfile.execution..."
cat > deployment/docker/Dockerfile.execution << EOF
# Dockerfile.execution
FROM python:3.11-slim as python-base

# Configuração de ambiente não interativo
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on \\
    PIP_DEFAULT_TIMEOUT=100 \\
    POETRY_HOME="/opt/poetry" \\
    POETRY_VIRTUALENVS_IN_PROJECT=true \\
    POETRY_NO_INTERACTION=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$POETRY_HOME/bin:\$VENV_PATH/bin:\$PATH"

# Instalação de dependências do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    build-essential \\
    curl \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Instalação do Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Cópia dos arquivos de configuração do Poetry
WORKDIR \$PYSETUP_PATH
COPY pyproject.toml poetry.lock* ./

# Instalação das dependências do projeto
RUN poetry install --without dev --no-root

# Imagem final
FROM python:3.11-slim as production

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$VENV_PATH/bin:\$PATH"

# Instalação de dependências mínimas do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    libpq5 \\
    libopenblas0 \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Cópia do ambiente virtual da etapa anterior
COPY --from=python-base \$VENV_PATH \$VENV_PATH

# Cópia do código fonte
WORKDIR /app
COPY . .

# Configuração de variáveis de ambiente
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Comando para iniciar o serviço de execução
CMD ["python", "-m", "src.execution_service.main"]
EOF

# Cria Dockerfile.training
echo "Criando Dockerfile.training..."
cat > deployment/docker/Dockerfile.training << EOF
# Dockerfile.training
FROM python:3.11-slim as python-base

# Configuração de ambiente não interativo
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=off \\
    PIP_DISABLE_PIP_VERSION_CHECK=on \\
    PIP_DEFAULT_TIMEOUT=100 \\
    POETRY_HOME="/opt/poetry" \\
    POETRY_VIRTUALENVS_IN_PROJECT=true \\
    POETRY_NO_INTERACTION=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$POETRY_HOME/bin:\$VENV_PATH/bin:\$PATH"

# Instalação de dependências do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    build-essential \\
    curl \\
    libopenblas-dev \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Instalação do Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Cópia dos arquivos de configuração do Poetry
WORKDIR \$PYSETUP_PATH
COPY pyproject.toml poetry.lock* ./

# Instalação das dependências do projeto
RUN poetry install --without dev --no-root

# Imagem final
FROM python:3.11-slim as production

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYSETUP_PATH="/opt/pysetup" \\
    VENV_PATH="/opt/pysetup/.venv"

ENV PATH="\$VENV_PATH/bin:\$PATH"

# Instalação de dependências mínimas do sistema
RUN apt-get update \\
    && apt-get install --no-install-recommends -y \\
    libpq5 \\
    libopenblas0 \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Cópia do ambiente virtual da etapa anterior
COPY --from=python-base \$VENV_PATH \$VENV_PATH

# Cópia do código fonte
WORKDIR /app
COPY . .

# Configuração de variáveis de ambiente
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Comando para iniciar o serviço de treinamento
CMD ["python", "-m", "src.model_training.main"]
EOF

# Cria .dockerignore para reduzir o tamanho do contexto de build
echo "Criando .dockerignore..."
cat > .dockerignore << EOF
# Git
.git
.gitignore

# Virtual environments
.venv/
venv/
ENV/
env/

# Python
__pycache__/
*.py[cod]
*\$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/

# Jupyter Notebook
.ipynb_checkpoints

# Node.js
node_modules/
npm-debug.log*

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Data and models
data/
models/
*.h5
*.pkl

# Documentation
docs/
README.md

# CI/CD
.github/
.gitlab-ci.yml

# Test coverage
htmlcov/
.coverage

# Backup files
*.bak
*.backup

# Temporary files
tmp/
temp/
*.tmp

# Large files
*.zip
*.tar.gz
*.rar

# Docker build context exclusions
deployment/
scripts/
.env*
docker-compose*.yml
Dockerfile*
EOF

# Cria docker-compose.yml (sem a versão obsoleta)
echo "Criando docker-compose.yml..."
cat > docker-compose.yml << EOF
services:
  # Serviço de API
  api:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.api
    ports:
      - "8001:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      kafka:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://neuralbot:password@postgres:5432/neuralcryptobot
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    networks:
      - neuralbot-network
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  # Serviço de coleta de dados
  collector:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.collector
    depends_on:
      postgres:
        condition: service_healthy
      kafka:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://neuralbot:password@postgres:5432/neuralcryptobot
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    networks:
      - neuralbot-network
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  # Serviço de execução de ordens
  execution:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.execution
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      kafka:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://neuralbot:password@postgres:5432/neuralcryptobot
      - REDIS_URL=redis://redis:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    networks:
      - neuralbot-network
    restart: unless-stopped
    volumes:
      - ./logs:/app/logs

  # Serviço de treinamento de modelos
  training:
    build:
      context: .
      dockerfile: deployment/docker/Dockerfile.training
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      - DATABASE_URL=postgresql://neuralbot:password@postgres:5432/neuralcryptobot
      - REDIS_URL=redis://redis:6379/0
      - MODEL_STORAGE_PATH=/app/models
    volumes:
      - model-storage:/app/models
      - ./logs:/app/logs
    networks:
      - neuralbot-network
    restart: unless-stopped

  # Banco de dados PostgreSQL com TimescaleDB
  postgres:
    image: timescale/timescaledb:latest-pg16
    ports:
      - "5433:5432"
    environment:
      - POSTGRES_USER=neuralbot
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=neuralcryptobot
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - neuralbot-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U neuralbot -d neuralcryptobot"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis para cache e armazenamento em memória
  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - neuralbot-network
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Kafka para mensageria
  kafka:
    image: confluentinc/cp-kafka:7.5.1
    ports:
      - "9092:9092"
    environment:
      - KAFKA_BROKER_ID=1
      - KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=PLAINTEXT:PLAINTEXT,PLAINTEXT_HOST:PLAINTEXT
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:29092,PLAINTEXT_HOST://localhost:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
      - KAFKA_AUTO_CREATE_TOPICS_ENABLE=true
      - KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS=0
      - KAFKA_TRANSACTION_STATE_LOG_MIN_ISR=1
      - KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR=1
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
    depends_on:
      zookeeper:
        condition: service_healthy
    networks:
      - neuralbot-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "kafka-topics --bootstrap-server localhost:9092 --list"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Zookeeper para Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.1
    ports:
      - "2181:2181"
    environment:
      - ZOOKEEPER_CLIENT_PORT=2181
      - ZOOKEEPER_TICK_TIME=2000
    networks:
      - neuralbot-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "echo srvr | nc localhost 2181 | grep Mode"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Grafana para visualização de dados
  grafana:
    image: grafana/grafana:10.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=neuralbot
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - neuralbot-network
    depends_on:
      - prometheus
    restart: unless-stopped

  # Prometheus para coleta de métricas
  prometheus:
    image: prom/prometheus:v2.47.0
    ports:
      - "9090:9090"
    volumes:
      - ./deployment/prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    networks:
      - neuralbot-network
    restart: unless-stopped

networks:
  neuralbot-network:
    driver: bridge

volumes:
  postgres-data:
  redis-data:
  model-storage:
  grafana-data:
  prometheus-data:
EOF

echo "✅ Configuração Docker concluída com sucesso!"
echo ""
echo "Principais correções aplicadas:"
echo "- ✅ Corrigido libopenblas-base para libopenblas0"
echo "- ✅ Adicionado .dockerignore para reduzir contexto de build"
echo "- ✅ Removido 'version' obsoleto do docker-compose.yml"
echo "- ✅ Configurado PostgreSQL na porta 5433 (externa)"
echo "- ✅ Adicionado KAFKA_ZOOKEEPER_CONNECT"
echo ""
echo "Para aplicar as mudanças:"
echo "1. Execute: docker-compose down"
echo "2. Execute: docker system prune -f"
echo "3. Execute: ./scripts/start_docker.sh"