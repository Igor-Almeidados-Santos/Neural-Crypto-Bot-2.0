#!/bin/bash
# setup_configs.sh

echo "=== Configurando arquivos de ambiente para o Neural Crypto Bot ==="

# Cria .env.example
echo "Criando .env.example..."
cat > .env.example << EOF
# Configurações gerais
ENVIRONMENT=development  # development, testing, production
DEBUG=True
LOG_LEVEL=INFO
TIMEZONE=UTC

# Configurações de banco de dados
DATABASE_URL=postgresql://neuralbot:password@localhost:5432/neuralcryptobot
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=ncb-group
KAFKA_AUTO_OFFSET_RESET=earliest

# Segurança
SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_SECONDS=86400
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Exchanges
BINANCE_API_KEY=
BINANCE_API_SECRET=
COINBASE_API_KEY=
COINBASE_API_SECRET=
KRAKEN_API_KEY=
KRAKEN_API_SECRET=

# Configurações de trading
DEFAULT_TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT
MAX_POSITION_SIZE=0.1  # Porcentagem do portfólio
MAX_LEVERAGE=3.0
MAX_DRAWDOWN_PERCENT=5.0
RISK_FREE_RATE=0.03

# Configurações de execução
ORDER_TIMEOUT_SECONDS=30
MAX_RETRY_ATTEMPTS=3
RETRY_DELAY_SECONDS=2

# Configurações de ML
MODEL_STORAGE_PATH=./models
FEATURE_STORE_PATH=./feature_store
BATCH_SIZE=128
EPOCHS=100
LEARNING_RATE=0.001
EARLY_STOPPING_PATIENCE=10

# Telemetria e monitoramento
ENABLE_TELEMETRY=True
PROMETHEUS_PORT=9090
JAEGER_AGENT_HOST=localhost
JAEGER_AGENT_PORT=6831
EOF

# Cria .env a partir de .env.example
cp .env.example .env

# Cria arquivo .pre-commit-config.yaml
echo "Criando .pre-commit-config.yaml..."
cat > .pre-commit-config.yaml << EOF
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-toml
    -   id: detect-private-key
    -   id: check-merge-conflict

-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.11
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]

-   repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
    -   id: black
        language_version: python3.11

-   repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
    -   id: isort
        name: isort (python)

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
    -   id: mypy
        additional_dependencies: [
            "types-requests",
            "types-PyYAML",
            "types-python-dateutil",
            "pydantic>=2.5.0",
        ]
EOF

# Cria README.md simples (vamos criar um mais completo posteriormente)
echo "Criando README.md básico..."
cat > README.md << EOF
# Neural Crypto Bot

Um bot de trading de criptomoedas avançado com recursos de ML, desenvolvido com arquitetura moderna e práticas de engenharia de elite.

_Detalhes completos serão adicionados posteriormente._
EOF

echo "✅ Configuração de ambiente concluída com sucesso!"