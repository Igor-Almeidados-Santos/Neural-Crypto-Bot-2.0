#!/bin/bash
# setup_configs.sh

echo "=== Configurando arquivos de ambiente para o Trading Bot ==="

# Cria .env.example
echo "Criando .env.example..."
cat > .env.example << EOF
# Configurações gerais
ENVIRONMENT=development  # development, testing, production
DEBUG=True
LOG_LEVEL=INFO
TIMEZONE=UTC

# Configurações de banco de dados
DATABASE_URL=postgresql://tradingbot:password@localhost:5432/tradingbot
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=tradingbot-group
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
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
    -   id: check-toml
    -   id: detect-private-key
    -   id: check-merge-conflict

-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.286
    hooks:
    -   id: ruff
        args: [--fix, --exit-non-zero-on-fix]

-   repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
    -   id: black
        language_version: python3.11

-   repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
    -   id: isort
        name: isort (python)

-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
    -   id: mypy
        additional_dependencies: [
            "types-requests",
            "types-PyYAML",
            "types-python-dateutil",
            "pydantic>=2.0.0",
        ]
EOF

# Cria README.md
echo "Criando README.md..."
cat > README.md << EOF
# Advanced Cryptocurrency Trading Bot

Um bot de trading de criptomoedas avançado com recursos de ML, desenvolvido com arquitetura moderna e práticas de engenharia de elite.

## Características

- Arquitetura modular baseada em Domain-Driven Design
- Integração com múltiplas exchanges via CCXT
- Processamento de dados e sinais em tempo real
- Modelos de ML avançados para previsão de mercado
- Estratégias de trading configuráveis e extensíveis
- Sistema robusto de gestão de risco
- Análise de performance e visualização de dados
- API REST/GraphQL para interação externa

## Requisitos

- Python 3.11+
- Docker e Docker Compose
- Poetry para gerenciamento de dependências

## Instalação

\`\`\`bash
# Clone o repositório
git clone https://github.com/your-org/crypto-trading-bot.git
cd crypto-trading-bot

# Instale as dependências
./scripts/setup_poetry.sh

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o arquivo .env com suas configurações

# Inicie os serviços
./scripts/start_docker.sh
\`\`\`

## Uso

[Documentação de uso detalhada]

## Arquitetura

[Descrição da arquitetura do sistema]

## Contribuição

[Diretrizes para contribuição]

## Licença

MIT
EOF

echo "✅ Configuração de ambiente concluída com sucesso!"
