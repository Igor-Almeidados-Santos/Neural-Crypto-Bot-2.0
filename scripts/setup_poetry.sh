#!/bin/bash
# setup_poetry.sh

echo "=== Configurando Poetry para o Trading Bot ==="

# Verifica se o Poetry está instalado
if ! command -v poetry &> /dev/null; then
    echo "Instalando Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "Poetry já está instalado."
fi

# Configura o Poetry para criar venv no diretório do projeto
poetry config virtualenvs.in-project true

# Cria o arquivo pyproject.toml
echo "Criando pyproject.toml..."
cat > pyproject.toml << EOF
[tool.poetry]
name = "Neural-Crypto-Bot"
version = "1.0.0"
description = "Advanced cryptocurrency trading bot with ML capabilities"
authors = ["Igor Almeida Dos Santos <igor.almeidasantos2020@gmail.com.com>"]
readme = "README.md"
license = "MIT"
packages = [{include = "src"}]

[tool.poetry.dependencies]
python = ">=3.11,<3.12"

# APIs e frameworks
fastapi = "^0.103.1"
uvicorn = {extras = ["standard"], version = "^0.23.2"}
starlette = "^0.27.0"
graphql-core = "^3.2.3"
strawberry-graphql = "^0.195.0"
pydantic = "^2.3.0"

# Comunicação assíncrona e concorrência
asyncio = "^3.4.3"
httpx = "^0.24.1"
grpcio = "^1.57.0"
protobuf = "^4.24.3"
websockets = "^11.0.3"

# Processamento de dados e análise numérica
numpy = "^1.24.3"
scipy = "^1.11.2"
pandas = "^2.1.0"
polars = "^0.19.0"
numba = "^0.57.1"
pyarrow = "^13.0.0"

# Machine Learning
scikit-learn = "^1.3.0"
pytorch-lightning = "^2.0.9"
torch = "^2.0.1"
transformers = "^4.33.1"
xgboost = "^1.7.6"
optuna = "^3.3.0"
mlflow = "^2.7.1"
ray = {extras = ["tune"], version = "^2.6.3"}
tensorboard = "^2.14.0"

# Processamento de séries temporais
statsmodels = "^0.14.0"
prophet = "^1.1.4"
tslearn = "^0.6.2"

# Infraestrutura e armazenamento de dados
sqlalchemy = "^2.0.20"
alembic = "^1.12.0"
psycopg2-binary = "^2.9.7"
redis = "^5.0.0"
pymongo = "^4.5.0"
arctic = "^1.79.4"
kafka-python = "^2.0.2"
confluent-kafka = "^2.2.0"

# Trading e Exchange APIs
ccxt = "^3.1.40"

# Segurança e autenticação
pyjwt = "^2.8.0"
passlib = "^1.7.4"
python-jose = {extras = ["cryptography"], version = "^3.3.0"}
cryptography = "^41.0.3"

# Observabilidade e logging
opentelemetry-api = "^1.19.0"
opentelemetry-sdk = "^1.19.0"
prometheus-client = "^0.17.1"
structlog = "^23.1.0"
python-json-logger = "^2.0.7"

# Utilitários
python-dotenv = "^1.0.0"
pydantic-settings = "^2.0.3"
tenacity = "^8.2.3"
pytz = "^2023.3"
click = "^8.1.7"
tqdm = "^4.66.1"

# Visualização
matplotlib = "^3.7.2"
seaborn = "^0.12.2"
plotly = "^5.16.1"
bokeh = "^3.2.2"

[tool.poetry.group.dev.dependencies]
# Testes
pytest = "^7.4.0"
pytest-asyncio = "^0.21.1"
pytest-cov = "^4.1.0"
hypothesis = "^6.82.2"

# Desenvolvimento
black = "^23.7.0"
flake8 = "^6.1.0"
mypy = "^1.5.1"
isort = "^5.12.0"
pre-commit = "^3.3.3"
ruff = "^0.0.286"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.black]
line-length = 100
target-version = ["py311"]
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
strict_optional = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "--cov=src --cov-report=term-missing --cov-report=xml:coverage.xml"
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "system: System tests",
]

[tool.ruff]
line-length = 100
target-version = "py311"
select = ["E", "F", "B", "I"]
ignore = []
EOF

# Instala as dependências
echo "Instalando dependências com Poetry..."
poetry install

echo "✅ Poetry configurado com sucesso!"
