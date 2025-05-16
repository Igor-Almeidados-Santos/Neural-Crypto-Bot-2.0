#!/bin/bash
# setup_scripts.sh

echo "=== Configurando scripts utilitários para o Trading Bot ==="

mkdir -p scripts

# Cria script de setup do Poetry
echo "Criando scripts/setup_poetry.sh..."
cat > scripts/setup_poetry.sh << EOF
#!/bin/bash
# setup_poetry.sh - Script para instalação e configuração do Poetry

set -e

echo "=== Configurando ambiente de desenvolvimento para o bot de trading ==="

# Verifica se o Python 3.11+ está instalado
if ! command -v python3.11 &> /dev/null; then
    echo "Python 3.11 ou superior é necessário. Por favor, instale-o antes de continuar."
    exit 1
fi

# Instala o Poetry se não estiver instalado
if ! command -v poetry &> /dev/null; then
    echo "Instalando Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="\$HOME/.local/bin:\$PATH"
else
    echo "Poetry já está instalado."
fi

# Configura o Poetry para criar o ambiente virtual no diretório do projeto
poetry config virtualenvs.in-project true

# Instala as dependências do projeto
echo "Instalando dependências do projeto..."
poetry install

# Ativa o ambiente virtual
echo "Ativando o ambiente virtual..."
source \$(poetry env info --path)/bin/activate

# Configura hooks do pre-commit
echo "Configurando hooks do pre-commit..."
poetry run pre-commit install

echo "=== Configuração concluída com sucesso! ==="
echo "Para ativar o ambiente virtual, execute: poetry shell"
echo "Para executar comandos no ambiente virtual: poetry run <comando>"
EOF

# Cria script de inicialização do Docker
echo "Criando scripts/start_docker.sh..."
cat > scripts/start_docker.sh << EOF
#!/bin/bash
# start_docker.sh - Script para iniciar o ambiente Docker

set -e

echo "=== Iniciando ambiente Docker para o bot de trading ==="

# Verifica se o Docker está instalado
if ! command -v docker &> /dev/null; then
    echo "Docker não está instalado. Por favor, instale-o antes de continuar."
    exit 1
fi

# Verifica se o Docker Compose está instalado
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose não está instalado. Por favor, instale-o antes de continuar."
    exit 1
fi

# Verifica se o arquivo .env existe
if [ ! -f .env ]; then
    echo "Arquivo .env não encontrado
