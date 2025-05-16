#!/bin/bash
# install.sh

set -e  # Encerra o script se algum comando falhar

echo "=== Iniciando instalação do Trading Bot ==="

# Executa scripts na ordem correta
echo "Passo 1: Verificando pré-requisitos..."
bash ./scripts/check_prerequisites.sh

echo "Passo 2: Configurando Poetry..."
bash ./scripts/setup_poetry.sh

echo "Passo 3: Configurando Docker..."
bash ./scripts/setup_docker.sh

echo "Passo 4: Configurando arquivos de ambiente..."
bash ./scripts/setup_configs.sh

echo "Passo 5: Configurando scripts utilitários..."
bash ./scripts/setup_scripts.sh

echo "Passo 6: Configurando arquivos de domínio base..."
bash ./scripts/setup_base_domain.sh

# Torna os scripts executáveis
chmod +x scripts/*.sh

echo "=== Instalação concluída com sucesso! ==="
echo "Para instalar as dependências, execute: ./scripts/setup_poetry.sh"
echo "Para iniciar o ambiente Docker, execute: ./scripts/start_docker.sh"
