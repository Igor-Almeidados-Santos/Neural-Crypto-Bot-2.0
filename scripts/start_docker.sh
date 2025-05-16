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
    echo "Arquivo .env não encontrado. Criando a partir do .env.example..."
    cp .env.example .env
    echo "Por favor, edite o arquivo .env com suas configurações antes de continuar."
    exit 1
fi

# Inicia os serviços de infraestrutura
echo "Iniciando serviços de infraestrutura (PostgreSQL, Redis, Kafka)..."
docker-compose up -d postgres redis zookeeper kafka

echo "Aguardando a inicialização dos serviços..."
sleep 10

# Inicia os serviços da aplicação
echo "Iniciando serviços da aplicação..."
docker-compose up -d collector execution training api

echo "=== Ambiente Docker iniciado com sucesso! ==="
echo "Para visualizar os logs: docker-compose logs -f"
echo "Para parar os serviços: docker-compose down"
