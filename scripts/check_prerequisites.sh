#!/bin/bash
# check_prerequisites.sh

echo "=== Verificando pré-requisitos para o Trading Bot ==="

# Verifica Python 3.11+
if command -v python3.11 &> /dev/null; then
    echo "✅ Python 3.11+ instalado"
    python3.11 --version
else
    echo "❌ Python 3.11+ não encontrado"
    echo "Por favor, instale o Python 3.11 ou superior:"
    echo "Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-dev python3.11-venv"
    echo "macOS: brew install python@3.11"
    exit 1
fi

# Verifica Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker instalado"
    docker --version
else
    echo "❌ Docker não encontrado"
    echo "Por favor, instale o Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Verifica Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose instalado"
    docker-compose --version
else
    echo "❌ Docker Compose não encontrado"
    echo "Por favor, instale o Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

# Verifica Git
if command -v git &> /dev/null; then
    echo "✅ Git instalado"
    git --version
else
    echo "❌ Git não encontrado"
    echo "Por favor, instale o Git:"
    echo "Ubuntu/Debian: sudo apt update && sudo apt install git"
    echo "macOS: brew install git"
    exit 1
fi

echo "✅ Todos os pré-requisitos estão instalados!"
