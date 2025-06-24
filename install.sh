#!/bin/bash

# Aborta o script imediatamente se um comando falhar.
set -e

# Função para checar se um comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Função de log
log_info() {
    echo "INFO: $1"
}

log_error() {
    echo "ERROR: $1" >&2
    exit 1
}

# --- Verificação de Pré-requisitos ---
log_info "Verificando pré-requisitos essenciais..."

if ! command_exists git; then
    log_error "Git não está instalado. Por favor, instale o Git para continuar."
fi

if ! command_exists docker; then
    log_error "Docker não está instalado. Por favor, instale o Docker para continuar."
fi
if ! docker-compose --version >/dev/null 2>&1; then
    log_error "Docker Compose não está instalado ou não está no PATH. Por favor, instale-o."
fi

if ! command_exists poetry; then
    log_error "Poetry não está instalado. Instale-o com: 'curl -sSL https://install.python-poetry.org | python3 -'"
fi

log_info "✅ Todos os pré-requisitos foram encontrados."

# --- Instalação ---
log_info "Iniciando a instalação via Makefile..."
if ! make install; then
    log_error "A instalação das dependências via 'make install' falhou."
fi

# --- Configuração do Ambiente ---
if [ ! -f .env ]; then
    log_info "Arquivo .env não encontrado. Copiando de .env.example..."
    if [ -f .env.example ]; then
        cp .env.example .env
        log_info "✅ Arquivo .env criado. Por favor, edite-o com suas chaves de API e configurações."
    else
        log_error "Arquivo .env.example não encontrado. Não foi possível criar o arquivo .env."
    fi
fi

echo ""
log_info "🎉 Instalação concluída com sucesso!"
log_info "Para iniciar os serviços em background, use: make docker-up"
log_info "Para iniciar a API em modo de desenvolvimento, use: make run-api"
log_info "Para ver todos os comandos disponíveis, use: make help"