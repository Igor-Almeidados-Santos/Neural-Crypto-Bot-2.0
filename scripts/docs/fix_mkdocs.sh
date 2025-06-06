#!/bin/bash
# Neural Crypto Bot 2.0 - Script de correção para MkDocs
# Este script resolve problemas comuns com o servidor MkDocs

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Diretórios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DOCS_DIR="$PROJECT_ROOT/docs"
CONFIG_FILE="$PROJECT_ROOT/mkdocs.yml"

# Funções de log
log_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} ℹ️ $1"
}

log_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${NC} ✅ $1"
}

log_warning() {
    echo -e "${YELLOW}${BOLD}[WARNING]${NC} ⚠️ $1"
}

log_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} ❌ $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}"
}

# Banner
echo -e "${CYAN}${BOLD}"
echo "=========================================================="
echo "     NEURAL CRYPTO BOT 2.0 - CORRETOR DE DOCUMENTAÇÃO     "
echo "=========================================================="
echo -e "${NC}\n"

# Verificar Python
log_step "Verificando ambiente Python"

if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    log_error "Python não encontrado. Por favor, instale Python 3.6+"
    exit 1
fi

log_success "Python encontrado: $($PYTHON_CMD --version)"

# Verificar se o diretório docs existe
if [ ! -d "$DOCS_DIR" ]; then
    log_warning "Diretório docs não encontrado. Criando..."
    mkdir -p "$DOCS_DIR"
fi

# Matar processos usando a porta 8000
log_step "Verificando processos existentes"

ports=(8000 8080)
for port in "${ports[@]}"; do
    if command -v lsof &>/dev/null; then
        PIDs=$(lsof -ti :$port 2>/dev/null || echo "")
        if [ -n "$PIDs" ]; then
            log_warning "Processos usando a porta $port encontrados. Terminando..."
            for PID in $PIDs; do
                kill -9 $PID 2>/dev/null || true
                log_info "Processo $PID finalizado"
            done
        fi
    elif command -v netstat &>/dev/null; then
        # Alternativa usando netstat
        PIDs=$(netstat -tuln | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 || echo "")
        if [ -n "$PIDs" ]; then
            log_warning "Processos usando a porta $port encontrados. Terminando..."
            for PID in $PIDs; do
                kill -9 $PID 2>/dev/null || true
                log_info "Processo $PID finalizado"
            done
        fi
    else
        log_warning "Não foi possível verificar processos (lsof/netstat não encontrados)"
    fi
done

# Gerar arquivos README.md para as seções
log_step "Criando arquivos README.md"

# Usar o script Python se disponível, ou criar manualmente
PYTHON_SCRIPT="$SCRIPT_DIR/create_readmes.py"

if [ -f "$PYTHON_SCRIPT" ]; then
    log_info "Executando script Python para criar READMEs..."
    $PYTHON_CMD "$PYTHON_SCRIPT"
else
    log_warning "Script Python não encontrado. Criando READMEs manualmente..."
    
    # Criar diretórios de seção se não existirem
    for i in $(seq -f "%02g" 1 12); do
        SECTION_DIR="$DOCS_DIR/$i"
        
        # Determinar o nome da seção
        case $i in
            "01") SECTION_NAME="getting-started" ;;
            "02") SECTION_NAME="architecture" ;;
            "03") SECTION_NAME="development" ;;
            "04") SECTION_NAME="trading" ;;
            "05") SECTION_NAME="machine-learning" ;;
            "06") SECTION_NAME="integrations" ;;
            "07") SECTION_NAME="operations" ;;
            "08") SECTION_NAME="api-reference" ;;
            "09") SECTION_NAME="tutorials" ;;
            "10") SECTION_NAME="legal-compliance" ;;
            "11") SECTION_NAME="community" ;;
            "12") SECTION_NAME="appendices" ;;
        esac
        
        FULL_DIR="$DOCS_DIR/$i-$SECTION_NAME"
        mkdir -p "$FULL_DIR"
        
        # Criar README.md básico
        README_PATH="$FULL_DIR/README.md"
        TITLE=$(echo "$SECTION_NAME" | sed 's/-/ /g' | awk '{for(i=1;i<=NF;i++)sub(/./,toupper(substr($i,1,1)),$i)}1')
        
        cat > "$README_PATH" << EOF
# $TITLE

This section contains documentation about $TITLE for Neural Crypto Bot 2.0.

## Overview

This documentation is under development.

## Topics

- Topic 1
- Topic 2
- Topic 3

---

*Last updated: $(date +%Y-%m-%d)*
EOF
        
        log_info "Criado README para: $i-$SECTION_NAME"
    done
fi

# Atualizar a configuração para usar uma porta alternativa
log_step "Atualizando configuração MkDocs"

if [ -f "$CONFIG_FILE" ]; then
    # Escolher porta aleatória entre 8001 e 8999
    NEW_PORT=$((RANDOM % 999 + 8001))
    
    # Verificar se já existe configuração de porta
    if grep -q "dev_addr:" "$CONFIG_FILE"; then
        # Atualizar porta existente
        sed -i.bak "s/dev_addr:.*/dev_addr: '127.0.0.1:$NEW_PORT'/" "$CONFIG_FILE"
    else
        # Adicionar configuração de porta
        sed -i.bak "/site_name:/a \\\n# Development server settings\ndev_addr: '127.0.0.1:$NEW_PORT'" "$CONFIG_FILE"
    fi
    
    log_success "Porta do servidor atualizada para: $NEW_PORT"
else
    log_error "Arquivo de configuração MkDocs não encontrado: $CONFIG_FILE"
    log_info "Execute o script de configuração primeiro para criar mkdocs.yml"
fi

# Verificar arquivo index.md
log_step "Verificando index.md"

INDEX_PATH="$DOCS_DIR/index.md"
if [ ! -f "$INDEX_PATH" ]; then
    log_warning "Arquivo index.md não encontrado. Criando..."
    
    cat > "$INDEX_PATH" << 'EOF'
# Neural Crypto Bot 2.0

Welcome to the official documentation for Neural Crypto Bot 2.0, an enterprise-grade cryptocurrency trading bot powered by advanced machine learning and quantitative finance algorithms.

## 🚀 Features

- **Advanced ML Models**: Deep learning and reinforcement learning for predictive trading
- **Multi-Exchange Support**: Trade across major exchanges with a unified API
- **Risk Management**: Sophisticated risk controls and portfolio optimization
- **Microservices Architecture**: Scalable and resilient system design
- **Real-time Analytics**: Monitor performance with customizable dashboards
- **Extensible Strategy Framework**: Create and backtest custom strategies

## 📋 Documentation Sections

- [Getting Started](01-getting-started/README.md)
- [Architecture](02-architecture/README.md)
- [Development](03-development/README.md)
- [Trading](04-trading/README.md)
- [Machine Learning](05-machine-learning/README.md)
- [Integrations](06-integrations/README.md)
- [Operations](07-operations/README.md)
- [API Reference](08-api-reference/README.md)
- [Tutorials](09-tutorials/README.md)
- [Legal & Compliance](10-legal-compliance/README.md)
- [Community](11-community/README.md)
- [Appendices](12-appendices/README.md)

## 🛠️ Technical Overview

Neural Crypto Bot 2.0 is built with modern technologies and follows best practices.

## 📚 Resources

- [GitHub Repository](https://github.com/neural-crypto-bot/neural-crypto-bot-2.0)

---

*Documentation powered by [MkDocs](https://www.mkdocs.org)*
EOF
    
    log_success "Arquivo index.md criado"
else
    log_success "Arquivo index.md encontrado"
fi

# Verificar se MkDocs está instalado
log_step "Verificando instalação do MkDocs"

if $PYTHON_CMD -m pip list | grep -q "mkdocs"; then
    log_success "MkDocs encontrado"
else
    log_warning "MkDocs não encontrado. Instalando..."
    $PYTHON_CMD -m pip install mkdocs mkdocs-material
    log_success "MkDocs instalado"
fi

# Conclusão
log_step "Correção concluída"

log_success "Problemas corrigidos:"
echo -e "${GREEN}1. Processos anteriores do MkDocs finalizados${NC}"
echo -e "${GREEN}2. Arquivos README.md criados para todas as seções${NC}"
echo -e "${GREEN}3. Porta do servidor atualizada para evitar conflitos${NC}"
echo -e "${GREEN}4. Arquivo index.md verificado/criado${NC}"

log_info "Agora você pode executar o servidor MkDocs com:"
echo -e "${CYAN}$PYTHON_CMD -m mkdocs serve${NC}"

if [ -f "$CONFIG_FILE" ] && grep -q "dev_addr:" "$CONFIG_FILE"; then
    PORT=$(grep "dev_addr:" "$CONFIG_FILE" | sed "s/.*:\([0-9]*\).*/\1/")
    echo -e "${CYAN}A documentação estará disponível em: http://127.0.0.1:$PORT${NC}"
else
    echo -e "${CYAN}A documentação estará disponível em: http://127.0.0.1:8000${NC}"
fi

echo -e "\n${GREEN}${BOLD}Correção de problemas concluída com sucesso!${NC}"