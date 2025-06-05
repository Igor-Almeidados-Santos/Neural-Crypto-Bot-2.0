#!/bin/bash
# scripts/docs/check_prerequisites.sh - Verificação completa de pré-requisitos
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Verificação de dependências e ambiente

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ================================
# CONFIGURAÇÕES ESPECÍFICAS
# ================================

readonly MIN_DISK_SPACE_GB=3
readonly MIN_MEMORY_MB=2048
readonly REQUIRED_PORTS=(8000 8080 3000)

# ================================
# VERIFICAÇÕES DE PRÉ-REQUISITOS
# ================================

check_system_requirements() {
    log_step "Verificando Requisitos do Sistema"
    
    local issues=()
    
    # Verificar sistema operacional
    local os_type
    os_type=$(detect_os)
    
    case "$os_type" in
        "ubuntu"|"debian"|"fedora"|"centos"|"rhel"|"macos")
            log_success "Sistema operacional suportado: $os_type"
            ;;
        "windows")
            log_warning "Windows detectado. Use WSL2 para melhor compatibilidade"
            ;;
        *)
            log_warning "Sistema operacional não testado: $os_type"
            issues+=("Sistema operacional não totalmente suportado")
            ;;
    esac
    
    # Verificar arquitetura
    local arch
    arch=$(detect_arch)
    
    case "$arch" in
        "x86_64"|"arm64")
            log_success "Arquitetura suportada: $arch"
            ;;
        *)
            log_warning "Arquitetura não testada: $arch"
            issues+=("Arquitetura não totalmente suportada")
            ;;
    esac
    
    # Verificar espaço em disco
    if ! check_disk_space "$MIN_DISK_SPACE_GB"; then
        local available
        available=$(check_disk_space 0)
        log_error "Espaço em disco insuficiente: ${available}GB (mínimo: ${MIN_DISK_SPACE_GB}GB)"
        issues+=("Espaço em disco insuficiente")
    else
        local available
        available=$(check_disk_space 0)
        log_success "Espaço em disco suficiente: ${available}GB"
    fi
    
    # Verificar conectividade
    if check_internet_connectivity; then
        log_success "Conectividade com internet: OK"
    else
        log_error "Sem conectividade com internet"
        issues+=("Conectividade com internet necessária")
    fi
    
    return ${#issues[@]}
}

check_python_environment() {
    log_step "Verificando Ambiente Python"
    
    local issues=()
    
    # Verificar Python
    if PYTHON_CMD=$(check_python_version); then
        local python_version
        python_version=$($PYTHON_CMD --version 2>&1)
        log_success "Python encontrado: $python_version"
        save_config "PYTHON_CMD" "$PYTHON_CMD"
        
        # Verificar pip
        if $PYTHON_CMD -m pip --version >/dev/null 2>&1; then
            local pip_version
            pip_version=$($PYTHON_CMD -m pip --version)
            log_success "pip disponível: $pip_version"
        else
            log_error "pip não encontrado ou não funcional"
            issues+=("pip não disponível")
        fi
        
        # Verificar venv
        if $PYTHON_CMD -m venv --help >/dev/null 2>&1; then
            log_success "venv disponível"
        else
            log_warning "venv não disponível (pode impactar isolamento)"
        fi
        
    else
        log_error "Python ${PYTHON_MIN_VERSION}+ não encontrado"
        issues+=("Python ${PYTHON_MIN_VERSION}+ obrigatório")
        show_python_installation_guide
    fi
    
    return ${#issues[@]}
}

check_poetry_environment() {
    log_step "Verificando Poetry"
    
    local issues=()
    
    if command_exists poetry; then
        local poetry_version
        poetry_version=$(poetry --version 2>&1)
        log_success "Poetry encontrado: $poetry_version"
        
        # Verificar configuração do Poetry
        local poetry_venv_path
        poetry_venv_path=$(poetry config virtualenvs.path 2>/dev/null || echo "default")
        log_info "Poetry virtualenvs path: $poetry_venv_path"
        
        # Verificar se estamos em um projeto Poetry válido
        if [[ -f "$PROJECT_ROOT/pyproject.toml" ]]; then
            log_success "Projeto Poetry válido encontrado"
            
            # Verificar dependências do pyproject.toml
            if grep -q "mkdocs" "$PROJECT_ROOT/pyproject.toml"; then
                log_info "Dependências de documentação já configuradas"
            else
                log_info "Dependências de documentação serão adicionadas"
            fi
        else
            log_warning "pyproject.toml não encontrado"
            issues+=("Projeto Poetry não inicializado")
        fi
        
    else
        log_error "Poetry não encontrado"
        issues+=("Poetry obrigatório")
        show_poetry_installation_guide
    fi
    
    return ${#issues[@]}
}

check_git_environment() {
    log_step "Verificando Git"
    
    local issues=()
    
    if command_exists git; then
        local git_version
        git_version=$(git --version)
        log_success "Git encontrado: $git_version"
        
        # Verificar se estamos em um repositório Git
        if git rev-parse --git-dir >/dev/null 2>&1; then
            log_success "Repositório Git detectado"
            
            # Verificar configuração do Git
            local git_user_name
            local git_user_email
            git_user_name=$(git config user.name 2>/dev/null || echo "não configurado")
            git_user_email=$(git config user.email 2>/dev/null || echo "não configurado")
            
            log_info "Git user.name: $git_user_name"
            log_info "Git user.email: $git_user_email"
            
            if [[ "$git