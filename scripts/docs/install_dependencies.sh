#!/bin/bash
# scripts/docs/install_dependencies.sh - Instalação completa de dependências
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Instalação enterprise-grade de dependências

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ================================
# CONFIGURAÇÕES ESPECÍFICAS
# ================================

readonly DEPENDENCIES_VERSION="2.0.0"
readonly MAX_RETRY_ATTEMPTS=3
readonly TIMEOUT_SECONDS=300

# URLs e versões de dependências
readonly POETRY_INSTALLER_URL="https://install.python-poetry.org"
readonly NODE_VERSION="18.17.0"
readonly NPM_GLOBAL_PACKAGES=(
    "markdownlint-cli2@^0.10.0"
    "markdown-link-check@^3.11.0"
    "@mermaid-js/mermaid-cli@^10.6.0"
)

# ================================
# VERIFICAÇÃO DE DEPENDÊNCIAS PYTHON
# ================================

check_python_compatibility() {
    log_step "Verificando Compatibilidade Python"
    
    local python_cmd
    if ! python_cmd=$(check_python_version); then
        log_error "Python ${PYTHON_MIN_VERSION}+ não encontrado"
        show_python_installation_guide
        return 1
    fi
    
    log_success "Python compatível encontrado: $($python_cmd --version)"
    
    # Verificar módulos essenciais
    local essential_modules=(
        "pip"
        "venv"
        "ssl"
        "json"
        "urllib"
    )
    
    for module in "${essential_modules[@]}"; do
        if $python_cmd -c "import $module" 2>/dev/null; then
            log_debug "Módulo $module: disponível"
        else
            log_error "Módulo Python essencial não encontrado: $module"
            return 1
        fi
    done
    
    save_config "PYTHON_CMD" "$python_cmd"
    return 0
}

install_poetry() {
    log_step "Instalando/Verificando Poetry"
    
    if command_exists poetry; then
        local poetry_version
        poetry_version=$(poetry --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
        log_success "Poetry já instalado: v$poetry_version"
        
        # Verificar se a versão é compatível (>=1.4.0)
        if version_compare "$poetry_version" "1.4.0"; then
            log_success "Versão do Poetry é compatível"
        else
            log_warning "Versão do Poetry pode ser muito antiga, considerando atualização"
        fi
        
        return 0
    fi
    
    log_info "Instalando Poetry..."
    
    # Download e instalação do Poetry
    local temp_installer="/tmp/install-poetry.py"
    
    if download_with_retry "$POETRY_INSTALLER_URL" "$temp_installer"; then
        local python_cmd
        python_cmd=$(load_config "PYTHON_CMD")
        
        log_info "Executando instalador do Poetry..."
        if $python_cmd "$temp_installer" --yes; then
            log_success "Poetry instalado com sucesso"
            
            # Adicionar Poetry ao PATH se necessário
            export PATH="$HOME/.local/bin:$PATH"
            
            # Verificar instalação
            if command_exists poetry; then
                local installed_version
                installed_version=$(poetry --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
                log_success "Poetry v$installed_version verificado"
            else
                log_error "Poetry instalado mas não encontrado no PATH"
                log_info "Adicione ao seu ~/.bashrc ou ~/.zshrc:"
                log_info "export PATH=\"\$HOME/.local/bin:\$PATH\""
                return 1
            fi
        else
            log_error "Falha na instalação do Poetry"
            return 1
        fi
        
        # Limpar arquivo temporário
        rm -f "$temp_installer"
    else
        log_error "Falha no download do instalador do Poetry"
        return 1
    fi
    
    return 0
}

configure_poetry() {
    log_step "Configurando Poetry"
    
    # Configurações recomendadas para o Poetry
    local poetry_configs=(
        "virtualenvs.create true"
        "virtualenvs.in-project true"
        "virtualenvs.prefer-active-python true"
        "installer.parallel true"
        "installer.max-workers 4"
        "cache-dir $PROJECT_ROOT/.poetry-cache"
    )
    
    for config in "${poetry_configs[@]}"; do
        local key value
        key=$(echo "$config" | cut -d' ' -f1)
        value=$(echo "$config" | cut -d' ' -f2-)
        
        if poetry config "$key" "$value"; then
            log_debug "Configuração Poetry: $key = $value"
        else
            log_warning "Falha ao configurar Poetry: $key"
        fi
    done
    
    # Verificar configurações
    log_info "Configurações atuais do Poetry:"
    poetry config --list | while read -r line; do
        log_debug "  $line"
    done
    
    log_success "Poetry configurado"
    return 0
}

add_documentation_dependencies() {
    log_step "Adicionando Dependências de Documentação"
    
    # Verificar se pyproject.toml existe
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_error "pyproject.toml não encontrado"
        return 1
    fi
    
    # Fazer backup do pyproject.toml
    local backup_file
    backup_file=$(backup_file "$PROJECT_ROOT/pyproject.toml")
    
    # Verificar se já existe seção de docs
    if grep -q "\[tool.poetry.group.docs.dependencies\]" "$PROJECT_ROOT/pyproject.toml"; then
        log_info "Seção de documentação já existe no pyproject.toml"
    else
        log_info "Adicionando seção de documentação ao pyproject.toml"
        
        cat >> "$PROJECT_ROOT/pyproject.toml" << 'EOF'

[tool.poetry.group.docs.dependencies]
# Core MkDocs
mkdocs = "^1.5.3"
mkdocs-material = "^9.4.8"

# Essential Plugins
mkdocs-git-revision-date-localized-plugin = "^1.2.1"
mkdocs-git-committers-plugin-2 = "^2.2.2"
mkdocs-minify-plugin = "^0.7.1"
mkdocs-redirects = "^1.2.1"
mkdocs-macros-plugin = "^1.0.5"
mkdocs-awesome-pages-plugin = "^2.9.2"
mkdocs-glightbox = "^0.3.4"
mkdocs-exclude = "^1.0.2"

# API Documentation
mkdocs-swagger-ui-tag = "^0.6.6"
mkdocstrings = {extras = ["python"], version = "^0.24.0"}
mkdocs-gen-files = "^0.5.0"
mkdocs-literate-nav = "^0.6.1"
mkdocs-section-index = "^0.3.8"

# Versioning and Publishing
mike = "^2.0.0"
ghp-import = "^2.1.0"

# Enhancement Libraries
pymdown-extensions = "^10.4"
markdown-include = "^0.8.1"
pillow = "^10.1.0"
cairosvg = "^2.7.1"

# Quality and Validation
markdownlint-cli2 = "^0.11.0"

# Development Tools
livereload = "^2.6.3"
watchdog = "^3.0.0"

# Additional Tools
jinja2 = "^3.1.2"
pyyaml = "^6.0.1"
requests = "^2.31.0"
EOF
        
        log_success "Dependências de documentação adicionadas"
    fi
    
    return 0
}

install_python_dependencies() {
    log_step "Instalando Dependências Python"
    
    local start_time=$(date +%s)
    
    # Instalar dependências principais
    log_info "Instalando dependências principais..."
    if poetry install --with docs; then
        log_success "Dependências principais instaladas"
    else
        log_error "Falha na instalação de dependências principais"
        return 1
    fi
    
    # Instalar dependências opcionais
    log_info "Instalando dependências opcionais..."
    if poetry install --with dev --with docs --extras "all"; then
        log_success "Dependências opcionais instaladas"
    else
        log_warning "Algumas dependências opcionais falharam (continuando)"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Instalação Python concluída em ${duration}s"
    save_config "PYTHON_DEPS_INSTALL_TIME" "$duration"
    
    return 0
}

# ================================
# DEPENDÊNCIAS OPCIONAIS (NODE.JS)
# ================================

check_node_environment() {
    log_step "Verificando Ambiente Node.js"
    
    if ! command_exists node; then
        log_info "Node.js não encontrado (opcional para funcionalidades avançadas)"
        return 1
    fi
    
    local node_version
    node_version=$(node --version | sed 's/v//')
    log_success "Node.js encontrado: v$node_version"
    
    # Verificar npm
    if command_exists npm; then
        local npm_version
        npm_version=$(npm --version)
        log_success "npm encontrado: v$npm_version"
    else
        log_warning "npm não encontrado"
        return 1
    fi
    
    return 0
}

install_node_dependencies() {
    log_step "Instalando Dependências Node.js (Opcionais)"
    
    if ! check_node_environment; then
        log_info "Pulando instalação de dependências Node.js"
        return 0
    fi
    
    # Criar package.json se não existir
    if [[ ! -f "$PROJECT_ROOT/package.json" ]]; then
        log_info "Criando package.json..."
        
        cat > "$PROJECT_ROOT/package.json" << 'EOF'
{
  "name": "neural-crypto-bot-docs",
  "version": "2.0.0",
  "description": "Documentation tools for Neural Crypto Bot 2.0",
  "private": true,
  "scripts": {
    "docs:serve": "mkdocs serve",
    "docs:build": "mkdocs build",
    "docs:lint": "markdownlint-cli2 \"docs/**/*.md\"",
    "docs:link-check": "markdown-link-check docs/**/*.md",
    "mermaid": "mmdc"
  },
  "devDependencies": {},
  "keywords": ["documentation", "mkdocs", "neural-crypto-bot"],
  "author": "Neural Crypto Bot Team",
  "license": "MIT"
}
EOF
    fi
    
    # Instalar pacotes globais
    for package in "${NPM_GLOBAL_PACKAGES[@]}"; do
        local package_name="${package%%@*}"
        
        if npm list -g "$package_name" >/dev/null 2>&1; then
            log_info "Pacote já instalado: $package_name"
        else
            log_info "Instalando pacote global: $package"
            if npm install -g "$package"; then
                log_success "Pacote instalado: $package_name"
            else
                log_warning "Falha ao instalar: $package_name"
            fi
        fi
    done
    
    log_success "Dependências Node.js processadas"
    return 0
}

# ================================
# FUNÇÕES AUXILIARES
# ================================

download_with_retry() {
    local url="$1"
    local output="$2"
    local attempts=0
    
    while [[ $attempts -lt $MAX_RETRY_ATTEMPTS ]]; do
        ((attempts++))
        
        log_debug "Tentativa $attempts de download: $url"
        
        if curl -fsSL --max-time $TIMEOUT_SECONDS "$url" -o "$output"; then
            log_debug "Download bem-sucedido: $output"
            return 0
        elif wget --timeout=$TIMEOUT_SECONDS --tries=1 "$url" -O "$output" 2>/dev/null; then
            log_debug "Download bem-sucedido via wget: $output"
            return 0
        fi
        
        log_warning "Tentativa $attempts falhou, tentando novamente..."
        sleep 2
    done
    
    log_error "Falha no download após $MAX_RETRY_ATTEMPTS tentativas: $url"
    return 1
}

version_compare() {
    local version1="$1"
    local version2="$2"
    
    # Comparação simples de versões (assumindo formato X.Y.Z)
    local v1_parts=(${version1//./ })
    local v2_parts=(${version2//./ })
    
    for i in {0..2}; do
        local v1_part=${v1_parts[i]:-0}
        local v2_part=${v2_parts[i]:-0}
        
        if [[ $v1_part -gt $v2_part ]]; then
            return 0
        elif [[ $v1_part -lt $v2_part ]]; then
            return 1
        fi
    done
    
    return 0  # Versões iguais
}

verify_installation() {
    log_step "Verificando Instalação"
    
    local verification_errors=()
    
    # Verificar Poetry
    if ! command_exists poetry; then
        verification_errors+=("Poetry não encontrado após instalação")
    fi
    
    # Verificar ambiente virtual do Poetry
    if ! poetry env info >/dev/null 2>&1; then
        verification_errors+=("Ambiente virtual do Poetry não criado")
    fi
    
    # Verificar dependências críticas
    local critical_deps=(
        "mkdocs"
        "mkdocs-material"
        "mkdocstrings"
    )
    
    for dep in "${critical_deps[@]}"; do
        if ! poetry run python -c "import ${dep//-/_}" 2>/dev/null; then
            verification_errors+=("Dependência crítica não encontrada: $dep")
        fi
    done
    
    # Relatório de verificação
    if [[ ${#verification_errors[@]} -eq 0 ]]; then
        log_success "Todas as dependências verificadas com sucesso!"
        
        # Mostrar informações do ambiente
        echo -e "\n${CYAN}${BOLD}📦 Ambiente de Dependências:${NC}"
        
        if command_exists poetry; then
            echo -e "${WHITE}├── Poetry: $(poetry --version)${NC}"
            echo -e "${WHITE}├── Python: $(poetry run python --version)${NC}"
            echo -e "${WHITE}├── Virtual Env: $(poetry env info --path)${NC}"
        fi
        
        if command_exists node; then
            echo -e "${WHITE}├── Node.js: $(node --version)${NC}"
            echo -e "${WHITE}├── npm: $(npm --version)${NC}"
        fi
        
        # Contar dependências instaladas
        local total_deps
        total_deps=$(poetry show 2>/dev/null | wc -l)
        echo -e "${WHITE}└── Pacotes Python: $total_deps instalados${NC}"
        
        return 0
    else
        log_error "Problemas encontrados na verificação:"
        for error in "${verification_errors[@]}"; do
            log_error "  - $error"
        done
        return 1
    fi
}

create_dependency_report() {
    log_step "Gerando Relatório de Dependências"
    
    local report_file="$PROJECT_ROOT/.docs-dependencies-report.md"
    
    cat > "$report_file" << EOF
# Neural Crypto Bot 2.0 - Dependency Report

Generated on: $(date)
System: $(detect_os) $(detect_arch)

## Python Environment

EOF
    
    if command_exists poetry; then
        echo "### Poetry Environment" >> "$report_file"
        echo "- Version: $(poetry --version)" >> "$report_file"
        echo "- Virtual Environment: $(poetry env info --path 2>/dev/null || echo 'Not created')" >> "$report_file"
        echo "- Python Version: $(poetry run python --version 2>/dev/null || echo 'Not available')" >> "$report_file"
        echo "" >> "$report_file"
        
        echo "### Installed Packages" >> "$report_file"
        echo '```' >> "$report_file"
        poetry show 2>/dev/null >> "$report_file" || echo "No packages found" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    if command_exists node; then
        echo "## Node.js Environment" >> "$report_file"
        echo "- Node.js Version: $(node --version)" >> "$report_file"
        echo "- npm Version: $(npm --version)" >> "$report_file"
        echo "" >> "$report_file"
        
        echo "### Global Packages" >> "$report_file"
        echo '```' >> "$report_file"
        npm list -g --depth=0 2>/dev/null >> "$report_file" || echo "No global packages" >> "$report_file"
        echo '```' >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    echo "## System Information" >> "$report_file"
    echo "- OS: $(detect_os)" >> "$report_file"
    echo "- Architecture: $(detect_arch)" >> "$report_file"
    echo "- User: $(whoami)" >> "$report_file"
    echo "- Project Root: $PROJECT_ROOT" >> "$report_file"
    
    log_success "Relatório de dependências salvo: $report_file"
}

setup_development_environment() {
    log_step "Configurando Ambiente de Desenvolvimento"
    
    # Criar script de ativação do ambiente
    local activate_script="$PROJECT_ROOT/activate-docs-env.sh"
    
    cat > "$activate_script" << 'EOF'
#!/bin/bash
# Neural Crypto Bot 2.0 - Documentation Environment Activation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 Ativando ambiente de documentação..."

# Verificar Poetry
if ! command -v poetry >/dev/null 2>&1; then
    echo "❌ Poetry não encontrado. Execute primeiro: ./scripts/docs/install_dependencies.sh"
    exit 1
fi

# Ativar ambiente virtual do Poetry
echo "📦 Ativando ambiente virtual..."
poetry shell

echo "✅ Ambiente ativado! Comandos disponíveis:"
echo "  • mkdocs serve     - Servidor local de desenvolvimento"
echo "  • mkdocs build     - Build da documentação"
echo "  • poetry show      - Listar dependências"
echo "  • deactivate       - Sair do ambiente virtual"
EOF
    
    chmod +x "$activate_script"
    log_success "Script de ativação criado: activate-docs-env.sh"
    
    # Criar aliases úteis
    local aliases_file="$PROJECT_ROOT/.docs-aliases"
    
    cat > "$aliases_file" << 'EOF'
# Neural Crypto Bot 2.0 - Documentation Aliases
# Source this file: source .docs-aliases

# MkDocs shortcuts
alias docs-serve='poetry run mkdocs serve'
alias docs-build='poetry run mkdocs build'
alias docs-deploy='poetry run mkdocs gh-deploy'

# Quality tools
alias docs-lint='poetry run markdownlint-cli2 "docs/**/*.md"'
alias docs-check='poetry run python scripts/docs/validators/check_links.py'

# Development
alias docs-clean='rm -rf site .docs-cache'
alias docs-status='poetry show | grep mkdocs'
alias docs-update='poetry update'

# Quick navigation
alias cd-docs='cd docs'
alias cd-scripts='cd scripts/docs'

echo "📚 Documentation aliases loaded!"
echo "Available commands: docs-serve, docs-build, docs-lint, docs-check"
EOF
    
    log_success "Aliases de documentação criados: .docs-aliases"
    
    # Configurar pre-commit hooks se disponível
    if command_exists pre-commit; then
        log_info "Configurando pre-commit hooks..."
        if poetry run pre-commit install 2>/dev/null; then
            log_success "Pre-commit hooks configurados"
        else
            log_debug "Pre-commit não configurado (opcional)"
        fi
    fi
}

cleanup_installation() {
    log_step "Limpeza Pós-Instalação"
    
    # Limpar cache temporário
    local temp_files=(
        "/tmp/install-poetry.py"
        "/tmp/poetry-installer-*"
        "$PROJECT_ROOT/.poetry-cache/tmp/*"
    )
    
    for temp_file in "${temp_files[@]}"; do
        if [[ -f "$temp_file" ]] || [[ -d "$temp_file" ]]; then
            rm -rf "$temp_file" 2>/dev/null || true
            log_debug "Removido arquivo temporário: $(basename "$temp_file")"
        fi
    done
    
    # Otimizar cache do Poetry
    if command_exists poetry; then
        poetry cache clear --all pypi 2>/dev/null || true
        log_debug "Cache do Poetry otimizado"
    fi
    
    log_success "Limpeza concluída"
}

# ================================
# FUNÇÃO PRINCIPAL
# ================================

main() {
    # Inicializar sistema
    init_docs_system
    
    log_step "Iniciando Instalação de Dependências"
    log_info "Versão do instalador: $DEPENDENCIES_VERSION"
    
    local start_time=$(date +%s)
    local installation_steps=(
        "check_python_compatibility"
        "install_poetry"
        "configure_poetry"
        "add_documentation_dependencies"
        "install_python_dependencies"
        "install_node_dependencies"
        "verify_installation"
        "setup_development_environment"
        "create_dependency_report"
        "cleanup_installation"
    )
    
    local completed_steps=0
    local total_steps=${#installation_steps[@]}
    
    # Executar cada etapa
    for step in "${installation_steps[@]}"; do
        log_substep "Executando: $step"
        
        if $step; then
            ((completed_steps++))
            show_progress $completed_steps $total_steps "Instalando dependências"
        else
            log_error "Falha na etapa: $step"
            
            # Tentar recuperação automática para algumas falhas
            case "$step" in
                "install_node_dependencies")
                    log_warning "Falha nas dependências Node.js (continuando sem elas)"
                    ((completed_steps++))
                    ;;
                "setup_development_environment")
                    log_warning "Falha na configuração do ambiente (funcionalidade básica disponível)"
                    ((completed_steps++))
                    ;;
                *)
                    log_error "Falha crítica, interrompendo instalação"
                    return 1
                    ;;
            esac
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Salvar estatísticas
    save_config "DEPENDENCIES_INSTALL_TIME" "$duration"
    save_config "DEPENDENCIES_VERSION" "$DEPENDENCIES_VERSION"
    save_config "INSTALLATION_COMPLETED_STEPS" "$completed_steps"
    save_config "INSTALLATION_TOTAL_STEPS" "$total_steps"
    
    # Resultado final
    if [[ $completed_steps -eq $total_steps ]]; then
        log_success "Instalação de dependências concluída com sucesso!"
        
        echo -e "\n${GREEN}${BOLD}🎉 Dependências Instaladas!${NC}"
        echo -e "${GREEN}Tempo total: ${duration}s${NC}"
        echo -e "${GREEN}Etapas concluídas: $completed_steps/$total_steps${NC}"
        
        echo -e "\n${CYAN}${BOLD}📋 Próximos Passos:${NC}"
        echo -e "1. ${BOLD}Ativar ambiente:${NC} source activate-docs-env.sh"
        echo -e "2. ${BOLD}Carregar aliases:${NC} source .docs-aliases"
        echo -e "3. ${BOLD}Testar instalação:${NC} docs-serve"
        echo -e "4. ${BOLD}Continuar setup:${NC} ./scripts/docs/create_configuration.sh"
        
        return 0
    else
        log_warning "Instalação parcialmente concluída: $completed_steps/$total_steps etapas"
        log_info "Verifique o log para detalhes: $LOG_FILE"
        
        return 1
    fi
}

# ================================
# EXECUÇÃO
# ================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi