#!/bin/bash
# scripts/docs/create_directories.sh - Criação da estrutura completa de diretórios
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Estrutura de documentação enterprise-grade

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ================================
# CONFIGURAÇÕES ESPECÍFICAS
# ================================

readonly DIRECTORY_STRUCTURE_VERSION="2.0.0"
readonly TOTAL_EXPECTED_DIRS=120

# ================================
# ESTRUTURA DE DIRETÓRIOS PRINCIPAL
# ================================

# Definir estrutura completa de diretórios
declare -a DIRECTORY_STRUCTURE=(
    # Diretórios principais de documentação
    "docs"
    "docs/overrides"
    "docs/stylesheets"
    "docs/javascripts"
    "docs/images"
    "docs/templates"
    "docs/data"
    "docs/macros"
    "docs/examples"
    
    # Assets organizados
    "docs/assets"
    "docs/assets/icons"
    "docs/assets/images"
    "docs/assets/videos"
    "docs/assets/downloads"
    "docs/assets/diagrams"
    "docs/assets/screenshots"
    "docs/assets/logos"
    "docs/assets/fonts"
    
    # 01 - Getting Started
    "docs/01-getting-started"
    "docs/01-getting-started/installation"
    "docs/01-getting-started/configuration"
    "docs/01-getting-started/troubleshooting"
    "docs/01-getting-started/migration"
    "docs/01-getting-started/quickstart"
    
    # 02 - Architecture
    "docs/02-architecture"
    "docs/02-architecture/components"
    "docs/02-architecture/patterns"
    "docs/02-architecture/decision-records"
    "docs/02-architecture/diagrams"
    "docs/02-architecture/specifications"
    
    # 03 - Development
    "docs/03-development"
    "docs/03-development/testing"
    "docs/03-development/debugging"
    "docs/03-development/ci-cd"
    "docs/03-development/tools"
    "docs/03-development/guidelines"
    "docs/03-development/contributing"
    
    # 04 - Trading
    "docs/04-trading"
    "docs/04-trading/strategies"
    "docs/04-trading/risk-management"
    "docs/04-trading/execution"
    "docs/04-trading/backtesting"
    "docs/04-trading/optimization"
    "docs/04-trading/indicators"
    "docs/04-trading/signals"
    
    # 05 - Machine Learning
    "docs/05-machine-learning"
    "docs/05-machine-learning/models"
    "docs/05-machine-learning/feature-engineering"
    "docs/05-machine-learning/training"
    "docs/05-machine-learning/monitoring"
    "docs/05-machine-learning/experiments"
    "docs/05-machine-learning/pipelines"
    "docs/05-machine-learning/datasets"
    
    # 06 - Integrations
    "docs/06-integrations"
    "docs/06-integrations/exchanges"
    "docs/06-integrations/data-providers"
    "docs/06-integrations/notifications"
    "docs/06-integrations/external-apis"
    "docs/06-integrations/protocols"
    "docs/06-integrations/webhooks"
    
    # 07 - Operations
    "docs/07-operations"
    "docs/07-operations/deployment"
    "docs/07-operations/monitoring"
    "docs/07-operations/logging"
    "docs/07-operations/security"
    "docs/07-operations/backup"
    "docs/07-operations/scaling"
    "docs/07-operations/maintenance"
    "docs/07-operations/disaster-recovery"
    
    # 08 - API Reference
    "docs/08-api-reference"
    "docs/08-api-reference/endpoints"
    "docs/08-api-reference/webhooks"
    "docs/08-api-reference/sdks"
    "docs/08-api-reference/generated"
    "docs/08-api-reference/examples"
    "docs/08-api-reference/schemas"
    
    # 09 - Tutorials
    "docs/09-tutorials"
    "docs/09-tutorials/getting-started"
    "docs/09-tutorials/advanced"
    "docs/09-tutorials/case-studies"
    "docs/09-tutorials/video-tutorials"
    "docs/09-tutorials/workshops"
    "docs/09-tutorials/exercises"
    
    # 10 - Legal & Compliance
    "docs/10-legal-compliance"
    "docs/10-legal-compliance/licenses"
    "docs/10-legal-compliance/regulations"
    "docs/10-legal-compliance/audit"
    "docs/10-legal-compliance/policies"
    "docs/10-legal-compliance/certifications"
    
    # 11 - Community
    "docs/11-community"
    "docs/11-community/forums"
    "docs/11-community/events"
    "docs/11-community/resources"
    "docs/11-community/partnerships"
    
    # 12 - Appendices
    "docs/12-appendices"
    "docs/12-appendices/reference"
    "docs/12-appendices/tools"
    "docs/12-appendices/resources"
    "docs/12-appendices/archive"
    
    # Scripts de documentação
    "scripts/docs"
    "scripts/docs/templates"
    "scripts/docs/generators"
    "scripts/docs/validators"
    "scripts/docs/utils"
    "scripts/docs/themes"
    "scripts/docs/plugins"
    
    # Logs e cache
    ".docs-setup-logs"
    ".docs-cache"
    ".docs-build"
    
    # Ambientes e configurações
    "docs/environments"
    "docs/environments/development"
    "docs/environments/staging"
    "docs/environments/production"
    
    # Localizações (preparação para i18n)
    "docs/locales"
    "docs/locales/en"
    "docs/locales/pt"
    "docs/locales/es"
    
    # Diretórios para build e deploy
    "site"
    "site/assets"
    "site/static"
)

# ================================
# FUNÇÕES DE CRIAÇÃO DE DIRETÓRIOS
# ================================

create_base_directories() {
    log_step "Criando Estrutura Base de Diretórios"
    
    local created_count=0
    local total_count=${#DIRECTORY_STRUCTURE[@]}
    
    # Criar todos os diretórios
    for dir in "${DIRECTORY_STRUCTURE[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        
        if create_directory "$full_path" "$(basename "$dir")"; then
            ((created_count++))
        fi
        
        # Mostrar progresso
        if [[ $((created_count % 10)) -eq 0 ]] || [[ $created_count -eq $total_count ]]; then
            show_progress $created_count $total_count "Criando diretórios"
        fi
    done
    
    log_success "Estrutura base criada: $created_count/$total_count diretórios"
    save_config "DIRECTORIES_CREATED" "$created_count"
    
    return 0
}

create_special_directories() {
    log_step "Criando Diretórios Especiais"
    
    # Diretórios para diferentes ambientes
    local environments=("development" "staging" "production")
    
    for env in "${environments[@]}"; do
        local env_dir="$PROJECT_ROOT/docs/environments/$env"
        create_directory "$env_dir/configs" "environment config"
        create_directory "$env_dir/assets" "environment assets"
        create_directory "$env_dir/overrides" "environment overrides"
    done
    
    # Diretórios para diferentes tipos de documentos
    local doc_types=(
        "api-specs"
        "user-guides"
        "developer-guides"
        "admin-guides"
        "troubleshooting"
        "changelogs"
    )
    
    for doc_type in "${doc_types[@]}"; do
        create_directory "$PROJECT_ROOT/docs/templates/$doc_type" "template type"
    done
    
    # Diretórios para assets organizados por tipo
    local asset_types=(
        "architecture-diagrams"
        "flow-charts"
        "sequence-diagrams"
        "entity-relationship"
        "network-diagrams"
        "deployment-diagrams"
    )
    
    for asset_type in "${asset_types[@]}"; do
        create_directory "$PROJECT_ROOT/docs/assets/diagrams/$asset_type" "diagram type"
    done
    
    log_success "Diretórios especiais criados"
}

create_placeholder_files() {
    log_step "Criando Arquivos Placeholder"
    
    # Criar arquivos .gitkeep para manter diretórios vazios no Git
    local empty_dirs=(
        "docs/assets/videos"
        "docs/assets/downloads"
        "docs/assets/screenshots"
        "docs/assets/fonts"
        ".docs-cache"
        ".docs-build"
        "docs/environments/development"
        "docs/environments/staging"
        "docs/environments/production"
        "docs/locales/en"
        "docs/locales/pt"
        "docs/locales/es"
        "site/assets"
        "site/static"
    )
    
    for dir in "${empty_dirs[@]}"; do
        local gitkeep_file="$PROJECT_ROOT/$dir/.gitkeep"
        if [[ ! -f "$gitkeep_file" ]]; then
            echo "# Manter diretório no Git" > "$gitkeep_file"
            log_debug "Criado .gitkeep em: $dir"
        fi
    done
    
    # Criar arquivos README.md em diretórios principais
    create_section_readme_files
    
    log_success "Arquivos placeholder criados"
}

create_section_readme_files() {
    log_substep "Criando READMEs de Seção"
    
    # Definir seções principais com descrições
    declare -A SECTIONS=(
        ["01-getting-started"]="Guias de início rápido, instalação e configuração inicial"
        ["02-architecture"]="Documentação da arquitetura do sistema e decisões de design"
        ["03-development"]="Guias de desenvolvimento, testes e contribuição"
        ["04-trading"]="Estratégias de trading, gestão de risco e execução"
        ["05-machine-learning"]="Modelos de ML, feature engineering e pipelines"
        ["06-integrations"]="Integrações com exchanges, APIs e serviços externos"
        ["07-operations"]="Deployment, monitoramento e operações de produção"
        ["08-api-reference"]="Referência completa da API e documentação técnica"
        ["09-tutorials"]="Tutoriais práticos e estudos de caso"
        ["10-legal-compliance"]="Documentação legal, compliance e auditoria"
        ["11-community"]="Recursos da comunidade e contribuições"
        ["12-appendices"]="Apêndices, referências e recursos adicionais"
    )
    
    # Criar README para cada seção
    for section in "${!SECTIONS[@]}"; do
        local readme_file="$PROJECT_ROOT/docs/$section/README.md"
        local section_name="${section#*-}"
        section_name="${section_name//-/ }"
        section_name="$(tr '[:lower:]' '[:upper:]' <<< ${section_name:0:1})${section_name:1}"
        
        if [[ ! -f "$readme_file" ]]; then
            cat > "$readme_file" << EOF
# $section_name

${SECTIONS[$section]}

## 📋 Conteúdo desta Seção

<!-- Esta seção será preenchida automaticamente -->

## 🚀 Links Rápidos

- [Início](../index.md)
- [Guia de Contribuição](../11-community/contributing.md)
- [API Reference](../08-api-reference/README.md)

## 📝 Sobre esta Documentação

Esta documentação faz parte do **Neural Crypto Bot 2.0**, um sistema de trading algorítmico de classe mundial desenvolvido com tecnologias de ponta.

---

> **Nota**: Esta página é gerada automaticamente. Para contribuir, consulte nosso [guia de contribuição](../11-community/contributing.md).
EOF
            log_debug "Criado README para seção: $section"
        fi
    done
}

create_index_files() {
    log_step "Criando Arquivos de Índice"
    
    # Criar .pages files para awesome-pages plugin
    local sections_with_pages=(
        "docs"
        "docs/01-getting-started"
        "docs/02-architecture"
        "docs/03-development"
        "docs/04-trading"
        "docs/05-machine-learning"
        "docs/06-integrations"
        "docs/07-operations"
        "docs/08-api-reference"
        "docs/09-tutorials"
        "docs/10-legal-compliance"
        "docs/11-community"
        "docs/12-appendices"
    )
    
    # Criar arquivo .pages principal
    cat > "$PROJECT_ROOT/docs/.pages" << 'EOF'
title: Neural Crypto Bot 2.0 Documentation
nav:
  - index.md
  - 01-getting-started
  - 02-architecture
  - 03-development
  - 04-trading
  - 05-machine-learning
  - 06-integrations
  - 07-operations
  - 08-api-reference
  - 09-tutorials
  - 10-legal-compliance
  - 11-community
  - 12-appendices
  - CHANGELOG.md
  - CONTRIBUTING.md
  - SECURITY.md
  - CODE_OF_CONDUCT.md
EOF
    
    # Criar arquivos .pages para subseções
    for section_dir in "${sections_with_pages[@]}"; do
        if [[ "$section_dir" != "docs" ]]; then
            local pages_file="$PROJECT_ROOT/$section_dir/.pages"
            local section_name=$(basename "$section_dir")
            
            cat > "$pages_file" << EOF
title: ${section_name^}
nav:
  - README.md
  - ...
collapse_single_pages: true
EOF
            log_debug "Criado .pages para: $section_name"
        fi
    done
    
    log_success "Arquivos de índice criados"
}

setup_directory_permissions() {
    log_step "Configurando Permissões de Diretório"
    
    # Diretórios que precisam de permissões específicas
    local writable_dirs=(
        ".docs-setup-logs"
        ".docs-cache"
        ".docs-build"
        "site"
    )
    
    for dir in "${writable_dirs[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        if [[ -d "$full_path" ]]; then
            chmod 755 "$full_path" 2>/dev/null || true
            log_debug "Permissões configuradas para: $dir"
        fi
    done
    
    # Configurar permissões para scripts
    local script_dirs=(
        "scripts/docs"
        "scripts/docs/generators"
        "scripts/docs/validators"
    )
    
    for dir in "${script_dirs[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        if [[ -d "$full_path" ]]; then
            find "$full_path" -name "*.sh" -type f -exec chmod +x {} \; 2>/dev/null || true
            log_debug "Permissões de execução configuradas para scripts em: $dir"
        fi
    done
    
    log_success "Permissões de diretório configuradas"
}

create_configuration_directories() {
    log_step "Criando Diretórios de Configuração"
    
    # Diretórios para diferentes tipos de configuração
    local config_types=(
        "mkdocs"
        "themes"
        "plugins"
        "overrides"
        "macros"
        "hooks"
    )
    
    for config_type in "${config_types[@]}"; do
        create_directory "$PROJECT_ROOT/docs/configs/$config_type" "config type"
    done
    
    # Diretórios para templates personalizados
    local template_types=(
        "page-templates"
        "macro-templates"
        "component-templates"
        "layout-templates"
    )
    
    for template_type in "${template_types[@]}"; do
        create_directory "$PROJECT_ROOT/docs/templates/$template_type" "template type"
    done
    
    # Diretórios para diferentes ambientes de build
    local build_envs=(
        "local"
        "development"
        "staging"
        "production"
    )
    
    for env in "${build_envs[@]}"; do
        create_directory "$PROJECT_ROOT/.docs-build/$env" "build environment"
        create_directory "$PROJECT_ROOT/.docs-cache/$env" "cache environment"
    done
    
    log_success "Diretórios de configuração criados"
}

validate_directory_structure() {
    log_step "Validando Estrutura de Diretórios"
    
    local validation_errors=()
    local total_dirs=0
    local existing_dirs=0
    
    # Verificar se todos os diretórios foram criados
    for dir in "${DIRECTORY_STRUCTURE[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        ((total_dirs++))
        
        if [[ -d "$full_path" ]]; then
            ((existing_dirs++))
            log_debug "✓ Diretório existe: $dir"
        else
            validation_errors+=("Diretório ausente: $dir")
            log_error "✗ Diretório ausente: $dir"
        fi
    done
    
    # Verificar permissões de escrita em diretórios críticos
    local critical_dirs=(
        "docs"
        "scripts/docs"
        ".docs-setup-logs"
    )
    
    for dir in "${critical_dirs[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        if [[ -d "$full_path" ]] && ! check_write_permissions "$full_path"; then
            validation_errors+=("Sem permissão de escrita: $dir")
        fi
    done
    
    # Relatório de validação
    local success_rate=$((existing_dirs * 100 / total_dirs))
    
    echo -e "\n${CYAN}${BOLD}📊 Relatório de Validação:${NC}"
    echo -e "${WHITE}├── Diretórios criados: $existing_dirs/$total_dirs ($success_rate%)${NC}"
    echo -e "${WHITE}├── Estrutura base: $(if [[ $success_rate -ge 95 ]]; then echo "✅ OK"; else echo "❌ Incompleta"; fi)${NC}"
    echo -e "${WHITE}└── Erros encontrados: ${#validation_errors[@]}${NC}"
    
    if [[ ${#validation_errors[@]} -gt 0 ]]; then
        log_warning "Problemas encontrados na estrutura:"
        for error in "${validation_errors[@]}"; do
            log_warning "  - $error"
        done
        return 1
    else
        log_success "Estrutura de diretórios validada com sucesso!"
        return 0
    fi
}

generate_directory_map() {
    log_step "Gerando Mapa de Diretórios"
    
    local map_file="$PROJECT_ROOT/.docs-directory-map.txt"
    
    cat > "$map_file" << EOF
# Neural Crypto Bot 2.0 - Directory Structure Map
# Generated on: $(date)
# Version: $DIRECTORY_STRUCTURE_VERSION

PROJECT_ROOT: $PROJECT_ROOT

EOF
    
    # Gerar árvore de diretórios
    if command_exists tree; then
        echo "## Directory Tree (via tree command)" >> "$map_file"
        tree -d -L 4 "$PROJECT_ROOT/docs" >> "$map_file" 2>/dev/null || true
        echo "" >> "$map_file"
    fi
    
    # Gerar lista detalhada
    echo "## Detailed Directory List" >> "$map_file"
    echo "" >> "$map_file"
    
    for dir in "${DIRECTORY_STRUCTURE[@]}"; do
        local full_path="$PROJECT_ROOT/$dir"
        local status="❌"
        local size_info=""
        
        if [[ -d "$full_path" ]]; then
            status="✅"
            if command_exists du; then
                size_info=" ($(du -sh "$full_path" 2>/dev/null | cut -f1))"
            fi
        fi
        
        echo "$status $dir$size_info" >> "$map_file"
    done
    
    echo "" >> "$map_file"
    echo "## Statistics" >> "$map_file"
    echo "- Total Directories: ${#DIRECTORY_STRUCTURE[@]}" >> "$map_file"
    echo "- Created: $(date)" >> "$map_file"
    echo "- System: $(detect_os) $(detect_arch)" >> "$map_file"
    
    log_success "Mapa de diretórios salvo em: $map_file"
}

# ================================
# FUNÇÃO PRINCIPAL
# ================================

main() {
    # Inicializar sistema
    init_docs_system
    
    log_step "Iniciando Criação da Estrutura de Diretórios"
    log_info "Versão da estrutura: $DIRECTORY_STRUCTURE_VERSION"
    log_info "Diretórios esperados: ${#DIRECTORY_STRUCTURE[@]}"
    
    local start_time=$(date +%s)
    
    # Executar criação
    create_base_directories
    create_special_directories
    create_placeholder_files
    create_index_files
    create_configuration_directories
    setup_directory_permissions
    
    # Validar resultado
    if validate_directory_structure; then
        generate_directory_map
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        # Salvar estatísticas
        save_config "DIRECTORY_CREATION_TIME" "$duration"
        save_config "DIRECTORY_STRUCTURE_VERSION" "$DIRECTORY_STRUCTURE_VERSION"
        save_config "TOTAL_DIRECTORIES" "${#DIRECTORY_STRUCTURE[@]}"
        
        log_success "Estrutura de diretórios criada com sucesso!"
        log_info "Tempo de execução: ${duration}s"
        log_info "Mapa de diretórios disponível em: .docs-directory-map.txt"
        
        return 0
    else
        log_error "Falha na criação da estrutura de diretórios"
        return 1
    fi
}

# ================================
# EXECUÇÃO
# ================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
        "