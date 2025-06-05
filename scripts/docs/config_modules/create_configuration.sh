#!/bin/bash
# scripts/docs/create_configuration.sh - Script principal de configuração refatorado
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Configuração modular enterprise-grade

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ================================
# CONFIGURAÇÕES ESPECÍFICAS
# ================================

readonly CONFIG_VERSION="2.0.0"
readonly MKDOCS_CONFIG_FILE="$PROJECT_ROOT/mkdocs.yml"
readonly BACKUP_SUFFIX=".backup.$(date +%Y%m%d%H%M%S)"

# Módulos de configuração
readonly CONFIG_MODULES=(
    "config_mkdocs_main"
    "config_environments" 
    "config_macros"
    "config_styles"
    "config_javascript"
    "config_validation"
)

# ================================
# CARREGAR MÓDULOS DE CONFIGURAÇÃO
# ================================

load_configuration_modules() {
    log_step "Carregando Módulos de Configuração"
    
    local modules_dir="$SCRIPT_DIR/config_modules"
    local loaded_modules=0
    local missing_modules=()
    
    # Verificar se diretório existe
    if [[ ! -d "$modules_dir" ]]; then
        log_error "Diretório de módulos não encontrado: $modules_dir"
        return 1
    fi
    
    # Carregar cada módulo
    for module in "${CONFIG_MODULES[@]}"; do
        local module_file="$modules_dir/${module}.sh"
        
        if [[ -f "$module_file" ]]; then
            log_debug "Carregando módulo: $module"
            source "$module_file"
            ((loaded_modules++))
        else
            missing_modules+=("$module")
            log_error "Módulo não encontrado: $module_file"
        fi
    done
    
    # Relatório de carregamento
    log_info "Módulos carregados: $loaded_modules/${#CONFIG_MODULES[@]}"
    
    if [[ ${#missing_modules[@]} -gt 0 ]]; then
        log_error "Módulos ausentes: ${missing_modules[*]}"
        return 1
    fi
    
    log_success "Todos os módulos de configuração carregados ✅"
    return 0
}

# ================================
# FUNÇÕES DE COORDENAÇÃO
# ================================

backup_existing_configs() {
    log_step "Fazendo Backup de Configurações Existentes"
    
    local config_files=(
        "$MKDOCS_CONFIG_FILE"
        "$PROJECT_ROOT/mkdocs.dev.yml"
        "$PROJECT_ROOT/mkdocs.staging.yml"
        "$PROJECT_ROOT/mkdocs.prod.yml"
    )
    
    local backed_up=0
    
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            if backup_file "$config_file" "$BACKUP_SUFFIX"; then
                ((backed_up++))
            fi
        fi
    done
    
    if [[ $backed_up -gt 0 ]]; then
        log_success "Backup criado para $backed_up arquivo(s) de configuração"
    else
        log_info "Nenhum arquivo de configuração existente para backup"
    fi
    
    return 0
}

execute_configuration_modules() {
    log_step "Executando Módulos de Configuração"
    
    local completed_modules=0
    local failed_modules=()
    local total_modules=${#CONFIG_MODULES[@]}
    
    # Executar cada módulo
    for module in "${CONFIG_MODULES[@]}"; do
        log_substep "Executando: $module"
        
        # Verificar se a função existe
        if declare -f "$module" >/dev/null; then
            if $module; then
                ((completed_modules++))
                log_success "✅ $module concluído"
            else
                failed_modules+=("$module")
                log_error "❌ $module falhou"
                
                # Perguntar se deve continuar
                if ! confirm_action "Continuar mesmo com falha em $module?" "n"; then
                    log_error "Configuração interrompida pelo usuário"
                    return 1
                fi
            fi
        else
            failed_modules+=("$module (função não encontrada)")
            log_error "Função não encontrada: $module"
        fi
        
        show_progress $((completed_modules + ${#failed_modules[@]})) $total_modules "Configurando"
    done
    
    # Relatório de execução
    echo -e "\n${CYAN}${BOLD}📊 Resultado da Configuração:${NC}"
    echo -e "${WHITE}├── Módulos executados: $completed_modules/$total_modules${NC}"
    echo -e "${WHITE}├── Módulos com falha: ${#failed_modules[@]}${NC}"
    echo -e "${WHITE}└── Taxa de sucesso: $((completed_modules * 100 / total_modules))%${NC}"
    
    if [[ ${#failed_modules[@]} -gt 0 ]]; then
        log_warning "Módulos com falha:"
        for failed in "${failed_modules[@]}"; do
            log_warning "  • $failed"
        done
        return 1
    fi
    
    log_success "Todos os módulos executados com sucesso ✅"
    return 0
}

validate_final_configuration() {
    log_step "Validando Configuração Final"
    
    local validation_errors=()
    
    # Verificar arquivo principal
    if [[ ! -f "$MKDOCS_CONFIG_FILE" ]]; then
        validation_errors+=("mkdocs.yml não encontrado")
    else
        # Verificar sintaxe YAML básica
        if ! python3 -c "import yaml; yaml.safe_load(open('$MKDOCS_CONFIG_FILE'))" 2>/dev/null; then
            validation_errors+=("mkdocs.yml tem sintaxe YAML inválida")
        fi
    fi
    
    # Verificar arquivos de ambiente
    local env_configs=("dev" "staging" "prod")
    for env in "${env_configs[@]}"; do
        local env_file="$PROJECT_ROOT/mkdocs.${env}.yml"
        if [[ ! -f "$env_file" ]]; then
            validation_errors+=("Configuração de ambiente ausente: mkdocs.${env}.yml")
        fi
    done
    
    # Verificar diretórios essenciais
    local essential_dirs=(
        "docs/stylesheets"
        "docs/javascripts"
        "docs/macros"
        "docs/data"
    )
    
    for dir in "${essential_dirs[@]}"; do
        if [[ ! -d "$PROJECT_ROOT/$dir" ]]; then
            validation_errors+=("Diretório essencial ausente: $dir")
        fi
    done
    
    # Verificar arquivos críticos
    local critical_files=(
        "docs/macros/__init__.py"
        "docs/data/config.yml"
        "docs/stylesheets/extra.css"
        "docs/javascripts/mathjax.js"
    )
    
    for file in "${critical_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            validation_errors+=("Arquivo crítico ausente: $file")
        fi
    done
    
    # Teste básico com MkDocs se disponível
    if command_exists poetry && poetry run which mkdocs >/dev/null 2>&1; then
        log_info "Testando build de validação..."
        if poetry run mkdocs build --strict --quiet --site-dir .validation-build 2>/dev/null; then
            log_success "Build de validação bem-sucedido"
            rm -rf "$PROJECT_ROOT/.validation-build" 2>/dev/null || true
        else
            validation_errors+=("Build de validação falhou")
        fi
    fi
    
    # Resultado da validação
    if [[ ${#validation_errors[@]} -eq 0 ]]; then
        log_success "Configuração validada com sucesso ✅"
        return 0
    else
        log_error "Problemas encontrados na validação:"
        for error in "${validation_errors[@]}"; do
            log_error "  • $error"
        done
        return 1
    fi
}

create_configuration_summary() {
    log_step "Gerando Resumo da Configuração"
    
    local summary_file="$PROJECT_ROOT/.docs-configuration-summary.md"
    
    cat > "$summary_file" << EOF
# Neural Crypto Bot 2.0 - Configuration Summary

Generated on: $(date)
Configuration Version: $CONFIG_VERSION

## Modules Executed

$(for module in "${CONFIG_MODULES[@]}"; do
    echo "- ✅ \`$module\` - $(get_module_description "$module")"
done)

## Files Created

### Main Configuration
- \`mkdocs.yml\` - Main MkDocs configuration
- \`mkdocs.dev.yml\` - Development environment
- \`mkdocs.staging.yml\` - Staging environment  
- \`mkdocs.prod.yml\` - Production environment

### Assets and Resources
- \`docs/stylesheets/\` - Custom CSS styles
- \`docs/javascripts/\` - JavaScript enhancements
- \`docs/macros/\` - Python macros and templates
- \`docs/data/\` - Configuration data files

## Key Features Configured

### Theme and Design
- Material Design theme with Neural Crypto Bot branding
- Dark/light mode toggle with custom colors
- Responsive navigation and layout
- Custom typography (Inter + JetBrains Mono)

### Advanced Features
- Mathematical notation support (MathJax)
- Code syntax highlighting with copy functionality
- Interactive feedback system
- Keyboard shortcuts
- Search with instant results

### Multi-Environment Support
- Development configuration for local testing
- Staging configuration for pre-production
- Production configuration optimized for performance

### Developer Experience
- Live reload for development
- Asset optimization and minification
- Git integration for revision tracking
- Quality validation tools

## Next Steps

1. **Test Configuration**: Run \`poetry run mkdocs serve\`
2. **Customize Theme**: Edit files in \`docs/stylesheets/\`
3. **Add Content**: Create documentation in appropriate sections
4. **Setup Assets**: Add logos and images to \`docs/assets/\`
5. **Deploy**: Use \`./scripts/docs/publish_docs.sh\`

## Configuration Details

- **Primary Color**: #3f51b5 (Indigo)
- **Accent Color**: #2196f3 (Blue)
- **Font Text**: Inter
- **Font Code**: JetBrains Mono
- **Features**: $(wc -l < "$MKDOCS_CONFIG_FILE") lines in main config

## Troubleshooting

If you encounter issues:
1. Check syntax: \`poetry run mkdocs build --strict\`
2. Validate files: \`./scripts/docs/validators/validate_markdown.sh\`
3. Review logs: Check \`.docs-setup-logs/\`
4. Restore backup: Files saved with \`$BACKUP_SUFFIX\` suffix

---

*Generated by Neural Crypto Bot 2.0 Documentation System*
EOF
    
    log_success "Resumo da configuração salvo: $(basename "$summary_file")"
}

get_module_description() {
    local module="$1"
    
    case "$module" in
        "config_mkdocs_main")
            echo "Main MkDocs configuration with theme and plugins"
            ;;
        "config_environments")
            echo "Multi-environment configurations (dev/staging/prod)"
            ;;
        "config_macros")
            echo "Python macros and data templates"
            ;;
        "config_styles")
            echo "Custom CSS styles and themes"
            ;;
        "config_javascript")
            echo "JavaScript enhancements and interactivity"
            ;;
        "config_validation")
            echo "Configuration validation and testing"
            ;;
        *)
            echo "Configuration module"
            ;;
    esac
}

# ================================
# FUNÇÃO PRINCIPAL
# ================================

main() {
    # Inicializar sistema
    init_docs_system
    
    log_step "Iniciando Configuração Modular do MkDocs"
    log_info "Versão da configuração: $CONFIG_VERSION"
    
    local start_time=$(date +%s)
    
    # Pipeline de configuração
    local config_steps=(
        "load_configuration_modules"
        "backup_existing_configs"
        "execute_configuration_modules"
        "validate_final_configuration"
        "create_configuration_summary"
    )
    
    local completed_steps=0
    local total_steps=${#config_steps[@]}
    
    # Executar pipeline
    for step in "${config_steps[@]}"; do
        log_substep "Executando: $step"
        
        if $step; then
            ((completed_steps++))
            show_progress $completed_steps $total_steps "Configuração em progresso"
        else
            log_error "Falha na etapa: $step"
            
            # Permitir continuar em alguns casos
            case "$step" in
                "backup_existing_configs")
                    log_warning "Falha no backup não é crítica, continuando..."
                    ((completed_steps++))
                    ;;
                "create_configuration_summary")
                    log_warning "Falha na criação do resumo não é crítica"
                    ((completed_steps++))
                    ;;
                *)
                    log_error "Falha crítica, interrompendo configuração"
                    return 1
                    ;;
            esac
        fi
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Salvar estatísticas
    save_config "CONFIGURATION_CREATION_TIME" "$duration"
    save_config "CONFIGURATION_VERSION" "$CONFIG_VERSION"
    save_config "CONFIGURATION_MODULES" "${CONFIG_MODULES[*]}"
    
    # Resultado final
    echo -e "\n${GREEN}${BOLD}🎉 Configuração Modular Criada com Sucesso!${NC}"
    echo -e "${GREEN}Tempo de execução: ${duration}s${NC}"
    echo -e "${GREEN}Módulos processados: ${#CONFIG_MODULES[@]}${NC}"
    
    echo -e "\n${CYAN}${BOLD}📋 Próximos Passos:${NC}"
    echo -e "1. ${BOLD}Testar configuração:${NC} poetry run mkdocs serve"
    echo -e "2. ${BOLD}Personalizar tema:${NC} Editar docs/stylesheets/extra.css"
    echo -e "3. ${BOLD}Adicionar conteúdo:${NC} ./scripts/docs/create_base_docs.sh"
    echo -e "4. ${BOLD}Configurar assets:${NC} ./scripts/docs/create_assets.sh"
    
    echo -e "\n${CYAN}${BOLD}🔧 Comandos Úteis:${NC}"
    echo -e "• ${BOLD}Desenvolvimento:${NC} poetry run mkdocs serve -f mkdocs.dev.yml"
    echo -e "• ${BOLD}Build produção:${NC} poetry run mkdocs build -f mkdocs.prod.yml"
    echo -e "• ${BOLD}Deploy:${NC} poetry run mkdocs gh-deploy"
    
    return 0
}

# ================================
# EXECUÇÃO
# ================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi