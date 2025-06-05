#!/bin/bash
# scripts/docs/publish_docs.sh - Sistema completo de publicação de documentação
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Pipeline de deploy enterprise-grade

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ================================
# CONFIGURAÇÕES DE PUBLICAÇÃO
# ================================

readonly PUBLISH_VERSION="2.0.0"
readonly BUILD_DIR="$PROJECT_ROOT/site"
readonly BACKUP_DIR="$PROJECT_ROOT/.docs-backups"
readonly DEPLOY_LOG="$PROJECT_ROOT/.docs-deploy.log"

# Ambientes de deploy
readonly ENVIRONMENTS=("development" "staging" "production")
readonly DEFAULT_ENVIRONMENT="production"

# URLs de deploy
declare -A DEPLOY_URLS=(
    ["development"]="http://localhost:8000"
    ["staging"]="https://staging-docs.neuralcryptobot.com"
    ["production"]="https://docs.neuralcryptobot.com"
)

# Branches Git para cada ambiente
declare -A GIT_BRANCHES=(
    ["development"]="develop"
    ["staging"]="staging"
    ["production"]="main"
)

# ================================
# FUNÇÕES DE VALIDAÇÃO PRÉ-DEPLOY
# ================================

validate_pre_deploy() {
    log_step "Validando Pré-Deploy"
    
    local issues=()
    
    # Verificar se estamos em um repositório Git
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        issues+=("Não está em um repositório Git")
    fi
    
    # Verificar se há mudanças não commitadas
    if git status --porcelain | grep -q .; then
        issues+=("Há mudanças não commitadas no repositório")
    fi
    
    # Verificar se Poetry está disponível
    if ! command_exists poetry; then
        issues+=("Poetry não encontrado")
    fi
    
    # Verificar se MkDocs está instalado
    if ! poetry run which mkdocs >/dev/null 2>&1; then
        issues+=("MkDocs não instalado no ambiente Poetry")
    fi
    
    # Verificar arquivos essenciais
    local essential_files=(
        "mkdocs.yml"
        "docs/index.md"
        "pyproject.toml"
    )
    
    for file in "${essential_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            issues+=("Arquivo essencial não encontrado: $file")
        fi
    done
    
    # Verificar configuração do ambiente
    local environment="${1:-$DEFAULT_ENVIRONMENT}"
    local config_file="$PROJECT_ROOT/mkdocs.${environment}.yml"
    
    if [[ ! -f "$config_file" && "$environment" != "production" ]]; then
        log_warning "Arquivo de configuração específico não encontrado: $config_file"
        log_info "Usando configuração padrão: mkdocs.yml"
    fi
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_error "Problemas encontrados na validação pré-deploy:"
        for issue in "${issues[@]}"; do
            log_error "  - $issue"
        done
        return 1
    fi
    
    log_success "Validação pré-deploy aprovada"
    return 0
}

run_quality_checks() {
    log_step "Executando Verificações de Qualidade"
    
    local checks_passed=0
    local total_checks=4
    
    # 1. Validação de Markdown
    log_substep "Validando Markdown..."
    if [[ -x "$SCRIPT_DIR/validators/validate_markdown.sh" ]]; then
        if "$SCRIPT_DIR/validators/validate_markdown.sh" >/dev/null 2>&1; then
            ((checks_passed++))
            log_success "✅ Validação de Markdown aprovada"
        else
            log_warning "⚠️ Validação de Markdown falhou"
        fi
    else
        log_info "📝 Validador de Markdown não encontrado (pulando)"
    fi
    
    # 2. Verificação de links
    log_substep "Verificando links..."
    if command_exists markdown-link-check; then
        local broken_links=0
        while IFS= read -r -d '' file; do
            if ! markdown-link-check "$file" >/dev/null 2>&1; then
                ((broken_links++))
            fi
        done < <(find "$PROJECT_ROOT/docs" -name "*.md" -print0 2>/dev/null)
        
        if [[ $broken_links -eq 0 ]]; then
            ((checks_passed++))
            log_success "✅ Verificação de links aprovada"
        else
            log_warning "⚠️ $broken_links arquivo(s) com links quebrados"
        fi
    else
        log_info "🔗 markdown-link-check não encontrado (pulando)"
    fi
    
    # 3. Teste de build
    log_substep "Testando build..."
    if build_documentation "test"; then
        ((checks_passed++))
        log_success "✅ Build de teste aprovado"
        # Limpar build de teste
        rm -rf "$PROJECT_ROOT/.test-build" 2>/dev/null || true
    else
        log_warning "⚠️ Build de teste falhou"
    fi
    
    # 4. Verificação de arquivos obrigatórios
    log_substep "Verificando estrutura..."
    local required_files=(
        "docs/index.md"
        "docs/CHANGELOG.md"
        "docs/CONTRIBUTING.md"
        "docs/SECURITY.md"
    )
    
    local missing_files=0
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            ((missing_files++))
        fi
    done
    
    if [[ $missing_files -eq 0 ]]; then
        ((checks_passed++))
        log_success "✅ Estrutura de arquivos aprovada"
    else
        log_warning "⚠️ $missing_files arquivo(s) obrigatório(s) ausente(s)"
    fi
    
    # Resultado das verificações
    local success_rate=$((checks_passed * 100 / total_checks))
    
    echo -e "\n${CYAN}${BOLD}📊 Resultado das Verificações de Qualidade:${NC}"
    echo -e "${WHITE}├── Verificações aprovadas: $checks_passed/$total_checks${NC}"
    echo -e "${WHITE}├── Taxa de sucesso: $success_rate%${NC}"
    echo -e "${WHITE}└── Status: $(if [[ $success_rate -ge 75 ]]; then echo "✅ Aprovado"; else echo "❌ Reprovado"; fi)${NC}"
    
    if [[ $success_rate -ge 75 ]]; then
        log_success "Verificações de qualidade aprovadas"
        return 0
    else
        log_error "Verificações de qualidade falharam"
        return 1
    fi
}

build_documentation() {
    local environment="${1:-production}"
    local output_dir="${2:-$BUILD_DIR}"
    
    log_step "Building Documentation para $environment"
    
    # Determinar arquivo de configuração
    local config_file="mkdocs.yml"
    if [[ -f "$PROJECT_ROOT/mkdocs.${environment}.yml" ]]; then
        config_file="mkdocs.${environment}.yml"
        log_info "Usando configuração específica: $config_file"
    else
        log_info "Usando configuração padrão: $config_file"
    fi
    
    # Criar diretório de saída
    if [[ "$environment" == "test" ]]; then
        output_dir="$PROJECT_ROOT/.test-build"
    fi
    
    create_directory "$(dirname "$output_dir")" "build parent directory"
    
    # Build com MkDocs
    log_info "Executando build do MkDocs..."
    
    local build_start=$(date +%s)
    
    if poetry run mkdocs build \
        --config-file "$config_file" \
        --site-dir "$output_dir" \
        --strict \
        --verbose 2>&1 | tee -a "$DEPLOY_LOG"; then
        
        local build_end=$(date +%s)
        local build_duration=$((build_end - build_start))
        
        log_success "Build concluído em ${build_duration}s"
        
        # Verificar se o build gerou arquivos
        if [[ -d "$output_dir" && -f "$output_dir/index.html" ]]; then
            local file_count
            file_count=$(find "$output_dir" -type f | wc -l)
            log_info "Build gerou $file_count arquivos"
            
            # Calcular tamanho total
            local total_size
            total_size=$(du -sh "$output_dir" 2>/dev/null | cut -f1)
            log_info "Tamanho total do build: $total_size"
            
            return 0
        else
            log_error "Build não gerou arquivos válidos"
            return 1
        fi
    else
        log_error "Falha no build do MkDocs"
        return 1
    fi
}

optimize_build() {
    log_step "Otimizando Build"
    
    if [[ ! -d "$BUILD_DIR" ]]; then
        log_error "Diretório de build não encontrado: $BUILD_DIR"
        return 1
    fi
    
    local original_size
    original_size=$(du -sb "$BUILD_DIR" | cut -f1)
    
    # Minificar HTML se htmlmin estiver disponível
    if command_exists htmlmin; then
        log_info "Minificando arquivos HTML..."
        find "$BUILD_DIR" -name "*.html" -exec htmlmin {} {} \; 2>/dev/null || true
    fi
    
    # Otimizar imagens se imagemin estiver disponível
    if command_exists imagemin; then
        log_info "Otimizando imagens..."
        find "$BUILD_DIR" -name "*.png" -o -name "*.jpg" -o -name "*.jpeg" | \
        xargs -r imagemin --out-dir="$BUILD_DIR" 2>/dev/null || true
    fi
    
    # Comprimir arquivos CSS e JS se disponível
    if command_exists uglifyjs; then
        log_info "Minificando JavaScript..."
        find "$BUILD_DIR" -name "*.js" ! -name "*.min.js" | while read -r file; do
            uglifyjs "$file" -o "$file" 2>/dev/null || true
        done
    fi
    
    # Calcular economia
    local optimized_size
    optimized_size=$(du -sb "$BUILD_DIR" | cut -f1)
    local saved_bytes=$((original_size - optimized_size))
    local saved_percent=$((saved_bytes * 100 / original_size))
    
    if [[ $saved_bytes -gt 0 ]]; then
        log_success "Otimização concluída: $(numfmt --to=iec $saved_bytes) economizados ($saved_percent%)"
    else
        log_info "Nenhuma otimização adicional aplicada"
    fi
    
    return 0
}

create_backup() {
    local environment="$1"
    
    log_step "Criando Backup"
    
    create_directory "$BACKUP_DIR" "backup directory"
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_name="${environment}_${timestamp}"
    local backup_path="$BACKUP_DIR/$backup_name"
    
    # Backup do build atual se existir
    if [[ -d "$BUILD_DIR" ]]; then
        log_info "Fazendo backup do build atual..."
        cp -r "$BUILD_DIR" "$backup_path"
        
        # Comprimir backup
        if command_exists tar; then
            log_info "Comprimindo backup..."
            cd "$BACKUP_DIR"
            tar -czf "${backup_name}.tar.gz" "$backup_name"
            rm -rf "$backup_name"
            backup_path="${backup_path}.tar.gz"
        fi
        
        log_success "Backup criado: $(basename "$backup_path")"
        save_config "LAST_BACKUP" "$backup_path"
    else
        log_info "Nenhum build anterior encontrado para backup"
    fi
    
    # Limpar backups antigos (manter apenas os 5 mais recentes)
    local backup_count
    backup_count=$(find "$BACKUP_DIR" -name "${environment}_*.tar.gz" | wc -l)
    
    if [[ $backup_count -gt 5 ]]; then
        log_info "Removendo backups antigos..."
        find "$BACKUP_DIR" -name "${environment}_*.tar.gz" | \
        sort | head -n $((backup_count - 5)) | \
        xargs rm -f
    fi
    
    return 0
}

deploy_to_github_pages() {
    local environment="$1"
    local branch="${GIT_BRANCHES[$environment]}"
    
    log_step "Deploy para GitHub Pages ($environment)"
    
    # Verificar se o remote origin existe
    if ! git remote get-url origin >/dev/null 2>&1; then
        log_error "Remote 'origin' não configurado"
        return 1
    fi
    
    # Verificar se estamos na branch correta
    local current_branch
    current_branch=$(git branch --show-current)
    
    if [[ "$current_branch" != "$branch" ]]; then
        log_warning "Branch atual ($current_branch) diferente da esperada ($branch)"
        
        if confirm_action "Fazer checkout para $branch?" "n"; then
            git checkout "$branch" || {
                log_error "Falha ao fazer checkout para $branch"
                return 1
            }
        else
            log_info "Continuando na branch atual"
        fi
    fi
    
    # Deploy usando mike para versionamento
    if command_exists mike; then
        log_info "Usando mike para deploy versionado..."
        
        local version
        version=$(load_config "PROJECT_VERSION" "2.0.0")
        
        # Deploy com mike
        if poetry run mike deploy \
            --config-file "mkdocs.${environment}.yml" \
            --push \
            --update-aliases \
            "$version" latest; then
            
            log_success "Deploy versionado concluído"
            
            # Definir versão padrão
            poetry run mike set-default --push latest
            
        else
            log_error "Falha no deploy versionado"
            return 1
        fi
    else
        # Deploy simples com gh-deploy
        log_info "Usando gh-deploy para deploy simples..."
        
        local config_file="mkdocs.yml"
        if [[ -f "$PROJECT_ROOT/mkdocs.${environment}.yml" ]]; then
            config_file="mkdocs.${environment}.yml"
        fi
        
        if poetry run mkdocs gh-deploy \
            --config-file "$config_file" \
            --force; then
            
            log_success "Deploy simples concluído"
        else
            log_error "Falha no deploy simples"
            return 1
        fi
    fi
    
    # Obter URL de deploy
    local deploy_url="${DEPLOY_URLS[$environment]}"
    log_success "Documentação disponível em: $deploy_url"
    
    return 0
}

deploy_to_custom_server() {
    local environment="$1"
    local server_config="${2:-}"
    
    log_step "Deploy para Servidor Customizado ($environment)"
    
    if [[ -z "$server_config" ]]; then
        log_error "Configuração do servidor não fornecida"
        return 1
    fi
    
    # Esta função pode ser expandida para diferentes tipos de deploy
    # como FTP, SCP, rsync, etc.
    
    log_info "Deploy para servidor customizado ainda não implementado"
    log_info "Use GitHub Pages ou configure manualmente"
    
    return 1
}

verify_deployment() {
    local environment="$1"
    local deploy_url="${DEPLOY_URLS[$environment]}"
    
    log_step "Verificando Deploy"
    
    # Aguardar um momento para propagação
    log_info "Aguardando propagação do deploy..."
    sleep 10
    
    # Verificar se o site está acessível
    if command_exists curl; then
        local response_code
        response_code=$(curl -s -o /dev/null -w "%{http_code}" "$deploy_url" || echo "000")
        
        case "$response_code" in
            200)
                log_success "✅ Site acessível: $deploy_url (HTTP $response_code)"
                ;;
            404)
                log_warning "⚠️ Site não encontrado: $deploy_url (HTTP $response_code)"
                log_info "Pode levar alguns minutos para o deploy ser propagado"
                ;;
            *)
                log_warning "⚠️ Resposta inesperada: $deploy_url (HTTP $response_code)"
                ;;
        esac
    else
        log_info "curl não disponível, verificação manual necessária"
        log_info "Verifique: $deploy_url"
    fi
    
    # Verificar conteúdo específico se possível
    if command_exists curl && [[ "$response_code" == "200" ]]; then
        local page_content
        page_content=$(curl -s "$deploy_url" || echo "")
        
        if echo "$page_content" | grep -q "Neural Crypto Bot 2.0"; then
            log_success "✅ Conteúdo válido detectado"
        else
            log_warning "⚠️ Conteúdo esperado não encontrado"
        fi
    fi
    
    return 0
}

generate_deploy_report() {
    local environment="$1"
    local start_time="$2"
    local end_time="$3"
    
    log_step "Gerando Relatório de Deploy"
    
    local duration=$((end_time - start_time))
    local deploy_report="$PROJECT_ROOT/.docs-deploy-report-${environment}.md"
    
    cat > "$deploy_report" << EOF
# Deployment Report - $environment

Generated on: $(date)
Environment: $environment
Duration: ${duration}s

## Summary

- **Start Time**: $(date -d @$start_time)
- **End Time**: $(date -d @$end_time)
- **Duration**: ${duration} seconds
- **Status**: $(if [[ $? -eq 0 ]]; then echo "✅ Success"; else echo "❌ Failed"; fi)
- **Deploy URL**: ${DEPLOY_URLS[$environment]}

## Build Information

- **Build Directory**: $BUILD_DIR
- **Configuration**: mkdocs.${environment}.yml
- **Git Branch**: ${GIT_BRANCHES[$environment]}
- **Git Commit**: $(git rev-parse HEAD 2>/dev/null || echo "N/A")

## Files Generated

\`\`\`
$(if [[ -d "$BUILD_DIR" ]]; then find "$BUILD_DIR" -type f | wc -l; else echo "0"; fi) files
$(if [[ -d "$BUILD_DIR" ]]; then du -sh "$BUILD_DIR" | cut -f1; else echo "0B"; fi) total size
\`\`\`

## Quality Checks

- Markdown validation: $(if [[ -x "$SCRIPT_DIR/validators/validate_markdown.sh" ]] && "$SCRIPT_DIR/validators/validate_markdown.sh" >/dev/null 2>&1; then echo "✅ Passed"; else echo "❌ Failed"; fi)
- Link checking: $(if command_exists markdown-link-check; then echo "✅ Available"; else echo "⚠️ Not available"; fi)
- Build test: $(if [[ -d "$BUILD_DIR" && -f "$BUILD_DIR/index.html" ]]; then echo "✅ Passed"; else echo "❌ Failed"; fi)

## Deploy Method

$(if command_exists mike; then echo "📦 Mike (versioned)"; else echo "🚀 MkDocs gh-deploy"; fi)

## Post-Deploy Verification

- Site accessibility: $(if command_exists curl; then 
    response=$(curl -s -o /dev/null -w "%{http_code}" "${DEPLOY_URLS[$environment]}" 2>/dev/null || echo "000")
    case "$response" in
        200) echo "✅ Accessible (HTTP $response)" ;;
        *) echo "⚠️ HTTP $response" ;;
    esac
else echo "⚠️ Cannot verify (curl not available)"; fi)

## Backup Information

$(if [[ -f "$(load_config "LAST_BACKUP" "")" ]]; then 
    echo "- Backup created: $(basename "$(load_config "LAST_BACKUP")")"
else 
    echo "- No backup created"
fi)

## Commands Used

\`\`\`bash
# Build command
poetry run mkdocs build --config-file mkdocs.${environment}.yml --site-dir $BUILD_DIR

# Deploy command
$(if command_exists mike; then 
    echo "poetry run mike deploy --config-file mkdocs.${environment}.yml --push --update-aliases latest"
else 
    echo "poetry run mkdocs gh-deploy --config-file mkdocs.${environment}.yml"
fi)
\`\`\`

## Next Steps

1. Verify deployment at: ${DEPLOY_URLS[$environment]}
2. Test all major sections and navigation
3. Check mobile responsiveness
4. Validate search functionality
5. Monitor for any issues

---

*Report generated by Neural Crypto Bot 2.0 Documentation System v$PUBLISH_VERSION*
EOF
    
    log_success "Relatório de deploy salvo: $(basename "$deploy_report")"
    save_config "LAST_DEPLOY_REPORT" "$deploy_report"
}

# ================================
# FUNÇÃO PRINCIPAL
# ================================

main() {
    # Inicializar sistema
    init_docs_system
    
    display_main_banner
    log_step "Iniciando Publicação da Documentação"
    log_info "Versão do sistema: $PUBLISH_VERSION"
    
    # Verificar argumentos
    local environment="${1:-}"
    local force_deploy="${2:-false}"
    
    # Selecionar ambiente se não fornecido
    if [[ -z "$environment" ]]; then
        echo -e "\n${CYAN}${BOLD}🌍 Selecione o ambiente de deploy:${NC}\n"
        echo -e "${WHITE}1. ${BOLD}development${NC} - Deploy local para testes"
        echo -e "${WHITE}2. ${BOLD}staging${NC} - Ambiente de homologação"
        echo -e "${WHITE}3. ${BOLD}production${NC} - Ambiente de produção"
        echo ""
        
        while true; do
            echo -ne "${CYAN}Escolha o ambiente [1-3]: ${NC}"
            read -r env_choice
            
            case $env_choice in
                1) environment="development"; break ;;
                2) environment="staging"; break ;;
                3) environment="production"; break ;;
                *) echo -e "${RED}Opção inválida. Escolha entre 1-3.${NC}" ;;
            esac
        done
    fi
    
    # Validar ambiente
    if [[ ! " ${ENVIRONMENTS[*]} " =~ " $environment " ]]; then
        log_error "Ambiente inválido: $environment"
        log_info "Ambientes válidos: ${ENVIRONMENTS[*]}"
        exit 1
    fi
    
    log_info "Ambiente selecionado: $environment"
    log_info "URL de deploy: ${DEPLOY_URLS[$environment]}"
    
    # Confirmar deploy se não forçado
    if [[ "$force_deploy" != "true" ]]; then
        echo ""
        if ! confirm_action "Continuar com o deploy para $environment?" "y"; then
            log_info "Deploy cancelado pelo usuário"
            exit 0
        fi
    fi
    
    local deploy_start=$(date +%s)
    
    # Pipeline de deploy
    local pipeline_steps=(
        "validate_pre_deploy $environment"
        "run_quality_checks"
        "create_backup $environment"
        "build_documentation $environment"
        "optimize_build"
    )
    
    # Adicionar step de deploy baseado no ambiente
    if [[ "$environment" == "development" ]]; then
        pipeline_steps+=("echo 'Deploy de desenvolvimento - servir localmente'")
    else
        pipeline_steps+=("deploy_to_github_pages $environment")
        pipeline_steps+=("verify_deployment $environment")
    fi
    
    local completed_steps=0
    local total_steps=${#pipeline_steps[@]}
    
    # Executar pipeline
    for step in "${pipeline_steps[@]}"; do
        log_substep "Executando: $step"
        
        if eval "$step"; then
            ((completed_steps++))
            show_progress $completed_steps $total_steps "Deploy em progresso"
        else
            log_error "Falha na etapa: $step"
            
            if confirm_action "Continuar mesmo com falha?" "n"; then
                ((completed_steps++))
            else
                log_error "Deploy interrompido"
                exit 1
            fi
        fi
    done
    
    local deploy_end=$(date +%s)
    
    # Gerar relatório
    generate_deploy_report "$environment" "$deploy_start" "$deploy_end"
    
    # Resultado final
    local duration=$((deploy_end - deploy_start))
    
    echo -e "\n${GREEN}${BOLD}🎉 DEPLOY CONCLUÍDO COM SUCESSO!${NC}"
    echo -e "${GREEN}Ambiente: $environment${NC}"
    echo -e "${GREEN}Duração: ${duration}s${NC}"
    echo -e "${GREEN}URL: ${DEPLOY_URLS[$environment]}${NC}"
    
    if [[ "$environment" == "development" ]]; then
        echo -e "\n${CYAN}${BOLD}📋 Para servir localmente:${NC}"
        echo -e "${WHITE}poetry run mkdocs serve -f mkdocs.${environment}.yml${NC}"
        echo -e "${WHITE}Acesse: http://localhost:8000${NC}"
    else
        echo -e "\n${CYAN}${BOLD}📋 Próximos Passos:${NC}"
        echo -e "${WHITE}1. Verificar site: ${DEPLOY_URLS[$environment]}${NC}"
        echo -e "${WHITE}2. Testar navegação e funcionalidades${NC}"
        echo -e "${WHITE}3. Verificar responsividade mobile${NC}"
        echo -e "${WHITE}4. Validar busca e links${NC}"
    fi
    
    # Salvar informações do deploy
    save_config "LAST_DEPLOY_ENVIRONMENT" "$environment"
    save_config "LAST_DEPLOY_TIME" "$deploy_end"
    save_config "LAST_DEPLOY_DURATION" "$duration"
    save_config "LAST_DEPLOY_URL" "${DEPLOY_URLS[$environment]}"
    
    return 0
}

# ================================
# EXECUÇÃO
# ================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi