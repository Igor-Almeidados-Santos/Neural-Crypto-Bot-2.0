#!/bin/bash
# scripts/docs/setup_documentation.sh - Script principal de configuração
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Sistema completo de documentação enterprise-grade

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ================================
# CONFIGURAÇÕES PRINCIPAIS
# ================================

readonly SETUP_VERSION="2.0.0"
readonly SETUP_NAME="Neural Crypto Bot 2.0 Documentation Setup"

# Scripts de configuração em ordem de execução
readonly SETUP_SCRIPTS=(
    "check_prerequisites.sh"
    "create_directories.sh"
    "install_dependencies.sh"
    "create_configuration.sh"
    "create_base_docs.sh"
    "setup_automation.sh"
    "create_assets.sh"
    "setup_github.sh"
    "create_templates.sh"
    "generate_content.sh"
    "setup_quality.sh"
    "validate.sh"
)

# ================================
# FUNÇÕES DE INTERFACE
# ================================

show_welcome_banner() {
    clear
    display_main_banner
    
    echo -e "${CYAN}${BOLD}Bem-vindo ao Sistema de Configuração da Documentação!${NC}\n"
    
    echo -e "${WHITE}Este assistente irá configurar um sistema de documentação de classe mundial para o${NC}"
    echo -e "${WHITE}Neural Crypto Bot 2.0, incluindo:${NC}\n"
    
    echo -e "${GREEN}📁 Estrutura completa de diretórios organizados${NC}"
    echo -e "${GREEN}🎨 Tema customizado com Material Design${NC}"
    echo -e "${GREEN}🔧 Configurações para desenvolvimento, staging e produção${NC}"
    echo -e "${GREEN}📝 Templates de documentação padronizados${NC}"
    echo -e "${GREEN}🤖 Automação de build e deploy${NC}"
    echo -e "${GREEN}✅ Validação de qualidade automática${NC}"
    echo -e "${GREEN}🔗 Integração com GitHub Pages${NC}"
    echo -e "${GREEN}📊 Analytics e feedback de usuários${NC}\n"
    
    echo -e "${YELLOW}${BOLD}⏱️  Tempo estimado: 5-10 minutos${NC}"
    echo -e "${YELLOW}${BOLD}📦 Espaço necessário: ~500MB${NC}"
    echo -e "${YELLOW}${BOLD}🌐 Conexão: Necessária para downloads${NC}\n"
}

show_setup_options() {
    echo -e "${CYAN}${BOLD}Opções de Configuração:${NC}\n"
    
    echo -e "${WHITE}1. ${BOLD}Setup Completo${NC} - Configuração completa automática (recomendado)"
    echo -e "${DIM}   Executa todos os scripts de configuração em sequência${NC}\n"
    
    echo -e "${WHITE}2. ${BOLD}Setup Interativo${NC} - Escolher quais componentes instalar"
    echo -e "${DIM}   Permite selecionar scripts específicos para executar${NC}\n"
    
    echo -e "${WHITE}3. ${BOLD}Setup Personalizado${NC} - Configuração avançada"
    echo -e "${DIM}   Para usuários experientes que querem controle total${NC}\n"
    
    echo -e "${WHITE}4. ${BOLD}Verificar Pré-requisitos${NC} - Apenas verificar sistema"
    echo -e "${DIM}   Testar se o sistema atende aos requisitos${NC}\n"
    
    echo -e "${WHITE}5. ${BOLD}Mostrar Ajuda${NC} - Informações detalhadas"
    echo -e "${DIM}   Documentação completa do sistema de setup${NC}\n"
    
    echo -e "${WHITE}6. ${BOLD}Sair${NC}"
    echo -e "${DIM}   Cancelar configuração${NC}\n"
}

get_user_choice() {
    while true; do
        echo -ne "${CYAN}Escolha uma opção [1-6]: ${NC}"
        read -r choice
        
        case $choice in
            1|2|3|4|5|6)
                echo "$choice"
                return 0
                ;;
            *)
                echo -e "${RED}Opção inválida. Por favor, escolha entre 1-6.${NC}"
                ;;
        esac
    done
}

confirm_action() {
    local message="$1"
    local default="${2:-n}"
    
    while true; do
        if [[ "$default" == "y" ]]; then
            echo -ne "${YELLOW}$message [Y/n]: ${NC}"
        else
            echo -ne "${YELLOW}$message [y/N]: ${NC}"
        fi
        
        read -r response
        response="${response:-$default}"
        
        case "$response" in
            [Yy]|[Yy][Ee][Ss])
                return 0
                ;;
            [Nn]|[Nn][Oo])
                return 1
                ;;
            *)
                echo -e "${RED}Por favor, responda sim (y) ou não (n).${NC}"
                ;;
        esac
    done
}

# ================================
# FUNÇÕES DE EXECUÇÃO
# ================================

run_complete_setup() {
    log_step "Executando Setup Completo"
    
    local total_scripts=${#SETUP_SCRIPTS[@]}
    local completed_scripts=0
    local failed_scripts=()
    local start_time=$(date +%s)
    
    echo -e "\n${CYAN}${BOLD}📋 Scripts a serem executados:${NC}"
    for script in "${SETUP_SCRIPTS[@]}"; do
        echo -e "${WHITE}  • $script${NC}"
    done
    echo ""
    
    if ! confirm_action "Continuar com o setup completo?" "y"; then
        log_info "Setup cancelado pelo usuário"
        return 1
    fi
    
    # Executar cada script
    for script in "${SETUP_SCRIPTS[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        
        log_substep "Executando: $script"
        
        if [[ -f "$script_path" && -x "$script_path" ]]; then
            if "$script_path"; then
                ((completed_scripts++))
                log_success "✅ $script concluído"
            else
                failed_scripts+=("$script")
                log_error "❌ $script falhou"
                
                # Perguntar se deve continuar
                if ! confirm_action "Continuar mesmo com falha em $script?" "n"; then
                    log_error "Setup interrompido pelo usuário"
                    return 1
                fi
            fi
        else
            log_warning "Script não encontrado ou não executável: $script"
            failed_scripts+=("$script (não encontrado)")
        fi
        
        show_progress $((completed_scripts + ${#failed_scripts[@]})) $total_scripts "Setup em progresso"
    done
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Relatório final
    echo -e "\n${CYAN}${BOLD}📊 Relatório do Setup Completo:${NC}"
    echo -e "${WHITE}├── Scripts executados: $completed_scripts/$total_scripts${NC}"
    echo -e "${WHITE}├── Scripts com falha: ${#failed_scripts[@]}${NC}"
    echo -e "${WHITE}├── Tempo total: ${duration}s${NC}"
    echo -e "${WHITE}└── Taxa de sucesso: $((completed_scripts * 100 / total_scripts))%${NC}"
    
    if [[ ${#failed_scripts[@]} -eq 0 ]]; then
        echo -e "\n${GREEN}${BOLD}🎉 SETUP COMPLETO COM SUCESSO!${NC}"
        show_completion_summary
        return 0
    else
        echo -e "\n${YELLOW}${BOLD}⚠️ SETUP CONCLUÍDO COM AVISOS${NC}"
        echo -e "${YELLOW}Scripts com falha:${NC}"
        for failed in "${failed_scripts[@]}"; do
            echo -e "${RED}  • $failed${NC}"
        done
        
        echo -e "\n${CYAN}Você pode tentar executar os scripts com falha individualmente:${NC}"
        for failed in "${failed_scripts[@]}"; do
            local script_name="${failed%% *}"
            echo -e "${WHITE}  ./scripts/docs/$script_name${NC}"
        done
        
        return 1
    fi
}

run_interactive_setup() {
    log_step "Executando Setup Interativo"
    
    echo -e "\n${CYAN}${BOLD}📋 Scripts Disponíveis:${NC}\n"
    
    local script_choices=()
    local i=1
    
    for script in "${SETUP_SCRIPTS[@]}"; do
        local description=""
        case "$script" in
            "check_prerequisites.sh")
                description="Verificar pré-requisitos do sistema"
                ;;
            "create_directories.sh")
                description="Criar estrutura de diretórios"
                ;;
            "install_dependencies.sh")
                description="Instalar dependências Python e Node.js"
                ;;
            "create_configuration.sh")
                description="Criar configurações do MkDocs"
                ;;
            "create_base_docs.sh")
                description="Criar documentos base"
                ;;
            "setup_automation.sh")
                description="Configurar scripts de automação"
                ;;
            "create_assets.sh")
                description="Criar assets e recursos visuais"
                ;;
            "setup_github.sh")
                description="Configurar integração GitHub"
                ;;
            "create_templates.sh")
                description="Criar templates de documentação"
                ;;
            "generate_content.sh")
                description="Gerar conteúdo de exemplo"
                ;;
            "setup_quality.sh")
                description="Configurar ferramentas de qualidade"
                ;;
            "validate.sh")
                description="Validar configuração final"
                ;;
        esac
        
        echo -e "${WHITE}$i. ${BOLD}$script${NC}"
        echo -e "${DIM}   $description${NC}"
        echo ""
        
        script_choices+=("$script")
        ((i++))
    done
    
    echo -e "${WHITE}$i. ${BOLD}Executar todos selecionados${NC}"
    echo -e "${WHITE}$((i+1)). ${BOLD}Voltar ao menu principal${NC}\n"
    
    local selected_scripts=()
    
    while true; do
        echo -ne "${CYAN}Escolha scripts para executar (números separados por espaço, ou 'a' para todos): ${NC}"
        read -r selections
        
        if [[ "$selections" == "a" || "$selections" == "all" ]]; then
            selected_scripts=("${SETUP_SCRIPTS[@]}")
            break
        elif [[ "$selections" == "$i" ]]; then
            break
        elif [[ "$selections" == "$((i+1))" ]]; then
            return 0
        else
            # Processar seleções numéricas
            selected_scripts=()
            for selection in $selections; do
                if [[ "$selection" =~ ^[0-9]+$ ]] && [[ $selection -ge 1 && $selection -le ${#SETUP_SCRIPTS[@]} ]]; then
                    selected_scripts+=("${script_choices[$((selection-1))]}")
                else
                    echo -e "${RED}Seleção inválida: $selection${NC}"
                    continue 2
                fi
            done
            break
        fi
    done
    
    if [[ ${#selected_scripts[@]} -eq 0 ]]; then
        log_info "Nenhum script selecionado"
        return 0
    fi
    
    echo -e "\n${CYAN}Scripts selecionados:${NC}"
    for script in "${selected_scripts[@]}"; do
        echo -e "${WHITE}  • $script${NC}"
    done
    
    if ! confirm_action "Executar scripts selecionados?" "y"; then
        return 0
    fi
    
    # Executar scripts selecionados
    local completed=0
    local failed=0
    
    for script in "${selected_scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        
        log_substep "Executando: $script"
        
        if [[ -f "$script_path" && -x "$script_path" ]]; then
            if "$script_path"; then
                ((completed++))
                log_success "✅ $script concluído"
            else
                ((failed++))
                log_error "❌ $script falhou"
            fi
        else
            ((failed++))
            log_error "Script não encontrado: $script"
        fi
        
        show_progress $((completed + failed)) ${#selected_scripts[@]} "Executando scripts"
    done
    
    echo -e "\n${CYAN}${BOLD}📊 Resultado do Setup Interativo:${NC}"
    echo -e "${WHITE}├── Scripts executados com sucesso: $completed${NC}"
    echo -e "${WHITE}├── Scripts com falha: $failed${NC}"
    echo -e "${WHITE}└── Total processado: ${#selected_scripts[@]}${NC}"
    
    if [[ $failed -eq 0 ]]; then
        echo -e "\n${GREEN}${BOLD}✅ Todos os scripts selecionados foram executados com sucesso!${NC}"
    else
        echo -e "\n${YELLOW}${BOLD}⚠️ Alguns scripts falharam. Verifique os logs para detalhes.${NC}"
    fi
    
    return 0
}

run_custom_setup() {
    log_step "Executando Setup Personalizado"
    
    echo -e "\n${CYAN}${BOLD}🔧 Configuração Avançada${NC}\n"
    
    echo -e "${WHITE}Opções disponíveis:${NC}\n"
    echo -e "${WHITE}1. ${BOLD}Configurar ambiente de desenvolvimento${NC}"
    echo -e "${WHITE}2. ${BOLD}Configurar ambiente de produção${NC}"
    echo -e "${WHITE}3. ${BOLD}Configurar apenas documentação${NC}"
    echo -e "${WHITE}4. ${BOLD}Configurar apenas automação${NC}"
    echo -e "${WHITE}5. ${BOLD}Restaurar configuração de backup${NC}"
    echo -e "${WHITE}6. ${BOLD}Configuração mínima (apenas essencial)${NC}"
    echo -e "${WHITE}7. ${BOLD}Voltar ao menu principal${NC}\n"
    
    local custom_choice
    while true; do
        echo -ne "${CYAN}Escolha uma opção [1-7]: ${NC}"
        read -r custom_choice
        
        case $custom_choice in
            1)
                run_development_setup
                break
                ;;
            2)
                run_production_setup
                break
                ;;
            3)
                run_docs_only_setup
                break
                ;;
            4)
                run_automation_only_setup
                break
                ;;
            5)
                run_restore_setup
                break
                ;;
            6)
                run_minimal_setup
                break
                ;;
            7)
                return 0
                ;;
            *)
                echo -e "${RED}Opção inválida. Escolha entre 1-7.${NC}"
                ;;
        esac
    done
}

run_development_setup() {
    log_info "Configurando ambiente de desenvolvimento..."
    
    local dev_scripts=(
        "check_prerequisites.sh"
        "create_directories.sh"
        "install_dependencies.sh"
        "create_configuration.sh"
        "setup_automation.sh"
    )
    
    execute_script_list "${dev_scripts[@]}"
}

run_production_setup() {
    log_info "Configurando ambiente de produção..."
    
    local prod_scripts=(
        "check_prerequisites.sh"
        "create_directories.sh"
        "install_dependencies.sh"
        "create_configuration.sh"
        "create_base_docs.sh"
        "setup_github.sh"
        "setup_quality.sh"
        "validate.sh"
    )
    
    execute_script_list "${prod_scripts[@]}"
}

run_docs_only_setup() {
    log_info "Configurando apenas documentação..."
    
    local docs_scripts=(
        "create_directories.sh"
        "create_configuration.sh"
        "create_base_docs.sh"
        "create_templates.sh"
        "generate_content.sh"
    )
    
    execute_script_list "${docs_scripts[@]}"
}

run_automation_only_setup() {
    log_info "Configurando apenas automação..."
    
    local automation_scripts=(
        "setup_automation.sh"
        "setup_github.sh"
        "setup_quality.sh"
    )
    
    execute_script_list "${automation_scripts[@]}"
}

run_minimal_setup() {
    log_info "Configurando setup mínimo..."
    
    local minimal_scripts=(
        "check_prerequisites.sh"
        "create_directories.sh"
        "install_dependencies.sh"
        "create_configuration.sh"
    )
    
    execute_script_list "${minimal_scripts[@]}"
}

run_restore_setup() {
    log_info "Função de restauração ainda não implementada"
    log_info "Esta funcionalidade será adicionada em versões futuras"
    return 0
}

execute_script_list() {
    local scripts=("$@")
    local completed=0
    local failed=0
    
    for script in "${scripts[@]}"; do
        local script_path="$SCRIPT_DIR/$script"
        
        if [[ -f "$script_path" && -x "$script_path" ]]; then
            if "$script_path"; then
                ((completed++))
            else
                ((failed++))
            fi
        else
            ((failed++))
        fi
    done
    
    echo -e "\n${CYAN}Resultado: $completed sucesso, $failed falhas${NC}"
    return $((failed > 0 ? 1 : 0))
}

run_prerequisites_check() {
    log_step "Verificando Pré-requisitos"
    
    local prereq_script="$SCRIPT_DIR/check_prerequisites.sh"
    
    if [[ -f "$prereq_script" && -x "$prereq_script" ]]; then
        "$prereq_script"
    else
        log_error "Script de pré-requisitos não encontrado: $prereq_script"
        return 1
    fi
}

show_help() {
    clear
    echo -e "${PURPLE}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                     NEURAL CRYPTO BOT 2.0 - SETUP HELP                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
    
    echo -e "${CYAN}${BOLD}📖 Documentação do Sistema de Setup${NC}\n"
    
    echo -e "${WHITE}${BOLD}VISÃO GERAL${NC}"
    echo -e "Este sistema configura uma documentação enterprise-grade usando MkDocs Material"
    echo -e "com recursos avançados de navegação, busca, versionamento e publicação.\n"
    
    echo -e "${WHITE}${BOLD}COMPONENTES PRINCIPAIS${NC}"
    echo -e "${GREEN}• Estrutura de Diretórios:${NC} Organização hierárquica padronizada"
    echo -e "${GREEN}• Dependências:${NC} Python (Poetry), Node.js (NPM), ferramentas auxiliares"
    echo -e "${GREEN}• Configuração:${NC} MkDocs com Material theme personalizado"
    echo -e "${GREEN}• Conteúdo Base:${NC} Templates e documentos iniciais"
    echo -e "${GREEN}• Automação:${NC} Scripts de build, deploy e validação"
    echo -e "${GREEN}• Assets:${NC} Estilos, JavaScript, imagens e recursos visuais"
    echo -e "${GREEN}• Integração:${NC} GitHub Pages, CI/CD, analytics"
    echo -e "${GREEN}• Qualidade:${NC} Validação de markdown, links, acessibilidade\n"
    
    echo -e "${WHITE}${BOLD}REQUISITOS DO SISTEMA${NC}"
    echo -e "${CYAN}Mínimos:${NC}"
    echo -e "• Python 3.11+ com pip"
    echo -e "• Poetry (gerenciador de dependências Python)"
    echo -e "• Git (controle de versão)"
    echo -e "• 4GB RAM, 10GB espaço livre"
    echo -e "• Conexão com internet estável"
    echo -e "\n${CYAN}Recomendados:${NC}"
    echo -e "• Node.js 18+ com npm (ferramentas auxiliares)"
    echo -e "• Docker (para ambiente containerizado)"
    echo -e "• 8GB+ RAM, 20GB+ espaço livre\n"
    
    echo -e "${WHITE}${BOLD}MODOS DE INSTALAÇÃO${NC}"
    echo -e "${GREEN}1. Setup Completo:${NC} Instala tudo automaticamente (recomendado)"
    echo -e "${GREEN}2. Setup Interativo:${NC} Escolha componentes específicos"
    echo -e "${GREEN}3. Setup Personalizado:${NC} Configurações avançadas por ambiente"
    echo -e "${GREEN}4. Verificação:${NC} Apenas testa pré-requisitos\n"
    
    echo -e "${WHITE}${BOLD}ESTRUTURA DE ARQUIVOS${NC}"
    echo -e "Após a instalação, você terá:"
    echo -e "${DIM}Neural-Crypto-Bot-2.0/${NC}"
    echo -e "${DIM}├── docs/                    # Documentação${NC}"
    echo -e "${DIM}├── mkdocs.yml              # Configuração principal${NC}"
    echo -e "${DIM}├── scripts/docs/           # Scripts de automação${NC}"
    echo -e "${DIM}├── site/                   # Build da documentação${NC}"
    echo -e "${DIM}└── .docs-*                 # Arquivos de configuração${NC}\n"
    
    echo -e "${WHITE}${BOLD}COMANDOS ÚTEIS${NC}"
    echo -e "${CYAN}Desenvolvimento:${NC}"
    echo -e "• ${BOLD}poetry run mkdocs serve${NC} - Servidor local"
    echo -e "• ${BOLD}poetry run mkdocs build${NC} - Build estático"
    echo -e "• ${BOLD}./scripts/docs/validate.sh${NC} - Validar documentação"
    echo -e "\n${CYAN}Produção:${NC}"
    echo -e "• ${BOLD}poetry run mkdocs gh-deploy${NC} - Deploy GitHub Pages"
    echo -e "• ${BOLD}./scripts/docs/publish_docs.sh${NC} - Pipeline completo"
    echo -e "• ${BOLD}./scripts/docs/version_docs.sh${NC} - Versionamento\n"
    
    echo -e "${WHITE}${BOLD}SOLUÇÃO DE PROBLEMAS${NC}"
    echo -e "${RED}Erro de permissão:${NC} Execute com sudo ou ajuste permissões"
    echo -e "${RED}Dependência não encontrada:${NC} Execute check_prerequisites.sh"
    echo -e "${RED}Falha no build:${NC} Verifique sintaxe do mkdocs.yml"
    echo -e "${RED}Links quebrados:${NC} Execute validators/validate_links.sh\n"
    
    echo -e "${WHITE}${BOLD}SUPORTE${NC}"
    echo -e "• ${CYAN}Logs:${NC} .docs-setup-logs/"
    echo -e "• ${CYAN}Configuração:${NC} .docs-config"
    echo -e "• ${CYAN}Relatórios:${NC} .docs-*-report.md"
    echo -e "• ${CYAN}GitHub:${NC} https://github.com/neural-crypto-bot/neural-crypto-bot-2.0"
    echo -e "• ${CYAN}Documentação:${NC} https://docs.neuralcryptobot.com\n"
    
    echo -ne "${YELLOW}Pressione ENTER para voltar ao menu principal...${NC}"
    read -r
}

show_completion_summary() {
    echo -e "\n${GREEN}${BOLD}🎉 CONFIGURAÇÃO CONCLUÍDA COM SUCESSO!${NC}\n"
    
    echo -e "${CYAN}${BOLD}📋 O que foi configurado:${NC}"
    echo -e "${WHITE}✅ Estrutura completa de diretórios${NC}"
    echo -e "${WHITE}✅ Dependências Python e Node.js instaladas${NC}"
    echo -e "${WHITE}✅ Configuração MkDocs Material personalizada${NC}"
    echo -e "${WHITE}✅ Templates e documentos base criados${NC}"
    echo -e "${WHITE}✅ Scripts de automação configurados${NC}"
    echo -e "${WHITE}✅ Assets e recursos visuais${NC}"
    echo -e "${WHITE}✅ Integração GitHub configurada${NC}"
    echo -e "${WHITE}✅ Ferramentas de qualidade ativadas${NC}"
    echo -e "${WHITE}✅ Validação de configuração aprovada${NC}\n"
    
    echo -e "${CYAN}${BOLD}🚀 Próximos Passos:${NC}"
    echo -e "${WHITE}1. ${BOLD}Testar o servidor local:${NC}"
    echo -e "${DIM}   cd $(basename "$PROJECT_ROOT")${NC}"
    echo -e "${DIM}   poetry run mkdocs serve${NC}"
    echo -e "${DIM}   Abrir: http://localhost:8000${NC}\n"
    
    echo -e "${WHITE}2. ${BOLD}Personalizar conteúdo:${NC}"
    echo -e "${DIM}   Editar arquivos em docs/01-getting-started/${NC}"
    echo -e "${DIM}   Adicionar logos em docs/assets/images/${NC}"
    echo -e "${DIM}   Personalizar cores em docs/stylesheets/${NC}\n"
    
    echo -e "${WHITE}3. ${BOLD}Configurar deploy:${NC}"
    echo -e "${DIM}   ./scripts/docs/setup_github.sh${NC}"
    echo -e "${DIM}   poetry run mkdocs gh-deploy${NC}\n"
    
    echo -e "${WHITE}4. ${BOLD}Desenvolver conteúdo:${NC}"
    echo -e "${DIM}   Usar templates em docs/templates/${NC}"
    echo -e "${DIM}   Validar com ./scripts/docs/validate.sh${NC}\n"
    
    echo -e "${CYAN}${BOLD}🔗 Links Úteis:${NC}"
    echo -e "${WHITE}• Servidor local: ${BOLD}http://localhost:8000${NC}"
    echo -e "${WHITE}• Documentação MkDocs: ${BOLD}https://www.mkdocs.org${NC}"
    echo -e "${WHITE}• Material Theme: ${BOLD}https://squidfunk.github.io/mkdocs-material${NC}"
    echo -e "${WHITE}• Repositório: ${BOLD}https://github.com/neural-crypto-bot/neural-crypto-bot-2.0${NC}\n"
    
    echo -e "${GREEN}${BOLD}Documentação Neural Crypto Bot 2.0 está pronta! 🚀${NC}"
}

# ================================
# FUNÇÃO PRINCIPAL
# ================================

main() {
    # Inicializar sistema
    init_docs_system
    
    # Mostrar banner de boas-vindas
    show_welcome_banner
    
    # Menu principal
    while true; do
        show_setup_options
        local choice
        choice=$(get_user_choice)
        
        case $choice in
            1)
                echo -e "\n${CYAN}${BOLD}🚀 Iniciando Setup Completo...${NC}"
                if run_complete_setup; then
                    break
                else
                    echo -ne "\n${YELLOW}Pressione ENTER para voltar ao menu...${NC}"
                    read -r
                fi
                ;;
            2)
                echo -e "\n${CYAN}${BOLD}🎛️ Iniciando Setup Interativo...${NC}"
                run_interactive_setup
                echo -ne "\n${YELLOW}Pressione ENTER para voltar ao menu...${NC}"
                read -r
                ;;
            3)
                echo -e "\n${CYAN}${BOLD}🔧 Iniciando Setup Personalizado...${NC}"
                run_custom_setup
                echo -ne "\n${YELLOW}Pressione ENTER para voltar ao menu...${NC}"
                read -r
                ;;
            4)
                echo -e "\n${CYAN}${BOLD}🔍 Verificando Pré-requisitos...${NC}"
                run_prerequisites_check
                echo -ne "\n${YELLOW}Pressione ENTER para voltar ao menu...${NC}"
                read -r
                ;;
            5)
                show_help
                ;;
            6)
                echo -e "\n${CYAN}Saindo do setup. Até logo! 👋${NC}"
                exit 0
                ;;
        esac
        
        clear
        display_main_banner
    done
    
    # Mostrar mensagem final
    echo -ne "\n${GREEN}Setup concluído! Pressione ENTER para sair...${NC}"
    read -r
    
    return 0
}

# ================================
# TRATAMENTO DE SINAIS
# ================================

cleanup_on_exit() {
    echo -e "\n\n${YELLOW}Setup interrompido pelo usuário.${NC}"
    echo -e "${YELLOW}Arquivos temporários sendo limpos...${NC}"
    
    # Limpar arquivos temporários se necessário
    rm -f /tmp/docs-setup-* 2>/dev/null || true
    
    echo -e "${GREEN}Limpeza concluída. Até logo! 👋${NC}"
    exit 0
}

trap cleanup_on_exit INT TERM

# ================================
# EXECUÇÃO
# ================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi