#!/bin/bash
# scripts/docs/common.sh - Funções e constantes comuns para o sistema de documentação
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Sistema de documentação de classe mundial

set -euo pipefail

# ================================
# CONFIGURAÇÕES GLOBAIS E CONSTANTES
# ================================

# Versões e configurações essenciais
readonly DOCS_SYSTEM_VERSION="2.0.0"
readonly MKDOCS_VERSION="1.5.3"
readonly MATERIAL_VERSION="9.4.8"
readonly PYTHON_MIN_VERSION="3.11"
readonly NODE_MIN_VERSION="18"

# Paths principais
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
readonly DOCS_DIR="$PROJECT_ROOT/docs"
readonly SCRIPTS_DOCS_DIR="$PROJECT_ROOT/scripts/docs"
readonly ASSETS_DIR="$DOCS_DIR/assets"
readonly TEMPLATES_DIR="$SCRIPTS_DOCS_DIR/templates"
readonly VALIDATORS_DIR="$SCRIPTS_DOCS_DIR/validators"

# Configurações de sistema
readonly LOG_DIR="$PROJECT_ROOT/.docs-setup-logs"
readonly LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
readonly CONFIG_FILE="$PROJECT_ROOT/.docs-config"

# ================================
# CORES E FORMATAÇÃO
# ================================

# Cores ANSI para output elegante
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[0;37m'
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly UNDERLINE='\033[4m'
readonly BLINK='\033[5m'
readonly REVERSE='\033[7m'
readonly NC='\033[0m' # No Color

# Ícones e símbolos
readonly ICON_SUCCESS="✅"
readonly ICON_ERROR="❌"
readonly ICON_WARNING="⚠️"
readonly ICON_INFO="ℹ️"
readonly ICON_ROCKET="🚀"
readonly ICON_GEAR="⚙️"
readonly ICON_BOOK="📚"
readonly ICON_COMPUTER="💻"
readonly ICON_SHIELD="🛡️"
readonly ICON_SPARKLES="✨"
readonly ICON_FIRE="🔥"
readonly ICON_LIGHTNING="⚡"
readonly ICON_TARGET="🎯"
readonly ICON_ROBOT="🤖"

# ================================
# FUNÇÕES DE LOGGING AVANÇADAS
# ================================

# Inicializar sistema de logging
init_logging() {
    mkdir -p "$LOG_DIR"
    touch "$LOG_FILE"
    
    # Escrever cabeçalho do log
    cat > "$LOG_FILE" << EOF
================================================================================
Neural Crypto Bot 2.0 - Documentation Setup Log
Timestamp: $(date '+%Y-%m-%d %H:%M:%S %Z')
System: $(uname -a)
User: $(whoami)
Script: ${0##*/}
================================================================================

EOF
}

# Função de logging com timestamp
log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Funções de logging com estilo
log_info() {
    local message="$1"
    echo -e "${BLUE}${BOLD}[INFO]${NC} ${ICON_INFO} $message"
    log_message "INFO" "$message"
}

log_success() {
    local message="$1"
    echo -e "${GREEN}${BOLD}[SUCCESS]${NC} ${ICON_SUCCESS} $message"
    log_message "SUCCESS" "$message"
}

log_warning() {
    local message="$1"
    echo -e "${YELLOW}${BOLD}[WARNING]${NC} ${ICON_WARNING} $message"
    log_message "WARNING" "$message"
}

log_error() {
    local message="$1"
    echo -e "${RED}${BOLD}[ERROR]${NC} ${ICON_ERROR} $message" >&2
    log_message "ERROR" "$message"
}

log_step() {
    local message="$1"
    echo -e "\n${CYAN}${BOLD}${ICON_ROCKET} === $message ===${NC}"
    log_message "STEP" "$message"
}

log_substep() {
    local message="$1"
    echo -e "${PURPLE}${BOLD}${ICON_GEAR} --- $message ---${NC}"
    log_message "SUBSTEP" "$message"
}

log_debug() {
    local message="$1"
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${DIM}[DEBUG] $message${NC}"
        log_message "DEBUG" "$message"
    fi
}

# ================================
# FUNÇÕES DE VERIFICAÇÃO DE SISTEMA
# ================================

# Verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar versão do Python
check_python_version() {
    local python_cmd=""
    
    # Tentar diferentes comandos Python
    for cmd in python3.11 python3 python; do
        if command_exists "$cmd"; then
            python_cmd="$cmd"
            break
        fi
    done
    
    if [[ -z "$python_cmd" ]]; then
        return 1
    fi
    
    local version
    version=$($python_cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    
    if [[ -z "$version" ]]; then
        return 1
    fi
    
    local major minor
    major=$(echo "$version" | cut -d"." -f1)
    minor=$(echo "$version" | cut -d"." -f2)
    
    if [[ $major -eq 3 && $minor -ge 11 ]]; then
        echo "$python_cmd"
        return 0
    else
        return 1
    fi
}

# Verificar versão do Node.js
check_node_version() {
    if ! command_exists node; then
        return 1
    fi
    
    local version
    version=$(node --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+')
    local major
    major=$(echo "$version" | cut -d"." -f1)
    
    if [[ $major -ge 18 ]]; then
        echo "node"
        return 0
    else
        return 1
    fi
}

# Verificar espaço em disco disponível
check_disk_space() {
    local required_gb="${1:-2}"
    local path="${2:-$PROJECT_ROOT}"
    
    if command_exists df; then
        local available_kb
        available_kb=$(df -k "$path" | awk 'NR==2 {print $4}')
        local available_gb=$((available_kb / 1024 / 1024))
        
        if [[ $available_gb -ge $required_gb ]]; then
            echo "$available_gb"
            return 0
        else
            return 1
        fi
    else
        # Fallback para sistemas sem df
        return 0
    fi
}

# Verificar conectividade com a internet
check_internet_connectivity() {
    local test_urls=(
        "https://pypi.org"
        "https://github.com"
        "https://registry.npmjs.org"
    )
    
    for url in "${test_urls[@]}"; do
        if curl -s --head --max-time 10 "$url" >/dev/null 2>&1; then
            return 0
        fi
    done
    
    return 1
}

# ================================
# FUNÇÕES DE MANIPULAÇÃO DE ARQUIVOS
# ================================

# Criar diretório com logs e verificação
create_directory() {
    local dir_path="$1"
    local description="${2:-directory}"
    
    if [[ -d "$dir_path" ]]; then
        log_debug "Diretório já existe: $dir_path"
        return 0
    fi
    
    if mkdir -p "$dir_path" 2>/dev/null; then
        log_debug "Criado $description: $(basename "$dir_path")"
        return 0
    else
        log_error "Falha ao criar $description: $dir_path"
        return 1
    fi
}

# Fazer backup de arquivo
backup_file() {
    local file_path="$1"
    local backup_suffix="${2:-.backup.$(date +%Y%m%d%H%M%S)}"
    
    if [[ -f "$file_path" ]]; then
        local backup_path="${file_path}${backup_suffix}"
        if cp "$file_path" "$backup_path"; then
            log_info "Backup criado: $(basename "$backup_path")"
            echo "$backup_path"
            return 0
        else
            log_error "Falha ao criar backup de: $(basename "$file_path")"
            return 1
        fi
    fi
    
    return 0
}

# Verificar permissões de escrita
check_write_permissions() {
    local path="$1"
    
    if [[ -w "$path" ]]; then
        return 0
    elif [[ ! -e "$path" ]] && [[ -w "$(dirname "$path")" ]]; then
        return 0
    else
        return 1
    fi
}

# ================================
# FUNÇÕES DE CONFIGURAÇÃO
# ================================

# Salvar configuração
save_config() {
    local key="$1"
    local value="$2"
    
    local config_dir
    config_dir="$(dirname "$CONFIG_FILE")"
    create_directory "$config_dir" "configuration directory"
    
    # Criar arquivo de configuração se não existir
    if [[ ! -f "$CONFIG_FILE" ]]; then
        cat > "$CONFIG_FILE" << 'EOF'
# Neural Crypto Bot 2.0 - Documentation System Configuration
# Generated automatically - do not edit manually
EOF
    fi
    
    # Remover configuração existente e adicionar nova
    grep -v "^$key=" "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" 2>/dev/null || true
    echo "$key=$value" >> "${CONFIG_FILE}.tmp"
    mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
    
    log_debug "Configuração salva: $key=$value"
}

# Carregar configuração
load_config() {
    local key="$1"
    local default_value="${2:-}"
    
    if [[ -f "$CONFIG_FILE" ]]; then
        local value
        value=$(grep "^$key=" "$CONFIG_FILE" | cut -d'=' -f2- | tail -1)
        echo "${value:-$default_value}"
    else
        echo "$default_value"
    fi
}

# ================================
# FUNÇÕES DE VALIDAÇÃO
# ================================

# Validar estrutura de projeto
validate_project_structure() {
    local required_files=(
        "pyproject.toml"
        "README.md"
        ".gitignore"
    )
    
    local missing_files=()
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            missing_files+=("$file")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "Arquivos obrigatórios não encontrados: ${missing_files[*]}"
        return 1
    fi
    
    return 0
}

# Validar dependências Python
validate_python_dependencies() {
    local python_cmd="$1"
    
    # Verificar Poetry
    if ! command_exists poetry; then
        log_error "Poetry não encontrado. Instale com: curl -sSL https://install.python-poetry.org | python3 -"
        return 1
    fi
    
    # Verificar se estamos em um projeto Poetry válido
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        log_error "Arquivo pyproject.toml não encontrado"
        return 1
    fi
    
    return 0
}

# ================================
# FUNÇÕES DE PROGRESSO E UI
# ================================

# Barra de progresso
show_progress() {
    local current="$1"
    local total="$2"
    local description="${3:-Processing}"
    local width=50
    
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    local remaining=$((width - completed))
    
    printf "\r${BLUE}%s:${NC} [" "$description"
    printf "%*s" $completed | tr ' ' '='
    printf "%*s" $remaining | tr ' ' '-'
    printf "] %d%% (%d/%d)" $percentage $current $total
    
    if [[ $current -eq $total ]]; then
        echo ""
    fi
}

# Spinner de loading
show_spinner() {
    local pid="$1"
    local message="${2:-Loading}"
    local chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    local i=0
    
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r${BLUE}%s${NC} %s" "$message" "${chars:$i:1}"
        i=$(((i + 1) % ${#chars}))
        sleep 0.1
    done
    
    printf "\r%*s\r" $((${#message} + 3)) ""
}

# ================================
# FUNÇÕES DE DETECÇÃO DE SISTEMA
# ================================

# Detectar sistema operacional
detect_os() {
    case "$(uname -s)" in
        Linux*)
            if command_exists lsb_release; then
                echo "$(lsb_release -si | tr '[:upper:]' '[:lower:]')"
            elif [[ -f /etc/os-release ]]; then
                grep '^ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"'
            else
                echo "linux"
            fi
            ;;
        Darwin*)
            echo "macos"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "windows"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Detectar arquitetura
detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64)
            echo "x86_64"
            ;;
        arm64|aarch64)
            echo "arm64"
            ;;
        armv7l)
            echo "arm"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# ================================
# FUNÇÕES DE CLEANUP E RECUPERAÇÃO
# ================================

# Função de cleanup em caso de erro
cleanup_on_error() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        log_error "Setup falhou com código de saída: $exit_code"
        log_info "Log completo disponível em: $LOG_FILE"
        
        # Oferecer opções de recuperação
        echo -e "\n${YELLOW}Opções de recuperação:${NC}"
        echo "1. Verificar o log de erro: cat $LOG_FILE"
        echo "2. Executar diagnóstico: ./scripts/docs/check_prerequisites.sh"
        echo "3. Reiniciar setup: ./scripts/docs/setup_documentation.sh"
    fi
    
    exit $exit_code
}

# Configurar trap para cleanup
setup_error_handling() {
    trap cleanup_on_error ERR
    trap 'log_info "Setup interrompido pelo usuário"' INT TERM
}

# ================================
# FUNÇÕES DE BANNER E APRESENTAÇÃO
# ================================

# Banner principal do sistema
display_main_banner() {
    echo -e "${PURPLE}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                        NEURAL CRYPTO BOT 2.0                                ║
║                     DOCUMENTATION SYSTEM SETUP                              ║
║                                                                              ║
║                    🚀 Enterprise-Grade Documentation                        ║
║                         powered by MkDocs Material                          ║
║                                                                              ║
║                     Desenvolvido por Oasis Systems                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
}

# Banner de conclusão
display_completion_banner() {
    echo -e "\n${GREEN}${BOLD}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                          🎉 SETUP CONCLUÍDO! 🎉                            ║
║                                                                              ║
║              Sistema de documentação configurado com sucesso!               ║
║                                                                              ║
║                    Neural Crypto Bot 2.0 está pronto                       ║
║                      para documentação de classe mundial                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
}

# ================================
# FUNÇÕES DE MÉTRICAS E ESTATÍSTICAS
# ================================

# Coletar estatísticas do setup
collect_setup_stats() {
    local start_time="$1"
    local end_time="$2"
    local total_files="${3:-0}"
    local total_dirs="${4:-0}"
    
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    save_config "SETUP_DURATION" "$duration"
    save_config "SETUP_FILES_CREATED" "$total_files"
    save_config "SETUP_DIRS_CREATED" "$total_dirs"
    save_config "SETUP_TIMESTAMP" "$(date -d @$end_time '+%Y-%m-%d %H:%M:%S')"
    
    cat << EOF

${CYAN}${BOLD}📊 Estatísticas do Setup:${NC}
${WHITE}├── Duração: ${minutes}m ${seconds}s${NC}
${WHITE}├── Arquivos criados: $total_files${NC}
${WHITE}├── Diretórios criados: $total_dirs${NC}
${WHITE}├── Sistema: $(detect_os) $(detect_arch)${NC}
${WHITE}├── Python: $(check_python_version 2>/dev/null || echo "não encontrado")${NC}
${WHITE}├── Log: $LOG_FILE${NC}
${WHITE}└── Timestamp: $(date)${NC}

EOF
}

# ================================
# FUNÇÕES DE DIAGNÓSTICO
# ================================

# Executar diagnóstico completo do sistema
run_system_diagnostics() {
    log_step "Executando Diagnóstico do Sistema"
    
    # Informações do sistema
    log_info "Sistema Operacional: $(detect_os) $(detect_arch)"
    log_info "Usuário: $(whoami)"
    log_info "Diretório de trabalho: $(pwd)"
    log_info "Shell: $SHELL"
    
    # Verificar recursos do sistema
    if command_exists free; then
        local memory_info
        memory_info=$(free -h | grep '^Mem:' | awk '{print $2 " total, " $7 " available"}')
        log_info "Memória: $memory_info"
    fi
    
    if command_exists df; then
        local disk_info
        disk_info=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4 " available on " $6}')
        log_info "Espaço em disco: $disk_info"
    fi
    
    # Verificar conectividade
    if check_internet_connectivity; then
        log_success "Conectividade com internet: OK"
    else
        log_warning "Conectividade com internet: Limitada"
    fi
    
    # Verificar dependências principais
    local deps_status=()
    
    if command_exists python3; then
        deps_status+=("Python: $(python3 --version)")
    else
        deps_status+=("Python: ❌ Não encontrado")
    fi
    
    if command_exists poetry; then
        deps_status+=("Poetry: $(poetry --version)")
    else
        deps_status+=("Poetry: ❌ Não encontrado")
    fi
    
    if command_exists git; then
        deps_status+=("Git: $(git --version)")
    else
        deps_status+=("Git: ❌ Não encontrado")
    fi
    
    if command_exists node; then
        deps_status+=("Node.js: $(node --version)")
    else
        deps_status+=("Node.js: ⚠️ Opcional")
    fi
    
    for status in "${deps_status[@]}"; do
        log_info "$status"
    done
}

# ================================
# INICIALIZAÇÃO
# ================================

# Inicializar sistema de logging e configuração
init_docs_system() {
    # Inicializar logging
    init_logging
    
    # Configurar tratamento de erros
    setup_error_handling
    
    # Salvar informações do sistema
    save_config "DOCS_SYSTEM_VERSION" "$DOCS_SYSTEM_VERSION"
    save_config "SETUP_OS" "$(detect_os)"
    save_config "SETUP_ARCH" "$(detect_arch)"
    save_config "SETUP_USER" "$(whoami)"
    save_config "PROJECT_ROOT" "$PROJECT_ROOT"
    
    log_debug "Sistema de documentação inicializado"
}

# ================================
# EXPORTAR VARIÁVEIS E FUNÇÕES
# ================================

# Exportar variáveis para scripts filhos
export DOCS_SYSTEM_VERSION
export PROJECT_ROOT
export DOCS_DIR
export SCRIPTS_DOCS_DIR
export LOG_FILE
export CONFIG_FILE

# Verificar se script está sendo executado diretamente
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo -e "${YELLOW}Este é um arquivo de funções comuns.${NC}"
    echo -e "Para usar, inclua em seu script: ${BOLD}source $(basename "$0")${NC}"
    echo -e "\nFunções disponíveis:"
    echo -e "  ${CYAN}• Logging:${NC} log_info, log_success, log_warning, log_error"
    echo -e "  ${CYAN}• Sistema:${NC} check_python_version, detect_os, command_exists"
    echo -e "  ${CYAN}• Arquivos:${NC} create_directory, backup_file, check_write_permissions"
    echo -e "  ${CYAN}• Config:${NC} save_config, load_config"
    echo -e "  ${CYAN}• UI:${NC} show_progress, show_spinner, display_main_banner"
    echo -e "  ${CYAN}• Diagnóstico:${NC} run_system_diagnostics, validate_project_structure"
fi