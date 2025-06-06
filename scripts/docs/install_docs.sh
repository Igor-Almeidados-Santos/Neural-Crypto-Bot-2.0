#!/bin/bash
# Neural Crypto Bot 2.0 - Instalador de Documentação para Unix (macOS/Linux)
# Este script detecta o ambiente e executa o instalador apropriado

set -e

# Definir cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Diretórios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DOCS_DIR="$PROJECT_ROOT/docs"
SCRIPTS_DIR="$PROJECT_ROOT/scripts/docs"

# Banner
echo -e "${PURPLE}${BOLD}"
echo "==============================================="
echo "    NEURAL CRYPTO BOT 2.0 - DOCUMENTATION     "
echo "           UNIX INSTALLER (macOS/Linux)       "
echo "==============================================="
echo -e "${NC}\n"

# Funções de log
log_info() {
    echo -e "${BLUE}${BOLD}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}${BOLD}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}${BOLD}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}${BOLD}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}${BOLD}=== $1 ===${NC}"
}

# Detectar sistema operacional
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            echo "$ID"
        else
            echo "linux"
        fi
    else
        echo "unknown"
    fi
}

# Verificar dependências básicas
check_dependencies() {
    log_step "Verificando dependências básicas"
    
    # Verificar Python
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python não encontrado. Por favor, instale Python 3.11 ou superior."
        return 1
    fi
    
    # Verificar versão do Python
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 11) ]]; then
        log_warning "Versão do Python ($PYTHON_VERSION) é menor que a recomendada (3.11+)"
    else
        log_success "Python $PYTHON_VERSION encontrado"
    fi
    
    # Verificar pip
    if $PYTHON_CMD -m pip --version &>/dev/null; then
        log_success "pip encontrado"
    else
        log_error "pip não encontrado. Por favor, instale pip."
        return 1
    fi
    
    # Verificar Git
    if command -v git &>/dev/null; then
        log_success "Git encontrado: $(git --version)"
    else
        log_warning "Git não encontrado. Algumas funcionalidades podem não funcionar corretamente."
    fi
    
    return 0
}

# Verificar permissões de escrita
check_write_permissions() {
    log_step "Verificando permissões de escrita"
    
    if [[ -w "$PROJECT_ROOT" ]]; then
        log_success "Permissões de escrita OK para $PROJECT_ROOT"
        return 0
    else
        log_error "Sem permissões de escrita para $PROJECT_ROOT"
        log_error "Por favor, execute o script com as permissões adequadas"
        return 1
    fi
}

# Executar script bash
run_bash_script() {
    local script="$1"
    local script_path="$SCRIPTS_DIR/$script"
    
    if [[ ! -f "$script_path" ]]; then
        log_error "Script não encontrado: $script_path"
        return 1
    fi
    
    # Garantir que o script tenha permissão de execução
    chmod +x "$script_path"
    
    # Executar o script
    log_info "Executando $script..."
    "$script_path"
    local status=$?
    
    if [[ $status -eq 0 ]]; then
        log_success "Script $script executado com sucesso"
        return 0
    else
        log_error "Falha ao executar $script (código de saída: $status)"
        return 1
    fi
}

# Executar script via Python
run_python_wrapper() {
    log_step "Executando via wrapper Python"
    
    local wrapper_path="$SCRIPTS_DIR/cross_platform_wrapper.py"
    
    # Verificar se o wrapper existe
    if [[ ! -f "$wrapper_path" ]]; then
        log_error "Wrapper Python não encontrado: $wrapper_path"
        return 1
    fi
    
    # Dar permissão de execução
    chmod +x "$wrapper_path"
    
    # Executar
    "$PYTHON_CMD" "$wrapper_path"
    return $?
}

# Menu principal
show_menu() {
    echo -e "${CYAN}${BOLD}Opções de instalação:${NC}\n"
    echo -e "1. ${BOLD}Configuração Nativa${NC} - Usar scripts bash nativos"
    echo -e "2. ${BOLD}Configuração Python${NC} - Usar wrapper Python multiplataforma"
    echo -e "3. ${BOLD}Verificar dependências${NC} - Apenas verificar o ambiente"
    echo -e "4. ${BOLD}Sair${NC}"
    echo ""
    
    read -p "Escolha uma opção [1-4]: " OPTION
    echo ""
    
    case $OPTION in
        1)
            run_native_setup
            ;;
        2)
            run_python_wrapper
            ;;
        3)
            check_dependencies
            ;;
        4)
            log_info "Saindo do instalador"
            exit 0
            ;;
        *)
            log_error "Opção inválida"
            return 1
            ;;
    esac
}

# Executar setup nativo
run_native_setup() {
    log_step "Executando configuração nativa"
    
    # Verificar permissões primeiro
    check_write_permissions || return 1
    
    # Oferecer opções de scripts
    echo -e "${CYAN}${BOLD}Escolha o script a executar:${NC}\n"
    echo -e "1. ${BOLD}Setup Completo${NC} - Configuração completa do sistema de documentação"
    echo -e "2. ${BOLD}Verificar Pré-requisitos${NC} - Verificar dependências do sistema"
    echo -e "3. ${BOLD}Criar Diretórios${NC} - Criar estrutura de diretórios"
    echo -e "4. ${BOLD}Instalar Dependências${NC} - Instalar pacotes necessários"
    echo -e "5. ${BOLD}Criar Configuração${NC} - Configurar MkDocs"
    echo -e "6. ${BOLD}Publicar Documentação${NC} - Build e deploy da documentação"
    echo -e "7. ${BOLD}Voltar${NC}"
    echo ""
    
    read -p "Escolha uma opção [1-7]: " SCRIPT_OPTION
    echo ""
    
    case $SCRIPT_OPTION in
        1)
            run_bash_script "setup_documentation.sh"
            ;;
        2)
            run_bash_script "check_prerequisites.sh"
            ;;
        3)
            run_bash_script "create_directories.sh"
            ;;
        4)
            run_bash_script "install_dependencies.sh"
            ;;
        5)
            run_bash_script "create_configuration.sh"
            ;;
        6)
            run_bash_script "publish_docs.sh"
            ;;
        7)
            return 0
            ;;
        *)
            log_error "Opção inválida"
            return 1
            ;;
    esac
}

# Função principal
main() {
    # Detectar sistema operacional
    OS_TYPE=$(detect_os)
    log_info "Sistema operacional detectado: $OS_TYPE"
    
    # Verificar se estamos no diretório correto
    if [[ ! -d "$SCRIPTS_DIR" ]]; then
        log_error "Diretório de scripts não encontrado: $SCRIPTS_DIR"
        log_error "Por favor, execute este script do diretório correto"
        exit 1
    fi
    
    # Verificar dependências básicas
    if ! check_dependencies; then
        log_error "Falha na verificação de dependências básicas"
        exit 1
    fi
    
    # Criar cross_platform_wrapper.py se não existir
    if [[ ! -f "$SCRIPTS_DIR/cross_platform_wrapper.py" ]]; then
        log_warning "Wrapper Python não encontrado. Criando..."
        
        # Aqui poderíamos criar o wrapper, mas para simplicidade
        # vamos sugerir que o usuário use o método nativo
        log_info "Recomendamos usar a configuração nativa para este ambiente"
    fi
    
    # Mostrar menu e executar opção escolhida
    while true; do
        show_menu
        
        echo ""
        read -p "Pressione ENTER para continuar ou 'q' para sair: " CONTINUE
        [[ "$CONTINUE" == "q" ]] && break
        echo ""
    done
    
    log_success "Instalação concluída!"
    return 0
}

# Executar função principal
main