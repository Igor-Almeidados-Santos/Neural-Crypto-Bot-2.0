#!/bin/bash
# scripts/setup.sh - Setup completo do Neural Crypto Bot 2.0
# Script master que executa toda a configuração inicial

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# ================================
# CONFIGURAÇÕES E VARIÁVEIS
# ================================

# Cores para output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[0;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configurações do projeto
readonly PROJECT_NAME="Neural Crypto Bot 2.0"
readonly MIN_PYTHON_VERSION="3.11"
readonly REQUIRED_DISK_GB=10
readonly REQUIRED_RAM_GB=4

# Detectar sistema operacional
detect_os() {
    case "$(uname -s)" in
        Linux*)     echo "Linux";;
        Darwin*)    echo "macOS";;
        MINGW*|MSYS*|CYGWIN*)    echo "Windows";;
        *)          echo "Unknown";;
    esac
}

readonly OS_TYPE=$(detect_os)

# ================================
# FUNÇÕES AUXILIARES
# ================================

# Logging com cores
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "\n${CYAN}=== $1 ===${NC}"
}

# Verificar se comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Verificar versão do Python
check_python_version() {
    if command_exists python3.11; then
        PYTHON_CMD="python3.11"
    elif command_exists python3; then
        PYTHON_CMD="python3"
    elif command_exists python; then
        PYTHON_CMD="python"
    else
        return 1
    fi

    local version
    version=$($PYTHON_CMD --version 2>&1 | cut -d" " -f2)
    local major
    major=$(echo "$version" | cut -d"." -f1)
    local minor
    minor=$(echo "$version" | cut -d"." -f2)

    if [[ $major -eq 3 && $minor -ge 11 ]]; then
        echo "$PYTHON_CMD"
        return 0
    else
        return 1
    fi
}

# Verificar recursos do sistema
check_system_resources() {
    log_step "Verificando Recursos do Sistema"

    # Verificar RAM
    case $OS_TYPE in
        "Linux")
            local ram_gb
            ram_gb=$(free -g | awk '/^Mem:/{print $2}')
            ;;
        "macOS")
            local ram_bytes
            ram_bytes=$(sysctl -n hw.memsize)
            ram_gb=$((ram_bytes / 1024 / 1024 / 1024))
            ;;
        *)
            log_warning "Não foi possível verificar RAM no $OS_TYPE"
            return 0
            ;;
    esac

    if [[ ${ram_gb:-0} -lt $REQUIRED_RAM_GB ]]; then
        log_warning "RAM disponível: ${ram_gb}GB (recomendado: ${REQUIRED_RAM_GB}GB+)"
    else
        log_success "RAM disponível: ${ram_gb}GB ✓"
    fi

    # Verificar espaço em disco
    local disk_gb
    disk_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    
    if [[ ${disk_gb:-0} -lt $REQUIRED_DISK_GB ]]; then
        log_error "Espaço em disco insuficiente: ${disk_gb}GB (necessário: ${REQUIRED_DISK_GB}GB+)"
        return 1
    else
        log_success "Espaço em disco disponível: ${disk_gb}GB ✓"
    fi
}

# Verificar dependências do sistema
check_system_dependencies() {
    log_step "Verificando Dependências do Sistema"

    local missing_deps=()

    # Verificar Python
    if PYTHON_CMD=$(check_python_version); then
        log_success "Python ${MIN_PYTHON_VERSION}+ encontrado: $($PYTHON_CMD --version)"
    else
        log_error "Python ${MIN_PYTHON_VERSION}+ não encontrado"
        missing_deps+=("python${MIN_PYTHON_VERSION}+")
    fi

    # Verificar Docker
    if command_exists docker; then
        log_success "Docker encontrado: $(docker --version)"
        
        # Verificar se Docker está rodando
        if docker info >/dev/null 2>&1; then
            log_success "Docker daemon está rodando ✓"
        else
            log_warning "Docker instalado mas daemon não está rodando"
            case $OS_TYPE in
                "Linux")
                    log_info "Execute: sudo systemctl start docker"
                    ;;
                "macOS")
                    log_info "Inicie o Docker Desktop"
                    ;;
                "Windows")
                    log_info "Inicie o Docker Desktop"
                    ;;
            esac
        fi
    else
        log_error "Docker não encontrado"
        missing_deps+=("docker")
    fi

    # Verificar Docker Compose
    if command_exists docker-compose || (command_exists docker && docker compose version >/dev/null 2>&1); then
        log_success "Docker Compose encontrado ✓"
    else
        log_error "Docker Compose não encontrado"
        missing_deps+=("docker-compose")
    fi

    # Verificar Git
    if command_exists git; then
        log_success "Git encontrado: $(git --version)"
    else
        log_error "Git não encontrado"
        missing_deps+=("git")
    fi

    # Se há dependências faltando, exibir instruções
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Dependências faltando: ${missing_deps[*]}"
        show_installation_instructions "${missing_deps[@]}"
        return 1
    fi

    log_success "Todas as dependências do sistema estão disponíveis ✓"
}

# Mostrar instruções de instalação
show_installation_instructions() {
    local deps=("$@")
    
    log_step "Instruções de Instalação"
    
    case $OS_TYPE in
        "Linux")
            echo "Execute os seguintes comandos para instalar as dependências:"
            echo ""
            for dep in "${deps[@]}"; do
                case $dep in
                    "python${MIN_PYTHON_VERSION}+")
                        echo "# Instalar Python ${MIN_PYTHON_VERSION}"
                        echo "sudo apt update"
                        echo "sudo apt install -y python3.11 python3.11-dev python3.11-venv"
                        ;;
                    "docker")
                        echo "# Instalar Docker"
                        echo "curl -fsSL https://get.docker.com -o get-docker.sh"
                        echo "sudo sh get-docker.sh"
                        echo "sudo usermod -aG docker \$USER"
                        ;;
                    "docker-compose")
                        echo "# Docker Compose (incluído no Docker moderno)"
                        echo "sudo apt install -y docker-compose-plugin"
                        ;;
                    "git")
                        echo "# Instalar Git"
                        echo "sudo apt install -y git"
                        ;;
                esac
                echo ""
            done
            ;;
        "macOS")
            echo "Execute os seguintes comandos para instalar as dependências:"
            echo ""
            echo "# Instalar Homebrew (se não tiver)"
            echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            echo ""
            for dep in "${deps[@]}"; do
                case $dep in
                    "python${MIN_PYTHON_VERSION}+")
                        echo "brew install python@3.11"
                        ;;
                    "docker"|"docker-compose")
                        echo "brew install --cask docker"
                        ;;
                    "git")
                        echo "brew install git"
                        ;;
                esac
            done
            ;;
        "Windows")
            echo "Instale as seguintes dependências:"
            echo ""
            for dep in "${deps[@]}"; do
                case $dep in
                    "python${MIN_PYTHON_VERSION}+")
                        echo "- Python 3.11+: https://python.org/downloads/"
                        ;;
                    "docker"|"docker-compose")
                        echo "- Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
                        ;;
                    "git")
                        echo "- Git: https://git-scm.com/download/win"
                        ;;
                esac
            done
            ;;
    esac
}

# Configurar Poetry
setup_poetry() {
    log_step "Configurando Poetry"

    if command_exists poetry; then
        log_success "Poetry já está instalado: $(poetry --version)"
    else
        log_info "Instalando Poetry..."
        curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
        
        # Adicionar ao PATH
        case $OS_TYPE in
            "Linux"|"macOS")
                export PATH="$HOME/.local/bin:$PATH"
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
                ;;
            "Windows")
                log_warning "Adicione Poetry ao PATH manualmente: %APPDATA%\\Python\\Scripts"
                ;;
        esac
    fi

    # Configurar Poetry
    if command_exists poetry; then
        poetry config virtualenvs.in-project true
        log_success "Poetry configurado para criar venv no projeto ✓"
    else
        log_error "Falha na instalação do Poetry"
        return 1
    fi
}

# Instalar dependências do projeto
install_dependencies() {
    log_step "Instalando Dependências do Projeto"

    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml não encontrado"
        return 1
    fi

    log_info "Instalando dependências com Poetry..."
    poetry install --no-interaction

    log_success "Dependências instaladas com sucesso ✓"
}

# Configurar arquivos de ambiente
setup_environment_files() {
    log_step "Configurando Arquivos de Ambiente"

    # Criar .env a partir do template
    if [[ ! -f ".env" ]]; then
        if [[ -f ".env.example" ]]; then
            cp .env.example .env
            log_success "Arquivo .env criado a partir do template ✓"
        else
            log_warning ".env.example não encontrado, criando .env básico"
            create_basic_env_file
        fi
    else
        log_info "Arquivo .env já existe, mantendo configurações atuais"
    fi

    # Configurar permissões no .env
    chmod 600 .env
    log_success "Permissões do .env configuradas (600) ✓"
}

# Criar arquivo .env básico
create_basic_env_file() {
    cat > .env << EOF
# Configurações gerais
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO
TIMEZONE=UTC

# Configurações de banco de dados
DATABASE_URL=postgresql://neuralbot:password@localhost:5432/neuralcryptobot
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CONSUMER_GROUP=ncb-group

# Segurança
SECRET_KEY=$(openssl rand -base64 32)
JWT_ALGORITHM=HS256
JWT_EXPIRATION_SECONDS=86400

# APIs de Exchange (CONFIGURE SUAS CHAVES AQUI)
BINANCE_API_KEY=
BINANCE_API_SECRET=
COINBASE_API_KEY=
COINBASE_API_SECRET=

# Configurações de trading
DEFAULT_TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT
MAX_POSITION_SIZE=0.05
MAX_LEVERAGE=2.0
RISK_FREE_RATE=0.03

# Telemetria
ENABLE_TELEMETRY=True
PROMETHEUS_PORT=9090
EOF

    log_success "Arquivo .env básico criado ✓"
}

# Configurar Docker
setup_docker() {
    log_step "Configurando Docker"

    # Verificar se docker-compose.yml existe
    if [[ ! -f "docker-compose.yml" ]]; then
        log_error "docker-compose.yml não encontrado"
        return 1
    fi

    # Criar diretórios necessários
    mkdir -p logs data models feature_store backups
    log_success "Diretórios criados ✓"

    # Fazer pull das imagens base
    log_info "Fazendo pull das imagens Docker base..."
    docker compose pull postgres redis kafka zookeeper grafana prometheus

    log_success "Configuração Docker concluída ✓"
}

# Configurar pre-commit hooks
setup_pre_commit() {
    log_step "Configurando Pre-commit Hooks"

    if [[ -f ".pre-commit-config.yaml" ]]; then
        poetry run pre-commit install
        log_success "Pre-commit hooks instalados ✓"
    else
        log_warning ".pre-commit-config.yaml não encontrado, pulando pre-commit"
    fi
}

# Executar testes básicos
run_basic_tests() {
    log_step "Executando Testes Básicos"

    # Testar importações básicas
    log_info "Testando importações do projeto..."
    if poetry run python -c "import src.common; print('✓ Importações funcionando')"; then
        log_success "Importações básicas funcionando ✓"
    else
        log_warning "Algumas importações falharam (normal se código ainda não estiver implementado)"
    fi

    # Testar se pytest funciona
    if [[ -d "tests" ]]; then
        log_info "Executando testes existentes..."
        poetry run pytest tests/ --maxfail=1 -q || log_warning "Alguns testes falharam"
    else
        log_info "Diretório de testes não encontrado, criando estrutura básica..."
        mkdir -p tests/{unit,integration,system}
        touch tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/system/__init__.py
    fi
}

# Verificar configuração final
verify_setup() {
    log_step "Verificando Configuração Final"

    local issues=()

    # Verificar .env
    if [[ -f ".env" ]]; then
        log_success ".env configurado ✓"
    else
        issues+=(".env não encontrado")
    fi

    # Verificar venv
    if [[ -d ".venv" ]]; then
        log_success "Ambiente virtual criado ✓"
    else
        issues+=("Ambiente virtual não criado")
    fi

    # Verificar Poetry
    if poetry --version >/dev/null 2>&1; then
        log_success "Poetry funcionando ✓"
    else
        issues+=("Poetry não funcionando")
    fi

    # Verificar Docker
    if docker info >/dev/null 2>&1; then
        log_success "Docker funcionando ✓"
    else
        issues+=("Docker não funcionando")
    fi

    if [[ ${#issues[@]} -eq 0 ]]; then
        log_success "Todas as verificações passaram ✓"
        return 0
    else
        log_error "Problemas encontrados:"
        for issue in "${issues[@]}"; do
            echo "  - $issue"
        done
        return 1
    fi
}

# Mostrar próximos passos
show_next_steps() {
    log_step "Setup Concluído - Próximos Passos"

    echo -e "${GREEN}✅ Setup do ${PROJECT_NAME} concluído com sucesso!${NC}"
    echo ""
    echo "📋 Próximos passos:"
    echo ""
    echo "1. ${CYAN}Configurar suas chaves de API:${NC}"
    echo "   - Edite o arquivo .env"
    echo "   - Adicione suas chaves da Binance, Coinbase, etc."
    echo ""
    echo "2. ${CYAN}Iniciar os serviços:${NC}"
    echo "   ./scripts/dev_utils.sh up"
    echo ""
    echo "3. ${CYAN}Verificar se tudo está funcionando:${NC}"
    echo "   curl http://localhost:8000/health"
    echo ""
    echo "4. ${CYAN}Acessar o dashboard:${NC}"
    echo "   - API: http://localhost:8000"
    echo "   - Docs: http://localhost:8000/docs"
    echo "   - Grafana: http://localhost:3000"
    echo ""
    echo "📚 Documentação completa: README.md"
    echo ""
    echo -e "${PURPLE}Happy Trading! 🚀${NC}"
}

# ================================
# FUNÇÃO PRINCIPAL
# ================================

main() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║              NEURAL CRYPTO BOT 2.0 SETUP                ║"
    echo "║          Advanced Trading System Setup Script           ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""

    log_info "Iniciando setup no $OS_TYPE..."
    echo ""

    # Execução sequencial das etapas
    check_system_resources || { log_error "Verificação de recursos falhou"; exit 1; }
    check_system_dependencies || { log_error "Dependências não satisfeitas"; exit 1; }
    setup_poetry || { log_error "Falha na configuração do Poetry"; exit 1; }
    install_dependencies || { log_error "Falha na instalação de dependências"; exit 1; }
    setup_environment_files || { log_error "Falha na configuração de ambiente"; exit 1; }
    setup_docker || { log_error "Falha na configuração do Docker"; exit 1; }
    setup_pre_commit || log_warning "Pre-commit não configurado"
    run_basic_tests || log_warning "Alguns testes básicos falharam"
    verify_setup || { log_error "Verificação final falhou"; exit 1; }

    show_next_steps
}

# ================================
# EXECUÇÃO
# ================================

# Verificar se está no diretório correto
if [[ ! -f "pyproject.toml" ]]; then
    log_error "Este script deve ser executado no diretório raiz do projeto"
    log_info "Navegue até o diretório que contém pyproject.toml"
    exit 1
fi

# Executar função principal
main "$@"