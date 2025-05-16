#!/bin/bash
# setup_scripts.sh

echo "=== Configurando scripts utilitários para o Neural Crypto Bot ==="

# Detectar sistema operacional
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    MINGW*|MSYS*|CYGWIN*)    OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Sistema operacional detectado: $OS_TYPE"

mkdir -p scripts

# Cria script para iniciar o ambiente virtual no PowerShell (Windows)
if [ "$OS_TYPE" = "Windows" ]; then
    echo "Criando activate.ps1 para Windows..."
    cat > activate.ps1 << EOF
# Script para ativar o ambiente virtual no PowerShell
Write-Host "=== Ativando ambiente virtual para o Neural Crypto Bot ===" -ForegroundColor Green

if (Test-Path .venv/Scripts/Activate.ps1) {
    & .venv/Scripts/Activate.ps1
} else {
    Write-Host "Ambiente virtual não encontrado. Executando setup_poetry.sh primeiro..." -ForegroundColor Yellow
    bash ./scripts/setup_poetry.sh
    if (Test-Path .venv/Scripts/Activate.ps1) {
        & .venv/Scripts/Activate.ps1
    } else {
        Write-Host "Falha ao criar ambiente virtual. Por favor, verifique a instalação." -ForegroundColor Red
    }
}
EOF
fi

# Cria script para Windows que inicia o ambiente Docker usando PowerShell
if [ "$OS_TYPE" = "Windows" ]; then
    echo "Criando start_docker.ps1 para Windows..."
    cat > start_docker.ps1 << EOF
# Script para iniciar o ambiente Docker no PowerShell
Write-Host "=== Iniciando ambiente Docker para o Neural Crypto Bot ===" -ForegroundColor Green

# Verifica se o Docker está instalado
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker não está instalado. Por favor, instale-o antes de continuar." -ForegroundColor Red
    Write-Host "Visite https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}

# Verifica se o arquivo .env existe
if (-not (Test-Path .env)) {
    Write-Host "Arquivo .env não encontrado. Criando a partir do .env.example..." -ForegroundColor Yellow
    Copy-Item .env.example .env
    Write-Host "Por favor, edite o arquivo .env com suas configurações antes de continuar." -ForegroundColor Yellow
    notepad.exe .env
    Read-Host "Pressione Enter para continuar após configurar o arquivo .env ou Ctrl+C para cancelar..."
}

# Cria diretório de logs se não existir
if (-not (Test-Path logs)) {
    New-Item -Path logs -ItemType Directory | Out-Null
}

# Inicia os serviços de infraestrutura
Write-Host "Iniciando serviços de infraestrutura (PostgreSQL, Redis, Zookeeper, Kafka)..." -ForegroundColor Cyan
docker-compose up -d postgres redis zookeeper kafka

Write-Host "Aguardando a inicialização dos serviços de infraestrutura..." -ForegroundColor Cyan
Start-Sleep -Seconds 15

# Inicia os serviços da aplicação
Write-Host "Iniciando serviços da aplicação..." -ForegroundColor Cyan
docker-compose up -d collector execution training api

# Se o sistema tiver Grafana/Prometheus, inicie-os também
if (Select-String -Path docker-compose.yml -Pattern "grafana") {
    Write-Host "Iniciando serviços de monitoramento (Prometheus, Grafana)..." -ForegroundColor Cyan
    docker-compose up -d prometheus grafana
}

Write-Host "=== Ambiente Docker iniciado com sucesso! ===" -ForegroundColor Green

# Exibe informações importantes
Write-Host ""
Write-Host "Informações importantes:" -ForegroundColor Green
Write-Host "- API disponível em: http://localhost:8000" -ForegroundColor White
if (Select-String -Path docker-compose.yml -Pattern "grafana") {
    Write-Host "- Dashboard Grafana: http://localhost:3000 (usuário: admin, senha: neuralbot)" -ForegroundColor White
}
if (Select-String -Path docker-compose.yml -Pattern "prometheus") {
    Write-Host "- Prometheus: http://localhost:9090" -ForegroundColor White
}
Write-Host ""
Write-Host "Comandos úteis:" -ForegroundColor Green
Write-Host "- Para visualizar os logs: docker-compose logs -f" -ForegroundColor White
Write-Host "- Para parar os serviços: docker-compose down" -ForegroundColor White
Write-Host "- Para reiniciar um serviço específico: docker-compose restart <serviço>" -ForegroundColor White
Write-Host ""
Write-Host "Enjoy trading!" -ForegroundColor Green
EOF
fi

# Cria script de utilidades para desenvolvimento
echo "Criando scripts/dev_utils.sh..."
cat > scripts/dev_utils.sh << EOF
#!/bin/bash
# dev_utils.sh - Utilitários para desenvolvimento

# Detectar sistema operacional
OS="\$(uname -s)"
case "\${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    MINGW*|MSYS*|CYGWIN*)    OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:\${OS}"
esac

# Configuração de cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para executar docker-compose ou docker compose
run_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        docker-compose \$@
    else
        docker compose \$@
    fi
}

case "\$1" in
    "init")
        echo -e "\${GREEN}Inicializando ambiente de desenvolvimento do Neural Crypto Bot...\${NC}"
        ./scripts/check_prerequisites.sh && ./scripts/setup_poetry.sh
        ;;
    "test")
        echo -e "\${GREEN}Executando testes...\${NC}"
        poetry run pytest -v tests/
        ;;
    "lint")
        echo -e "\${GREEN}Verificando estilo de código...\${NC}"
        poetry run black src tests
        poetry run isort src tests
        poetry run ruff src tests
        ;;
    "build")
        echo -e "\${GREEN}Construindo serviços Docker...\${NC}"
        run_docker_compose build
        ;;
    "up")
        echo -e "\${GREEN}Iniciando serviços...\${NC}"
        ./scripts/start_docker.sh
        ;;
    "down")
        echo -e "\${GREEN}Parando serviços...\${NC}"
        run_docker_compose down
        ;;
    "logs")
        echo -e "\${GREEN}Exibindo logs...\${NC}"
        run_docker_compose logs -f
        ;;
    "restart")
        if [ -z "\$2" ]; then
            echo -e "\${RED}Erro: Especifique o serviço para reiniciar\${NC}"
            echo -e "Uso: \$0 restart <serviço>"
            exit 1
        fi
        echo -e "\${GREEN}Reiniciando serviço \$2...\${NC}"
        run_docker_compose restart \$2
        ;;
    "clean")
        echo -e "\${YELLOW}Atenção: Esta operação removerá todos os volumes, containers e networks.\${NC}"
        read -p "Deseja continuar? (s/n) " -n 1 -r
        echo
        if [[ \$REPLY =~ ^[Ss]$ ]]; then
            echo -e "\${GREEN}Limpando ambiente Docker...\${NC}"
            run_docker_compose down -v
            docker system prune -f
            find . -name "__pycache__" -type d -exec rm -rf {} +
            find . -name "*.pyc" -delete
        fi
        ;;
    "shell")
        if [ -z "\$2" ]; then
            echo -e "\${RED}Erro: Especifique o serviço para acessar o shell\${NC}"
            echo -e "Uso: \$0 shell <serviço>"
            exit 1
        fi
        echo -e "\${GREEN}Acessando shell do serviço \$2...\${NC}"
        run_docker_compose exec \$2 /bin/bash
        ;;
    "db-backup")
        echo -e "\${GREEN}Realizando backup do banco de dados...\${NC}"
        BACKUP_DIR="./backups"
        mkdir -p \$BACKUP_DIR
        BACKUP_FILE="\$BACKUP_DIR/neuralcryptobot_backup_\$(date +%Y%m%d_%H%M%S).sql"
        run_docker_compose exec postgres pg_dump -U neuralbot neuralcryptobot > \$BACKUP_FILE
        echo -e "\${GREEN}Backup salvo em \$BACKUP_FILE\${NC}"
        ;;
    "db-restore")
        if [ -z "\$2" ]; then
            echo -e "\${RED}Erro: Especifique o arquivo de backup para restaurar\${NC}"
            echo -e "Uso: \$0 db-restore <arquivo_backup>"
            exit 1
        fi
        if [ ! -f "\$2" ]; then
            echo -e "\${RED}Erro: Arquivo de backup não encontrado\${NC}"
            exit 1
        fi
        echo -e "\${YELLOW}Atenção: Esta operação substituirá o banco de dados atual.\${NC}"
        read -p "Deseja continuar? (s/n) " -n 1 -r
        echo
        if [[ \$REPLY =~ ^[Ss]$ ]]; then
            echo -e "\${GREEN}Restaurando backup do banco de dados...\${NC}"
            cat \$2 | run_docker_compose exec -T postgres psql -U neuralbot neuralcryptobot
        fi
        ;;
    "help"|*)
        echo -e "\${GREEN}Neural Crypto Bot - Utilitários de Desenvolvimento\${NC}"
        echo -e "\${BLUE}Uso: \$0 <comando>\${NC}"
        echo
        echo -e "\${YELLOW}Comandos disponíveis:\${NC}"
        echo -e "  init        - Inicializa o ambiente de desenvolvimento"
        echo -e "  test        - Executa testes automatizados"
        echo -e "  lint        - Verifica o estilo do código"
        echo -e "  build       - Constrói os serviços Docker"
        echo -e "  up          - Inicia todos os serviços"
        echo -e "  down        - Para todos os serviços"
        echo -e "  logs        - Exibe logs de todos os serviços"
        echo -e "  restart     - Reinicia um serviço específico"
        echo -e "  clean       - Remove containers, volumes e arquivos temporários"
        echo -e "  shell       - Acessa o shell de um serviço"
        echo -e "  db-backup   - Realiza backup do banco de dados"
        echo -e "  db-restore  - Restaura backup do banco de dados"
        echo -e "  help        - Exibe esta mensagem de ajuda"
        ;;
esac
EOF

# Torna os novos scripts executáveis
chmod +x scripts/dev_utils.sh

echo "✅ Scripts utilitários configurados com sucesso!"