#!/bin/bash
# start_docker.sh - Script para iniciar o ambiente Docker

set -e

echo "=== Iniciando ambiente Docker para o Neural Crypto Bot ==="

# Detectar sistema operacional
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    MINGW*|MSYS*|CYGWIN*)    OS_TYPE=Windows;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac

echo "Sistema operacional detectado: $OS_TYPE"

# Função para verificar programas no Windows
check_windows_program() {
    where.exe $1 > /dev/null 2>&1
    return $?
}

# Verifica se o Docker está instalado
if [ "$OS_TYPE" = "Windows" ]; then
    if ! check_windows_program docker; then
        echo "Docker não está instalado. Por favor, instale-o antes de continuar."
        echo "Visite https://docs.docker.com/desktop/install/windows-install/"
        exit 1
    fi
else
    # Linux ou Mac
    if ! command -v docker &> /dev/null; then
        echo "Docker não está instalado. Por favor, instale-o antes de continuar."
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Visite https://docs.docker.com/engine/install/"
        elif [ "$OS_TYPE" = "Mac" ]; then
            echo "Visite https://docs.docker.com/desktop/install/mac-install/"
        fi
        exit 1
    fi
fi

# Verifica se o Docker Compose está instalado
if [ "$OS_TYPE" = "Windows" ]; then
    if ! (check_windows_program docker-compose || check_windows_program "docker" "compose"); then
        echo "Docker Compose não está instalado. Por favor, instale o Docker Desktop que inclui o Docker Compose."
        exit 1
    fi
else
    # Linux ou Mac
    if ! command -v docker-compose &> /dev/null && ! (command -v docker &> /dev/null && docker compose version &> /dev/null); then
        echo "Docker Compose não está instalado. Por favor, instale-o antes de continuar."
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Visite https://docs.docker.com/compose/install/linux/"
        elif [ "$OS_TYPE" = "Mac" ]; then
            echo "Atualize o Docker Desktop para uma versão mais recente que inclui o Docker Compose"
        fi
        exit 1
    fi
fi

# Verifica se o arquivo .env existe
if [ ! -f .env ]; then
    echo "Arquivo .env não encontrado. Criando a partir do .env.example..."
    cp .env.example .env
    echo "Por favor, edite o arquivo .env com suas configurações antes de continuar."
    
    # Se estiver no Windows, abra o arquivo com o Notepad
    if [ "$OS_TYPE" = "Windows" ]; then
        notepad.exe .env
    fi
    
    read -p "Pressione Enter para continuar após configurar o arquivo .env ou Ctrl+C para cancelar..." dummy
fi

# Cria diretório de logs se não existir
mkdir -p logs

# Função para executar docker-compose ou docker compose
run_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        docker-compose $@
    else
        docker compose $@
    fi
}

# Inicia os serviços de infraestrutura
echo "Iniciando serviços de infraestrutura (PostgreSQL, Redis, Zookeeper, Kafka)..."
run_docker_compose up -d postgres redis zookeeper kafka

echo "Aguardando a inicialização dos serviços de infraestrutura..."
sleep 15

# Verifica se os serviços de infraestrutura estão saudáveis
echo "Verificando a saúde dos serviços de infraestrutura..."

# Função para verificar a saúde de um serviço
check_service_health() {
    SERVICE=$1
    if run_docker_compose ps $SERVICE | grep -q "Up"; then
        echo "✅ $SERVICE está em execução"
        return 0
    else
        echo "❌ $SERVICE não está em execução corretamente"
        return 1
    fi
}

# Verifica cada serviço
INFRA_HEALTHY=true
if ! check_service_health postgres; then INFRA_HEALTHY=false; fi
if ! check_service_health redis; then INFRA_HEALTHY=false; fi
if ! check_service_health zookeeper; then INFRA_HEALTHY=false; fi
if ! check_service_health kafka; then INFRA_HEALTHY=false; fi

if [ "$INFRA_HEALTHY" = false ]; then
    echo "⚠️ Alguns serviços de infraestrutura não estão saudáveis. Verificando logs..."
    run_docker_compose logs postgres redis zookeeper kafka
    echo "Por favor, resolva os problemas antes de continuar."
    exit 1
fi

# Inicia os serviços da aplicação
echo "Iniciando serviços da aplicação..."
run_docker_compose up -d collector execution training api

# Se o sistema tiver Grafana/Prometheus, inicie-os também
if grep -q "grafana" docker-compose.yml; then
    echo "Iniciando serviços de monitoramento (Prometheus, Grafana)..."
    run_docker_compose up -d prometheus grafana
fi

echo "Aguardando a inicialização dos serviços da aplicação..."
sleep 10

# Verifica se os serviços da aplicação estão em execução
echo "Verificando a saúde dos serviços da aplicação..."
APP_HEALTHY=true
if ! check_service_health collector; then APP_HEALTHY=false; fi
if ! check_service_health execution; then APP_HEALTHY=false; fi
if ! check_service_health training; then APP_HEALTHY=false; fi
if ! check_service_health api; then APP_HEALTHY=false; fi

if [ "$APP_HEALTHY" = false ]; then
    echo "⚠️ Alguns serviços da aplicação não estão saudáveis. Verificando logs..."
    run_docker_compose logs collector execution training api
    echo "Por favor, resolva os problemas antes de continuar."
    exit 1
fi

echo "=== Ambiente Docker iniciado com sucesso! ==="

# Exibe informações importantes
echo ""
echo "Informações importantes:"
echo "- API disponível em: http://localhost:8000"
if grep -q "grafana" docker-compose.yml; then
    echo "- Dashboard Grafana: http://localhost:3000 (usuário: admin, senha: neuralbot)"
fi
if grep -q "prometheus" docker-compose.yml; then
    echo "- Prometheus: http://localhost:9090"
fi
echo ""
echo "Comandos úteis:"
echo "- Para visualizar os logs: docker-compose logs -f"
echo "- Para parar os serviços: docker-compose down"
echo "- Para reiniciar um serviço específico: docker-compose restart <serviço>"
echo ""
echo "Enjoy trading!"