#!/bin/bash
# check_prerequisites.sh

echo "=== Verificando pré-requisitos para o Neural Crypto Bot ==="

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

# Verificar Python 3.11+
if [ "$OS_TYPE" = "Windows" ]; then
    if check_windows_program python; then
        PYTHON_VERSION=$(python --version 2>&1 | cut -d " " -f 2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            echo "✅ Python 3.11+ instalado: $PYTHON_VERSION"
        else
            echo "❌ Python 3.11+ é necessário. Versão detectada: $PYTHON_VERSION"
            echo "Por favor, instale o Python 3.11 ou superior: https://www.python.org/downloads/"
            exit 1
        fi
    else
        echo "❌ Python não encontrado"
        echo "Por favor, instale o Python 3.11 ou superior: https://www.python.org/downloads/"
        exit 1
    fi
else
    # Linux ou Mac
    if command -v python3.11 &> /dev/null; then
        echo "✅ Python 3.11+ instalado"
        python3.11 --version
    elif command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d " " -f 2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            echo "✅ Python 3.11+ instalado: $PYTHON_VERSION"
        else
            echo "❌ Python 3.11+ é necessário. Versão detectada: $PYTHON_VERSION"
            if [ "$OS_TYPE" = "Linux" ]; then
                echo "Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-dev python3.11-venv"
            elif [ "$OS_TYPE" = "Mac" ]; then
                echo "macOS: brew install python@3.11"
            fi
            exit 1
        fi
    else
        echo "❌ Python 3.11+ não encontrado"
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-dev python3.11-venv"
        elif [ "$OS_TYPE" = "Mac" ]; then
            echo "macOS: brew install python@3.11"
        fi
        exit 1
    fi
fi

# Verificar Docker
if [ "$OS_TYPE" = "Windows" ]; then
    if check_windows_program docker; then
        echo "✅ Docker instalado"
        docker --version
    else
        echo "❌ Docker não encontrado"
        echo "Por favor, instale o Docker Desktop para Windows: https://docs.docker.com/desktop/install/windows-install/"
        exit 1
    fi
else
    # Linux ou Mac
    if command -v docker &> /dev/null; then
        echo "✅ Docker instalado"
        docker --version
    else
        echo "❌ Docker não encontrado"
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Por favor, instale o Docker: https://docs.docker.com/engine/install/"
        elif [ "$OS_TYPE" = "Mac" ]; then
            echo "Por favor, instale o Docker Desktop: https://docs.docker.com/desktop/install/mac-install/"
        fi
        exit 1
    fi
fi

# Verificar Docker Compose
if [ "$OS_TYPE" = "Windows" ]; then
    if check_windows_program docker-compose || check_windows_program "docker" "compose"; then
        echo "✅ Docker Compose instalado"
        if check_windows_program docker-compose; then
            docker-compose --version
        else
            docker compose version
        fi
    else
        echo "❌ Docker Compose não encontrado"
        echo "Por favor, instale o Docker Desktop para Windows que já inclui o Docker Compose"
        exit 1
    fi
else
    # Linux ou Mac
    if command -v docker-compose &> /dev/null; then
        echo "✅ Docker Compose instalado"
        docker-compose --version
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        echo "✅ Docker Compose (plugin) instalado"
        docker compose version
    else
        echo "❌ Docker Compose não encontrado"
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Por favor, instale o Docker Compose: https://docs.docker.com/compose/install/linux/"
        elif [ "$OS_TYPE" = "Mac" ]; then
            echo "Por favor, atualize o Docker Desktop para uma versão mais recente que inclui o Docker Compose"
        fi
        exit 1
    fi
fi

# Verificar Git
if [ "$OS_TYPE" = "Windows" ]; then
    if check_windows_program git; then
        echo "✅ Git instalado"
        git --version
    else
        echo "❌ Git não encontrado"
        echo "Por favor, instale o Git para Windows: https://git-scm.com/download/win"
        exit 1
    fi
else
    # Linux ou Mac
    if command -v git &> /dev/null; then
        echo "✅ Git instalado"
        git --version
    else
        echo "❌ Git não encontrado"
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "Ubuntu/Debian: sudo apt update && sudo apt install git"
        elif [ "$OS_TYPE" = "Mac" ]; then
            echo "macOS: brew install git"
        fi
        exit 1
    fi
fi

echo "✅ Todos os pré-requisitos estão instalados!"