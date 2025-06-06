#!/bin/bash
# Neural Crypto Bot 2.0 - Instalador do Wrapper Multiplataforma
# Este script cria o wrapper Python caso ele não exista

set -e

# Definir cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Diretórios
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
WRAPPER_PATH="$SCRIPT_DIR/cross_platform_wrapper.py"

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

# Banner
echo -e "${CYAN}${BOLD}"
echo "=================================================="
echo "    NEURAL CRYPTO BOT 2.0 - WRAPPER INSTALLER     "
echo "=================================================="
echo -e "${NC}\n"

# Verificar Python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    log_error "Python não encontrado. Por favor, instale Python 3.11 ou superior."
    exit 1
fi

log_success "Python encontrado: $($PYTHON_CMD --version)"

# Verificar se o wrapper já existe
if [[ -f "$WRAPPER_PATH" ]]; then
    log_info "O wrapper multiplataforma já existe em: $WRAPPER_PATH"
    
    # Perguntar se quer recriar
    read -p "Deseja recriar o wrapper? (s/N): " RECREATE
    if [[ "$RECREATE" != "s" && "$RECREATE" != "S" ]]; then
        log_info "Mantendo o wrapper existente"
        exit 0
    fi
    
    log_info "Recriando o wrapper..."
fi

# Criar o wrapper
log_info "Criando wrapper multiplataforma em: $WRAPPER_PATH"

cat > "$WRAPPER_PATH" << 'EOF'
#!/usr/bin/env python3
"""
Neural Crypto Bot 2.0 - Cross-Platform Documentation System Wrapper
Este script permite executar os scripts de documentação em qualquer sistema operacional.
Ele detecta o sistema operacional e usa a abordagem apropriada para cada um.
"""

import os
import sys
import platform
import subprocess
import shutil
import argparse
from pathlib import Path

# Configurações globais
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.absolute()
DOCS_DIR = PROJECT_ROOT / "docs"
LOG_DIR = PROJECT_ROOT / ".docs-setup-logs"
CONFIG_FILE = PROJECT_ROOT / ".docs-config"

# Cores para saída de terminal (suporte a Windows e ANSI)
class Colors:
    RESET = "\033[0m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[0;37m"
    BOLD = "\033[1m"

# Desabilitar cores em Windows se não houver suporte ANSI
if platform.system() == "Windows":
    # Tentar habilitar o suporte ANSI para Windows 10+
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        # Se falhar, desativar as cores
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, "")

def print_banner():
    """Exibe o banner principal do sistema"""
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                        NEURAL CRYPTO BOT 2.0                                ║")
    print("║                     DOCUMENTATION SYSTEM SETUP                              ║")
    print("║                                                                              ║")
    print("║                    🚀 Enterprise-Grade Documentation                        ║")
    print("║                         powered by MkDocs Material                          ║")
    print("║                                                                              ║")
    print("║                     Desenvolvido por Oasis Systems                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")

def log_info(message):
    """Função para exibir mensagens de informação"""
    print(f"{Colors.BLUE}{Colors.BOLD}[INFO]{Colors.RESET} ℹ️ {message}")
    
def log_success(message):
    """Função para exibir mensagens de sucesso"""
    print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.RESET} ✅ {message}")
    
def log_warning(message):
    """Função para exibir mensagens de aviso"""
    print(f"{Colors.YELLOW}{Colors.BOLD}[WARNING]{Colors.RESET} ⚠️ {message}")
    
def log_error(message):
    """Função para exibir mensagens de erro"""
    print(f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} ❌ {message}")
    
def log_step(message):
    """Função para exibir etapas principais"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 === {message} ==={Colors.RESET}")

def ensure_directory(path):
    """Garante que um diretório exista"""
    os.makedirs(path, exist_ok=True)
    return path

def detect_os():
    """Detecta o sistema operacional"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system

def detect_shell():
    """Detecta o shell disponível para execução de scripts bash"""
    if platform.system() == "Windows":
        # Verificar se WSL está disponível
        try:
            subprocess.run(["wsl", "--list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "wsl"
        except:
            # Verificar se Git Bash está instalado
            git_bash = "C:\\Program Files\\Git\\bin\\bash.exe"
            if os.path.exists(git_bash):
                return git_bash
            
            # Verificar se há outro bash disponível
            try:
                subprocess.run(["bash", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return "bash"
            except:
                return None
    else:
        # Linux ou macOS
        return "bash"

def run_bash_script(script_path, args=None):
    """Executa um script bash usando o método apropriado para o sistema operacional"""
    log_info(f"Executando script: {script_path}")
    
    if not os.path.exists(script_path):
        log_error(f"Script não encontrado: {script_path}")
        return False
    
    # Garantir que o script tenha permissão de execução em sistemas Unix
    if platform.system() != "Windows":
        os.chmod(script_path, 0o755)
    
    shell_type = detect_shell()
    cmd = []
    
    if shell_type == "wsl":
        # Converter o caminho para formato WSL
        wsl_path = script_path.replace("\\", "/")
        if wsl_path[1] == ":":  # É um caminho Windows com letra de unidade
            drive = wsl_path[0].lower()
            wsl_path = f"/mnt/{drive}{wsl_path[2:]}"
        cmd = ["wsl", "bash", wsl_path]
    elif shell_type and shell_type.endswith("bash.exe"):
        # Git Bash no Windows
        cmd = [shell_type, "-c", script_path]
    elif shell_type == "bash":
        # Bash padrão
        cmd = ["bash", script_path]
    else:
        log_error("Não foi possível encontrar um shell compatível para executar scripts bash")
        log_info("Por favor, instale o WSL ou Git Bash no Windows")
        return False
    
    # Adicionar argumentos, se houver
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log_error(f"Erro ao executar script: {e}")
        return False

def setup_environment():
    """Configura o ambiente necessário para execução dos scripts"""
    log_step("Configurando ambiente")
    
    # Criar diretórios necessários
    ensure_directory(LOG_DIR)
    ensure_directory(DOCS_DIR)
    
    # Verificar requisitos básicos
    os_type = detect_os()
    log_info(f"Sistema operacional detectado: {os_type}")
    
    shell = detect_shell()
    if shell:
        log_success(f"Shell compatível encontrado: {shell}")
    else:
        log_error("Nenhum shell compatível encontrado")
        log_info("No Windows, instale WSL ou Git Bash para usar este sistema")
        return False
    
    # Verificar Python
    try:
        python_version = subprocess.check_output(["python3", "--version"], universal_newlines=True).strip()
        log_success(f"Python encontrado: {python_version}")
    except:
        try:
            python_version = subprocess.check_output(["python", "--version"], universal_newlines=True).strip()
            log_success(f"Python encontrado: {python_version}")
        except:
            log_error("Python 3.11+ é necessário para usar este sistema")
            return False
    
    return True

def run_documentation_script(script_name, args=None):
    """Executa um script específico do sistema de documentação"""
    script_path = SCRIPT_DIR / script_name
    
    if not os.path.exists(script_path):
        log_error(f"Script não encontrado: {script_name}")
        return False
    
    return run_bash_script(script_path, args)

def list_available_scripts():
    """Lista todos os scripts disponíveis no diretório de scripts"""
    scripts = []
    for item in os.listdir(SCRIPT_DIR):
        if item.endswith('.sh') and os.path.isfile(os.path.join(SCRIPT_DIR, item)):
            scripts.append(item)
    
    return sorted(scripts)

def show_menu():
    """Exibe o menu interativo para seleção de scripts"""
    print_banner()
    
    print(f"{Colors.CYAN}{Colors.BOLD}🌍 Neural Crypto Bot 2.0 - Sistema de Documentação Multiplataforma{Colors.RESET}\n")
    print(f"{Colors.WHITE}Escolha uma opção:{Colors.RESET}\n")
    
    print(f"{Colors.WHITE}1. {Colors.BOLD}Configuração Completa{Colors.RESET} - Setup completo do sistema de documentação")
    print(f"{Colors.WHITE}2. {Colors.BOLD}Verificar Pré-requisitos{Colors.RESET} - Verificar se o sistema atende aos requisitos")
    print(f"{Colors.WHITE}3. {Colors.BOLD}Criar Estrutura de Diretórios{Colors.RESET} - Criar a estrutura de pastas")
    print(f"{Colors.WHITE}4. {Colors.BOLD}Instalar Dependências{Colors.RESET} - Instalar pacotes necessários")
    print(f"{Colors.WHITE}5. {Colors.BOLD}Criar Configuração{Colors.RESET} - Configurar MkDocs")
    print(f"{Colors.WHITE}6. {Colors.BOLD}Publicar Documentação{Colors.RESET} - Build e deploy da documentação")
    print(f"{Colors.WHITE}7. {Colors.BOLD}Executar Script Específico{Colors.RESET} - Escolher um script para executar")
    print(f"{Colors.WHITE}8. {Colors.BOLD}Sair{Colors.RESET}")
    
    choice = input(f"\n{Colors.CYAN}Escolha uma opção [1-8]: {Colors.RESET}")
    
    return choice

def handle_menu_choice(choice):
    """Trata a escolha do usuário no menu"""
    if choice == "1":
        log_step("Iniciando configuração completa")
        return run_documentation_script("setup_documentation.sh")
    elif choice == "2":
        return run_documentation_script("check_prerequisites.sh")
    elif choice == "3":
        return run_documentation_script("create_directories.sh")
    elif choice == "4":
        return run_documentation_script("install_dependencies.sh")
    elif choice == "5":
        return run_documentation_script("create_configuration.sh")
    elif choice == "6":
        return run_documentation_script("publish_docs.sh")
    elif choice == "7":
        scripts = list_available_scripts()
        if not scripts:
            log_error("Nenhum script encontrado no diretório de scripts")
            return False
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}Scripts disponíveis:{Colors.RESET}\n")
        for i, script in enumerate(scripts, 1):
            print(f"{Colors.WHITE}{i}. {script}{Colors.RESET}")
        
        try:
            script_idx = int(input(f"\n{Colors.CYAN}Escolha um script [1-{len(scripts)}]: {Colors.RESET}")) - 1
            if 0 <= script_idx < len(scripts):
                return run_documentation_script(scripts[script_idx])
            else:
                log_error("Opção inválida")
                return False
        except ValueError:
            log_error("Entrada inválida. Por favor, digite um número.")
            return False
    elif choice == "8":
        log_info("Saindo do sistema de documentação")
        sys.exit(0)
    else:
        log_error("Opção inválida. Por favor, escolha entre 1 e 8.")
        return False

def parse_arguments():
    """Processa argumentos de linha de comando"""
    parser = argparse.ArgumentParser(description='Neural Crypto Bot 2.0 - Sistema de Documentação Multiplataforma')
    parser.add_argument('script', nargs='?', help='Script específico para executar')
    parser.add_argument('--args', nargs='+', help='Argumentos para passar ao script')
    return parser.parse_args()

def main():
    """Função principal"""
    args = parse_arguments()
    
    if not setup_environment():
        log_error("Falha na configuração do ambiente")
        return 1
    
    # Se um script específico foi passado como argumento
    if args.script:
        script_path = args.script
        if not script_path.endswith('.sh'):
            script_path += '.sh'
        
        # Verificar se o caminho é absoluto ou relativo
        if not os.path.isabs(script_path):
            script_path = os.path.join(SCRIPT_DIR, script_path)
        
        return 0 if run_bash_script(script_path, args.args) else 1
    
    # Modo interativo
    while True:
        choice = show_menu()
        result = handle_menu_choice(choice)
        
        if choice != "8":  # Se não escolheu sair
            input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.RESET}")

if __name__ == "__main__":
    sys.exit(main())
EOF

# Tornar o wrapper executável
chmod +x "$WRAPPER_PATH"

log_success "Wrapper multiplataforma criado com sucesso em: $WRAPPER_PATH"
log_info "Agora você pode executar o wrapper com: $PYTHON_CMD $WRAPPER_PATH"

# Verificar se o Python pode executar o script
if $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 6) else 1)"; then
    log_info "Testando o wrapper..."
    $PYTHON_CMD "$WRAPPER_PATH" --help >/dev/null 2>&1
    
    if [[ $? -eq 0 ]]; then
        log_success "Wrapper testado com sucesso!"
        echo ""
        echo -e "${GREEN}${BOLD}O wrapper multiplataforma está pronto para uso!${NC}"
        echo -e "${CYAN}Você pode executar o sistema de documentação em qualquer plataforma usando:${NC}"
        echo -e "${BOLD}$PYTHON_CMD $WRAPPER_PATH${NC}"
    else
        log_warning "O wrapper foi criado, mas o teste falhou. Verifique o código ou execute manualmente."
    fi
else
    log_warning "A versão do Python é muito antiga. O wrapper pode não funcionar corretamente."
    log_info "Recomendamos Python 3.6 ou superior."
fi