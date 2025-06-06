#!/usr/bin/env python3
"""
Neural Crypto Bot 2.0 - Inicializador Multiplataforma da Documentação
Este script é um ponto de entrada universal para o sistema de documentação,
funcionando em qualquer plataforma (Windows, macOS, Linux).

Ele detecta o sistema operacional e executa o método mais apropriado.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path

# Configurações globais
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
DOCS_SCRIPTS_DIR = PROJECT_ROOT / "scripts" / "docs"

# Cores para terminal
class Colors:
    RESET = "\033[0m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"

# Desativar cores se não houver suporte ANSI no Windows
if platform.system() == "Windows":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, "")

def print_banner():
    """Exibe o banner do sistema"""
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
    """Exibe mensagem de informação"""
    print(f"{Colors.BLUE}{Colors.BOLD}[INFO]{Colors.RESET} ℹ️ {message}")

def log_success(message):
    """Exibe mensagem de sucesso"""
    print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.RESET} ✅ {message}")

def log_warning(message):
    """Exibe mensagem de aviso"""
    print(f"{Colors.YELLOW}{Colors.BOLD}[WARNING]{Colors.RESET} ⚠️ {message}")

def log_error(message):
    """Exibe mensagem de erro"""
    print(f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} ❌ {message}")

def detect_os():
    """Detecta o sistema operacional"""
    system = platform.system().lower()
    if system == "darwin":
        return "macos"
    return system

def check_prerequisites():
    """Verifica requisitos básicos do sistema"""
    log_info(f"Sistema operacional detectado: {detect_os()}")
    
    # Verificar Python
    python_version = platform.python_version()
    log_info(f"Python versão: {python_version}")
    
    # Verificar se o diretório de scripts existe
    if not DOCS_SCRIPTS_DIR.exists():
        log_error(f"Diretório de scripts não encontrado: {DOCS_SCRIPTS_DIR}")
        log_error("Certifique-se de estar executando este script do diretório correto")
        return False
    
    # Verificar permissões de escrita
    try:
        test_file = PROJECT_ROOT / ".docs-test-file"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        log_error(f"Sem permissão de escrita no diretório do projeto: {e}")
        return False
    
    return True

def run_windows_setup():
    """Executa a configuração no Windows"""
    log_info("Iniciando setup para Windows...")
    
    # Tentar diferentes métodos em ordem de preferência
    
    # 1. PowerShell
    powershell_script = DOCS_SCRIPTS_DIR / "setup_docs.ps1"
    if powershell_script.exists():
        try:
            log_info("Tentando executar via PowerShell...")
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(powershell_script)],
                check=False
            )
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar via PowerShell")
        except Exception as e:
            log_warning(f"Erro ao executar PowerShell: {e}")
    
    # 2. Batch
    batch_script = DOCS_SCRIPTS_DIR / "install_docs.bat"
    if batch_script.exists():
        try:
            log_info("Tentando executar via Batch...")
            result = subprocess.run([str(batch_script)], check=False, shell=True)
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar via Batch")
        except Exception as e:
            log_warning(f"Erro ao executar Batch: {e}")
    
    # 3. Python wrapper
    wrapper_script = DOCS_SCRIPTS_DIR / "cross_platform_wrapper.py"
    if wrapper_script.exists():
        try:
            log_info("Tentando executar via Python wrapper...")
            result = subprocess.run([sys.executable, str(wrapper_script)], check=False)
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar via Python wrapper")
        except Exception as e:
            log_warning(f"Erro ao executar Python wrapper: {e}")
    
    # 4. WSL (se disponível)
    try:
        # Verificar se WSL está disponível
        wsl_check = subprocess.run(["wsl", "--list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if wsl_check.returncode == 0:
            bash_script = DOCS_SCRIPTS_DIR / "setup_documentation.sh"
            if bash_script.exists():
                log_info("Tentando executar via WSL...")
                # Converter caminho para formato WSL
                wsl_path = str(bash_script).replace("\\", "/")
                if wsl_path[1] == ":":  # É um caminho Windows com letra de unidade
                    drive = wsl_path[0].lower()
                    wsl_path = f"/mnt/{drive}{wsl_path[2:]}"
                
                result = subprocess.run(["wsl", "bash", wsl_path], check=False)
                if result.returncode == 0:
                    return True
                log_warning("Falha ao executar via WSL")
    except Exception as e:
        log_warning(f"WSL não disponível: {e}")
    
    log_error("Todos os métodos de execução falharam")
    return False

def run_unix_setup():
    """Executa a configuração em sistemas Unix (macOS/Linux)"""
    log_info(f"Iniciando setup para {detect_os()}...")
    
    # Tentar diferentes métodos em ordem de preferência
    
    # 1. Script de instalação direto
    install_script = DOCS_SCRIPTS_DIR / "install_docs.sh"
    if install_script.exists():
        try:
            os.chmod(install_script, 0o755)  # Garantir permissão de execução
            log_info("Tentando executar script de instalação...")
            result = subprocess.run([str(install_script)], check=False)
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar script de instalação")
        except Exception as e:
            log_warning(f"Erro ao executar script de instalação: {e}")
    
    # 2. Makefile (se existir na raiz do projeto)
    makefile = PROJECT_ROOT / "Makefile"
    if makefile.exists():
        try:
            log_info("Tentando executar via Makefile...")
            result = subprocess.run(["make", "setup"], cwd=PROJECT_ROOT, check=False)
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar via Makefile")
        except Exception as e:
            log_warning(f"Erro ao executar Makefile: {e}")
    
    # 3. Script de setup diretamente
    setup_script = DOCS_SCRIPTS_DIR / "setup_documentation.sh"
    if setup_script.exists():
        try:
            os.chmod(setup_script, 0o755)  # Garantir permissão de execução
            log_info("Tentando executar script de setup diretamente...")
            result = subprocess.run([str(setup_script)], check=False)
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar script de setup")
        except Exception as e:
            log_warning(f"Erro ao executar script de setup: {e}")
    
    # 4. Python wrapper
    wrapper_script = DOCS_SCRIPTS_DIR / "cross_platform_wrapper.py"
    if wrapper_script.exists():
        try:
            os.chmod(wrapper_script, 0o755)  # Garantir permissão de execução
            log_info("Tentando executar via Python wrapper...")
            result = subprocess.run([sys.executable, str(wrapper_script)], check=False)
            if result.returncode == 0:
                return True
            log_warning("Falha ao executar via Python wrapper")
        except Exception as e:
            log_warning(f"Erro ao executar Python wrapper: {e}")
    
    log_error("Todos os métodos de execução falharam")
    return False

def create_wrapper_if_needed():
    """Cria o wrapper Python multiplataforma se ele não existir"""
    wrapper_path = DOCS_SCRIPTS_DIR / "cross_platform_wrapper.py"
    if wrapper_path.exists():
        return True
    
    # O wrapper não existe, verificar se há script para criá-lo
    installer_script = DOCS_SCRIPTS_DIR / "install_wrapper.sh"
    if installer_script.exists():
        try:
            log_info("Instalando wrapper multiplataforma...")
            if platform.system() == "Windows":
                # No Windows, tentar usar WSL ou Python para criar
                try:
                    subprocess.run(["wsl", "bash", str(installer_script)], check=False)
                except:
                    # Criar wrapper manualmente via Python
                    log_info("Criando wrapper manualmente...")
                    # O conteúdo do wrapper seria adicionado aqui
            else:
                # Em sistemas Unix, executar o script diretamente
                os.chmod(installer_script, 0o755)
                subprocess.run([str(installer_script)], check=False)
            
            return wrapper_path.exists()
        except Exception as e:
            log_warning(f"Erro ao criar wrapper: {e}")
            return False
    
    log_warning("Script de instalação do wrapper não encontrado")
    return False

def main():
    """Função principal"""
    print_banner()
    
    if not check_prerequisites():
        log_error("Verificação de pré-requisitos falhou")
        return 1
    
    os_type = detect_os()
    create_wrapper_if_needed()
    
    if os_type == "windows":
        success = run_windows_setup()
    else:  # macOS ou Linux
        success = run_unix_setup()
    
    if success:
        log_success("Setup da documentação concluído com sucesso!")
        return 0
    else:
        log_error("Falha no setup da documentação")
        log_info("Tente executar os scripts manualmente:")
        
        if os_type == "windows":
            log_info(f"  - PowerShell: {DOCS_SCRIPTS_DIR}\\setup_docs.ps1")
            log_info(f"  - CMD: {DOCS_SCRIPTS_DIR}\\install_docs.bat")
        else:
            log_info(f"  - Bash: {DOCS_SCRIPTS_DIR}/install_docs.sh")
            log_info(f"  - Make: cd {PROJECT_ROOT} && make setup")
        
        return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n")
        log_info("Operação cancelada pelo usuário")
        sys.exit(1)
    except Exception as e:
        log_error(f"Erro inesperado: {e}")
        sys.exit(1)