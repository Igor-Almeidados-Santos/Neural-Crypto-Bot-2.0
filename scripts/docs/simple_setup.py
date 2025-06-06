#!/usr/bin/env python3
"""
Neural Crypto Bot 2.0 - Setup Simplificado de Documentação
Este script fornece uma versão simplificada e robusta para configurar
a documentação do Neural Crypto Bot 2.0 em qualquer sistema operacional.
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
import time

# Configurações globais
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_FILE = PROJECT_ROOT / "mkdocs.yml"
LOG_DIR = PROJECT_ROOT / ".docs-setup-logs"

# Cores para terminal (com fallback para sistemas sem suporte ANSI)
class Colors:
    RESET = "\033[0m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"

# Desabilitar cores se não suportado
if platform.system() == "Windows":
    # Tentar habilitar cores no Windows 10+
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        # Fallback: desabilitar cores
        for attr in dir(Colors):
            if not attr.startswith('__'):
                setattr(Colors, attr, "")

# Funções de utilitário
def print_banner():
    """Exibe o banner do sistema de documentação"""
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("="*70)
    print(f"           NEURAL CRYPTO BOT 2.0 - DOCUMENTATION SETUP")
    print(f"                    SIMPLIFIED VERSION")
    print("="*70)
    print(f"{Colors.RESET}\n")

def log(level, message):
    """Função genérica de log"""
    levels = {
        "INFO": f"{Colors.BLUE}{Colors.BOLD}[INFO]{Colors.RESET} ℹ️ ",
        "SUCCESS": f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.RESET} ✅ ",
        "WARNING": f"{Colors.YELLOW}{Colors.BOLD}[WARNING]{Colors.RESET} ⚠️ ",
        "ERROR": f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} ❌ "
    }
    print(f"{levels.get(level, '')} {message}")
    
    # Também salvar em arquivo de log
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = LOG_DIR / f"setup-{time.strftime('%Y%m%d-%H%M%S')}.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{level}] {message}\n")

def log_info(message): log("INFO", message)
def log_success(message): log("SUCCESS", message)
def log_warning(message): log("WARNING", message)
def log_error(message): log("ERROR", message)

def log_step(message):
    """Exibe um passo importante do processo"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}=== {message} ==={Colors.RESET}")

def ensure_directory(path):
    """Garante que um diretório exista"""
    os.makedirs(path, exist_ok=True)
    return path

def get_python_command():
    """Obtém o comando Python adequado para o sistema"""
    if shutil.which("python3"):
        return "python3"
    elif shutil.which("python"):
        return "python"
    else:
        return None

def get_pip_command(python_cmd):
    """Obtém o comando pip adequado para o sistema"""
    if python_cmd:
        return f"{python_cmd} -m pip"
    else:
        return None

def run_command(cmd, shell=False, cwd=None):
    """Executa um comando e retorna se foi bem-sucedido"""
    try:
        log_info(f"Executando: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        subprocess.run(cmd, shell=shell, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Comando falhou com código {e.returncode}")
        return False
    except Exception as e:
        log_error(f"Erro ao executar comando: {e}")
        return False

# Funções principais
def check_prerequisites():
    """Verifica os pré-requisitos do sistema"""
    log_step("Verificando Pré-requisitos")
    
    issues = []
    
    # Verificar Python
    python_cmd = get_python_command()
    if not python_cmd:
        log_error("Python não encontrado. Por favor, instale Python 3.6+")
        issues.append("Python não encontrado")
    else:
        try:
            version = subprocess.check_output([python_cmd, "--version"], 
                                            universal_newlines=True).strip()
            log_success(f"Python encontrado: {version}")
        except:
            log_error("Não foi possível verificar a versão do Python")
            issues.append("Versão do Python não detectada")
    
    # Verificar pip
    pip_cmd = get_pip_command(python_cmd)
    if pip_cmd:
        try:
            version = subprocess.check_output([pip_cmd, "--version"], 
                                            universal_newlines=True).strip()
            log_success(f"pip encontrado: {version}")
        except:
            log_error("pip não encontrado ou não funcional")
            issues.append("pip não disponível")
    
    # Verificar Git (opcional)
    if shutil.which("git"):
        try:
            version = subprocess.check_output(["git", "--version"], 
                                            universal_newlines=True).strip()
            log_success(f"Git encontrado: {version}")
        except:
            log_warning("Git encontrado, mas não foi possível obter a versão")
    else:
        log_warning("Git não encontrado. Algumas funcionalidades podem ser limitadas")
    
    # Verificar espaço em disco
    try:
        # Forma multiplataforma de verificar espaço em disco
        total, used, free = shutil.disk_usage(PROJECT_ROOT)
        free_gb = free / (1024**3)
        if free_gb < 1.0:
            log_warning(f"Pouco espaço livre em disco: {free_gb:.2f} GB")
            issues.append("Espaço em disco pode ser insuficiente")
        else:
            log_success(f"Espaço em disco disponível: {free_gb:.2f} GB")
    except:
        log_warning("Não foi possível verificar o espaço em disco")
    
    if issues:
        log_warning("Alguns pré-requisitos não foram atendidos")
        return False
    else:
        log_success("Todos os pré-requisitos atendidos")
        return True

def create_directory_structure():
    """Cria a estrutura básica de diretórios"""
    log_step("Criando Estrutura de Diretórios")
    
    directories = [
        DOCS_DIR,
        DOCS_DIR / "stylesheets",
        DOCS_DIR / "javascripts",
        DOCS_DIR / "images",
        DOCS_DIR / "assets",
        DOCS_DIR / "assets/images",
        DOCS_DIR / "assets/icons",
        DOCS_DIR / "assets/css",
        DOCS_DIR / "assets/js",
        DOCS_DIR / "01-getting-started",
        DOCS_DIR / "02-architecture",
        DOCS_DIR / "03-development",
        DOCS_DIR / "04-trading",
        DOCS_DIR / "05-machine-learning",
        DOCS_DIR / "06-integrations",
        DOCS_DIR / "07-operations",
        DOCS_DIR / "08-api-reference",
        DOCS_DIR / "09-tutorials",
        DOCS_DIR / "10-legal-compliance",
        DOCS_DIR / "11-community",
        DOCS_DIR / "12-appendices",
    ]
    
    created_count = 0
    for directory in directories:
        if not directory.exists():
            try:
                os.makedirs(directory)
                log_info(f"Criado diretório: {directory.relative_to(PROJECT_ROOT)}")
                created_count += 1
            except Exception as e:
                log_error(f"Erro ao criar diretório {directory}: {e}")
    
    log_success(f"Estrutura criada: {created_count} diretórios")
    
    # Criar arquivo index.md se não existir
    index_file = DOCS_DIR / "index.md"
    if not index_file.exists():
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                f.write("""# Neural Crypto Bot 2.0

Welcome to the official documentation for Neural Crypto Bot 2.0, an enterprise-grade cryptocurrency trading bot powered by advanced machine learning and quantitative finance algorithms.

## 🚀 Features

- **Advanced ML Models**: Deep learning and reinforcement learning for predictive trading
- **Multi-Exchange Support**: Trade across major exchanges with a unified API
- **Risk Management**: Sophisticated risk controls and portfolio optimization
- **Microservices Architecture**: Scalable and resilient system design
- **Real-time Analytics**: Monitor performance with customizable dashboards
- **Extensible Strategy Framework**: Create and backtest custom strategies

## 📋 Getting Started

See the [Getting Started](01-getting-started/README.md) section for installation and configuration instructions.
""")
            log_success("Arquivo index.md criado")
        except Exception as e:
            log_error(f"Erro ao criar index.md: {e}")
    
    return True

def install_dependencies():
    """Instala as dependências necessárias"""
    log_step("Instalando Dependências")
    
    python_cmd = get_python_command()
    pip_cmd = get_pip_command(python_cmd)
    
    if not pip_cmd:
        log_error("Não foi possível determinar o comando pip")
        return False
    
    # Instalar mkdocs e plugins essenciais
    packages = [
        "mkdocs>=1.5.3",
        "mkdocs-material>=9.4.8",
        "pymdown-extensions>=10.0",
        "mkdocs-minify-plugin>=0.7.0",
        "mkdocs-macros-plugin>=1.0.4",
        "pillow>=10.0.0",
        "markdown>=3.4.0"
    ]
    
    log_info("Instalando pacotes: " + ", ".join(packages))
    cmd = f"{pip_cmd} install {' '.join(packages)}"
    
    if run_command(cmd, shell=True):
        log_success("Dependências instaladas com sucesso")
        return True
    else:
        log_error("Falha ao instalar dependências")
        return False

def create_mkdocs_config():
    """Cria o arquivo de configuração mkdocs.yml"""
    log_step("Criando Configuração MkDocs")
    
    # Fazer backup do arquivo existente se necessário
    if CONFIG_FILE.exists():
        backup_path = CONFIG_FILE.with_suffix(f".yml.bak.{time.strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(CONFIG_FILE, backup_path)
        log_info(f"Backup criado: {backup_path}")
    
    # Configuração básica do MkDocs
    config = """# Neural Crypto Bot 2.0 - Documentation Configuration
# Generated by Simple Setup Script

site_name: Neural Crypto Bot 2.0 Documentation
site_description: Enterprise-grade cryptocurrency trading bot
site_author: Neural Crypto Bot Team
site_url: https://docs.neuralcryptobot.com

# Repository
repo_name: neural-crypto-bot/neural-crypto-bot-2.0
repo_url: https://github.com/neural-crypto-bot/neural-crypto-bot-2.0

# Theme configuration
theme:
  name: material
  
  # Appearance
  palette:
    # Light mode
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    
    # Dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  
  # Fonts
  font:
    text: Roboto
    code: Roboto Mono
  
  # Features
  features:
    - navigation.instant
    - navigation.tracking
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - search.suggest
    - content.code.copy

# Navigation structure
nav:
  - Home: index.md
  - Getting Started: 01-getting-started/README.md
  - Architecture: 02-architecture/README.md
  - Development: 03-development/README.md
  - Trading: 04-trading/README.md
  - Machine Learning: 05-machine-learning/README.md
  - Integrations: 06-integrations/README.md
  - Operations: 07-operations/README.md
  - API Reference: 08-api-reference/README.md
  - Tutorials: 09-tutorials/README.md
  - Legal & Compliance: 10-legal-compliance/README.md
  - Community: 11-community/README.md
  - Appendices: 12-appendices/README.md

# Extensions
markdown_extensions:
  - admonition
  - attr_list
  - def_list
  - footnotes
  - md_in_html
  - toc:
      permalink: true
  - pymdownx.highlight
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed
  - pymdownx.tasklist

# Plugins
plugins:
  - search
  - minify:
      minify_html: true
      minify_js: true
      minify_css: true

# Extra stylesheets and scripts
extra_css:
  - stylesheets/extra.css

extra_javascript:
  - javascripts/extra.js
"""
    
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(config)
        log_success("Arquivo mkdocs.yml criado com sucesso")
        
        # Criar arquivos CSS e JS básicos
        css_dir = DOCS_DIR / "stylesheets"
        js_dir = DOCS_DIR / "javascripts"
        
        os.makedirs(css_dir, exist_ok=True)
        os.makedirs(js_dir, exist_ok=True)
        
        with open(css_dir / "extra.css", "w", encoding="utf-8") as f:
            f.write("/* Neural Crypto Bot 2.0 - Custom Styles */\n\n")
        
        with open(js_dir / "extra.js", "w", encoding="utf-8") as f:
            f.write("/* Neural Crypto Bot 2.0 - Custom JavaScript */\n\n")
        
        return True
    except Exception as e:
        log_error(f"Erro ao criar mkdocs.yml: {e}")
        return False

def serve_documentation():
    """Inicia o servidor local para visualização da documentação"""
    log_step("Iniciando Servidor de Documentação")
    
    if not CONFIG_FILE.exists():
        log_error("Arquivo mkdocs.yml não encontrado. Execute a configuração primeiro.")
        return False
    
    python_cmd = get_python_command()
    if not python_cmd:
        log_error("Python não encontrado")
        return False
    
    # Verificar se mkdocs está instalado
    try:
        subprocess.run([python_cmd, "-m", "mkdocs", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        log_error("MkDocs não está instalado ou não funcional")
        log_info("Tentando instalar MkDocs...")
        run_command(f"{get_pip_command(python_cmd)} install mkdocs mkdocs-material", shell=True)
    
    # Iniciar servidor
    log_info("Iniciando servidor MkDocs. Pressione Ctrl+C para encerrar.")
    log_success("Acesse a documentação em: http://localhost:8000")
    
    try:
        subprocess.run([python_cmd, "-m", "mkdocs", "serve"], cwd=PROJECT_ROOT)
        return True
    except KeyboardInterrupt:
        log_info("Servidor encerrado pelo usuário")
        return True
    except Exception as e:
        log_error(f"Erro ao iniciar servidor: {e}")
        return False

def build_documentation():
    """Gera o build da documentação"""
    log_step("Gerando Build da Documentação")
    
    if not CONFIG_FILE.exists():
        log_error("Arquivo mkdocs.yml não encontrado. Execute a configuração primeiro.")
        return False
    
    python_cmd = get_python_command()
    if not python_cmd:
        log_error("Python não encontrado")
        return False
    
    # Verificar se mkdocs está instalado
    try:
        subprocess.run([python_cmd, "-m", "mkdocs", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        log_error("MkDocs não está instalado ou não funcional")
        return False
    
    # Executar build
    site_dir = PROJECT_ROOT / "site"
    if site_dir.exists():
        log_info(f"Limpando diretório de build anterior: {site_dir}")
        try:
            shutil.rmtree(site_dir)
        except Exception as e:
            log_warning(f"Não foi possível limpar diretório anterior: {e}")
    
    if run_command([python_cmd, "-m", "mkdocs", "build", "--clean"], cwd=PROJECT_ROOT):
        log_success(f"Build gerado com sucesso em: {site_dir}")
        return True
    else:
        log_error("Falha ao gerar build")
        return False

def deploy_documentation():
    """Realiza o deploy da documentação (GitHub Pages)"""
    log_step("Realizando Deploy da Documentação")
    
    if not CONFIG_FILE.exists():
        log_error("Arquivo mkdocs.yml não encontrado. Execute a configuração primeiro.")
        return False
    
    python_cmd = get_python_command()
    if not python_cmd:
        log_error("Python não encontrado")
        return False
    
    # Verificar se mkdocs está instalado
    try:
        subprocess.run([python_cmd, "-m", "mkdocs", "--version"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        log_error("MkDocs não está instalado ou não funcional")
        return False
    
    # Verificar se está em um repositório Git
    if not (PROJECT_ROOT / ".git").exists():
        log_error("Não está em um repositório Git. Deploy para GitHub Pages requer Git.")
        return False
    
    # Realizar deploy
    log_info("Realizando deploy para GitHub Pages...")
    if run_command([python_cmd, "-m", "mkdocs", "gh-deploy", "--force"], cwd=PROJECT_ROOT):
        log_success("Deploy realizado com sucesso!")
        return True
    else:
        log_error("Falha ao realizar deploy")
        return False

def show_menu():
    """Exibe o menu principal"""
    print_banner()
    
    print(f"{Colors.CYAN}{Colors.BOLD}Menu Principal:{Colors.RESET}\n")
    print(f"1. {Colors.BOLD}Configuração Completa{Colors.RESET} (Verificar, Instalar, Configurar)")
    print(f"2. {Colors.BOLD}Verificar Pré-requisitos{Colors.RESET}")
    print(f"3. {Colors.BOLD}Criar Estrutura de Diretórios{Colors.RESET}")
    print(f"4. {Colors.BOLD}Instalar Dependências{Colors.RESET}")
    print(f"5. {Colors.BOLD}Criar Configuração MkDocs{Colors.RESET}")
    print(f"6. {Colors.BOLD}Servir Documentação{Colors.RESET} (Servidor Local)")
    print(f"7. {Colors.BOLD}Gerar Build{Colors.RESET}")
    print(f"8. {Colors.BOLD}Publicar Documentação{Colors.RESET} (GitHub Pages)")
    print(f"9. {Colors.BOLD}Sair{Colors.RESET}")
    
    while True:
        try:
            choice = input(f"\n{Colors.CYAN}Escolha uma opção [1-9]:{Colors.RESET} ")
            if choice.strip() in map(str, range(1, 10)):
                return choice.strip()
            else:
                print(f"{Colors.RED}Opção inválida. Escolha entre 1 e 9.{Colors.RESET}")
        except KeyboardInterrupt:
            print("\n")
            return "9"  # Sair
        except:
            print(f"{Colors.RED}Entrada inválida. Tente novamente.{Colors.RESET}")

def handle_menu_choice(choice):
    """Trata a escolha do menu principal"""
    if choice == "1":
        # Configuração Completa
        log_step("Iniciando Configuração Completa")
        
        steps = [
            ("Verificando pré-requisitos", check_prerequisites),
            ("Criando estrutura de diretórios", create_directory_structure),
            ("Instalando dependências", install_dependencies),
            ("Criando configuração MkDocs", create_mkdocs_config)
        ]
        
        success = True
        for desc, func in steps:
            log_info(f"Etapa: {desc}")
            if not func():
                log_warning(f"Etapa falhou: {desc}")
                success = False
                if not input(f"{Colors.YELLOW}Continuar mesmo com erro? (s/N):{Colors.RESET} ").lower().startswith('s'):
                    break
        
        if success:
            log_success("Configuração Completa realizada com sucesso!")
            
            # Perguntar se quer iniciar servidor
            if input(f"{Colors.CYAN}Deseja iniciar o servidor local? (s/N):{Colors.RESET} ").lower().startswith('s'):
                serve_documentation()
        else:
            log_warning("Configuração Completa concluída com avisos.")
        
    elif choice == "2":
        check_prerequisites()
    elif choice == "3":
        create_directory_structure()
    elif choice == "4":
        install_dependencies()
    elif choice == "5":
        create_mkdocs_config()
    elif choice == "6":
        serve_documentation()
    elif choice == "7":
        build_documentation()
    elif choice == "8":
        deploy_documentation()
    elif choice == "9":
        log_info("Saindo...")
        sys.exit(0)
    else:
        log_error("Opção inválida.")
    
    input(f"\n{Colors.YELLOW}Pressione ENTER para continuar...{Colors.RESET}")

def main():
    """Função principal"""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    while True:
        choice = show_menu()
        handle_menu_choice(choice)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
        log_info("Operação cancelada pelo usuário")
        sys.exit(0)
    except Exception as e:
        log_error(f"Erro inesperado: {e}")
        sys.exit(1)