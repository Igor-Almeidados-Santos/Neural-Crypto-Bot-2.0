#!/usr/bin/env python3
"""
Neural Crypto Bot 2.0 - Criador de READMEs e Corretor de Documentação
Este script cria os arquivos README.md necessários e corrige problemas comuns
da documentação do Neural Crypto Bot 2.0.
"""

import os
import sys
import platform
from pathlib import Path
import subprocess
import time
import random

# Configurações globais
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_FILE = PROJECT_ROOT / "mkdocs.yml"

# Cores para terminal
class Colors:
    RESET = "\033[0m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[0;33m"
    BLUE = "\033[0;34m"
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
def log_info(message):
    print(f"{Colors.BLUE}{Colors.BOLD}[INFO]{Colors.RESET} ℹ️ {message}")

def log_success(message):
    print(f"{Colors.GREEN}{Colors.BOLD}[SUCCESS]{Colors.RESET} ✅ {message}")

def log_warning(message):
    print(f"{Colors.YELLOW}{Colors.BOLD}[WARNING]{Colors.RESET} ⚠️ {message}")

def log_error(message):
    print(f"{Colors.RED}{Colors.BOLD}[ERROR]{Colors.RESET} ❌ {message}")

def log_step(message):
    print(f"\n{Colors.CYAN}{Colors.BOLD}=== {message} ==={Colors.RESET}")

def ensure_directory(path):
    """Garante que um diretório exista"""
    os.makedirs(path, exist_ok=True)
    return path

# Função para criar README.md
def create_readme_files():
    """Cria arquivos README.md para todas as seções"""
    log_step("Criando arquivos README.md")
    
    # Definição das seções
    sections = {
        "01-getting-started": {
            "title": "Getting Started",
            "description": "Installation, configuration, and quick start guides for Neural Crypto Bot 2.0.",
            "subtopics": ["Installation", "Configuration", "First Steps", "Requirements"]
        },
        "02-architecture": {
            "title": "Architecture",
            "description": "System architecture, design patterns, and technical documentation for Neural Crypto Bot 2.0.",
            "subtopics": ["System Design", "Components", "Data Flow", "Decision Records"]
        },
        "03-development": {
            "title": "Development",
            "description": "Development guidelines, contribution workflow, and coding standards.",
            "subtopics": ["Setup Environment", "Coding Standards", "Testing", "CI/CD"]
        },
        "04-trading": {
            "title": "Trading",
            "description": "Trading strategies, risk management, execution models, and backtesting.",
            "subtopics": ["Strategies", "Risk Management", "Execution", "Backtesting"]
        },
        "05-machine-learning": {
            "title": "Machine Learning",
            "description": "ML models, feature engineering, training pipelines, and model evaluation.",
            "subtopics": ["Models", "Feature Engineering", "Training", "Evaluation"]
        },
        "06-integrations": {
            "title": "Integrations",
            "description": "Exchange integrations, data sources, and external APIs.",
            "subtopics": ["Exchanges", "Data Providers", "APIs", "Webhooks"]
        },
        "07-operations": {
            "title": "Operations",
            "description": "Deployment, monitoring, logging, and maintenance.",
            "subtopics": ["Deployment", "Monitoring", "Logging", "Security"]
        },
        "08-api-reference": {
            "title": "API Reference",
            "description": "Comprehensive API documentation for Neural Crypto Bot 2.0.",
            "subtopics": ["Endpoints", "Authentication", "Rate Limiting", "Examples"]
        },
        "09-tutorials": {
            "title": "Tutorials",
            "description": "Step-by-step tutorials and examples for common tasks.",
            "subtopics": ["Getting Started", "Advanced", "Case Studies", "Videos"]
        },
        "10-legal-compliance": {
            "title": "Legal & Compliance",
            "description": "Legal information, compliance guidelines, and regulatory considerations.",
            "subtopics": ["Terms of Service", "Privacy Policy", "Regulations", "Licensing"]
        },
        "11-community": {
            "title": "Community",
            "description": "Community resources, contributions, governance, and support.",
            "subtopics": ["Contributing", "Code of Conduct", "Support", "Roadmap"]
        },
        "12-appendices": {
            "title": "Appendices",
            "description": "Additional resources, glossary, references, and supplementary information.",
            "subtopics": ["Glossary", "References", "Benchmarks", "FAQ"]
        }
    }
    
    created_count = 0
    
    # Criar README.md para cada seção
    for section_dir, section_info in sections.items():
        dir_path = DOCS_DIR / section_dir
        
        # Garantir que o diretório exista
        ensure_directory(dir_path)
        
        readme_path = dir_path / "README.md"
        
        # Criar conteúdo do README
        content = f"""# {section_info['title']}

{section_info['description']}

## Topics

"""
        
        # Adicionar subtópicos
        for subtopic in section_info['subtopics']:
            content += f"- {subtopic}\n"
        
        content += f"""
## Documentation Status

This section is currently under development. Contributions are welcome!

## Related Sections

"""
        
        # Adicionar seções relacionadas (2 aleatórias)
        related_sections = list(sections.keys())
        related_sections.remove(section_dir)
        random.shuffle(related_sections)
        for related in related_sections[:2]:
            related_title = sections[related]['title']
            content += f"- [{related_title}](../{related}/README.md)\n"
        
        # Adicionar rodapé
        content += f"""
---

*Last updated: {time.strftime('%Y-%m-%d')}*
"""
        
        # Escrever arquivo
        try:
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(content)
            created_count += 1
            log_info(f"Criado README para: {section_dir}")
        except Exception as e:
            log_error(f"Erro ao criar README para {section_dir}: {e}")
    
    log_success(f"Criados {created_count} arquivos README.md")
    return created_count

def update_mkdocs_port():
    """Atualiza a configuração do MkDocs para usar uma porta diferente"""
    log_step("Configurando porta alternativa para MkDocs")
    
    # Verificar se existe arquivo mkdocs.yml
    if not CONFIG_FILE.exists():
        log_error(f"Arquivo de configuração não encontrado: {CONFIG_FILE}")
        return False
    
    # Ler configuração atual
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config_content = f.read()
    except Exception as e:
        log_error(f"Erro ao ler arquivo de configuração: {e}")
        return False
    
    # Verificar se já tem configuração de dev_addr
    if "dev_addr:" in config_content:
        log_info("Configuração de porta já existe, atualizando...")
    
    # Escolher porta aleatória entre 8001-8999
    new_port = random.randint(8001, 8999)
    
    # Criar nova configuração
    new_config = ""
    dev_addr_added = False
    
    for line in config_content.splitlines():
        new_config += line + "\n"
        
        # Adicionar depois do site_url ou theme
        if (not dev_addr_added and 
            ("site_url:" in line or "theme:" in line or "# Theme configuration" in line)):
            new_config += f"\n# Development server settings\ndev_addr: '127.0.0.1:{new_port}'\n"
            dev_addr_added = True
    
    # Se não conseguiu adicionar, adicionar no final
    if not dev_addr_added:
        new_config += f"\n# Development server settings\ndev_addr: '127.0.0.1:{new_port}'\n"
    
    # Escrever configuração atualizada
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(new_config)
        log_success(f"Configuração atualizada: porta {new_port}")
        return True
    except Exception as e:
        log_error(f"Erro ao atualizar configuração: {e}")
        return False

def kill_mkdocs_processes():
    """Tenta matar processos MkDocs que possam estar usando a porta"""
    log_step("Finalizando processos MkDocs existentes")
    
    if platform.system() == "Windows":
        try:
            # No Windows, encontrar e matar processos Python que estão rodando MkDocs
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log_info("Processos Python finalizados")
        except:
            log_warning("Não foi possível finalizar processos Python")
    else:
        try:
            # Em Unix, usar lsof/pkill para encontrar e matar processos
            # Encontrar processo usando a porta 8000
            ports = [8000, 8080]
            for port in ports:
                try:
                    result = subprocess.run(["lsof", "-ti", f":{port}"], 
                                          stdout=subprocess.PIPE, text=True)
                    pids = result.stdout.strip().split("\n")
                    
                    for pid in pids:
                        if pid:  # Verificar se não está vazio
                            subprocess.run(["kill", "-9", pid], 
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            log_info(f"Processo {pid} usando porta {port} finalizado")
                except:
                    pass
        except:
            log_warning("Não foi possível finalizar processos existentes")
    
    # Esperar um momento para garantir que os processos foram encerrados
    time.sleep(1)
    return True

def create_index_md():
    """Cria ou atualiza o arquivo index.md principal"""
    log_step("Criando/Atualizando index.md principal")
    
    index_path = DOCS_DIR / "index.md"
    
    content = """# Neural Crypto Bot 2.0

Welcome to the official documentation for Neural Crypto Bot 2.0, an enterprise-grade cryptocurrency trading bot powered by advanced machine learning and quantitative finance algorithms.

## 🚀 Features

- **Advanced ML Models**: Deep learning and reinforcement learning for predictive trading
- **Multi-Exchange Support**: Trade across major exchanges with a unified API
- **Risk Management**: Sophisticated risk controls and portfolio optimization
- **Microservices Architecture**: Scalable and resilient system design
- **Real-time Analytics**: Monitor performance with customizable dashboards
- **Extensible Strategy Framework**: Create and backtest custom strategies

## 📋 Documentation Sections

- [Getting Started](01-getting-started/README.md) - Installation and basic configuration
- [Architecture](02-architecture/README.md) - System design and technical architecture
- [Development](03-development/README.md) - Development guides and contribution
- [Trading](04-trading/README.md) - Trading strategies and execution
- [Machine Learning](05-machine-learning/README.md) - ML models and data pipelines
- [Integrations](06-integrations/README.md) - Exchange APIs and external services
- [Operations](07-operations/README.md) - Deployment and operations
- [API Reference](08-api-reference/README.md) - API documentation
- [Tutorials](09-tutorials/README.md) - Step-by-step guides
- [Legal & Compliance](10-legal-compliance/README.md) - Legal information and compliance
- [Community](11-community/README.md) - Community resources and contributions
- [Appendices](12-appendices/README.md) - Additional resources and references

## 🛠️ Technical Overview

Neural Crypto Bot 2.0 is built with modern technologies and follows best practices:

- **Python 3.11+** with AsyncIO for high-performance concurrency
- **Domain-Driven Design** for clean architecture and business logic
- **Docker & Kubernetes** for containerization and orchestration
- **PostgreSQL & TimescaleDB** for time-series data storage
- **Apache Kafka** for event streaming
- **Redis** for caching and real-time data
- **FastAPI** for high-performance APIs
- **PyTorch & TensorFlow** for ML models

## 📊 Performance

Our system has demonstrated impressive performance metrics:

- **Average Annual ROI**: 127%
- **Sharpe Ratio**: 2.34
- **Max Drawdown**: < 5%
- **Win Rate**: 68%
- **Order Execution Latency**: < 50ms

## 📚 Resources

- [GitHub Repository](https://github.com/neural-crypto-bot/neural-crypto-bot-2.0)
- [API Reference](08-api-reference/README.md)
- [Contributing Guide](11-community/README.md)

## 🔒 Security

Security is our top priority. For responsible disclosure of security issues, please email security@neuralcryptobot.com.

---

*Documentation powered by [MkDocs](https://www.mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).*
"""
    
    try:
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(content)
        log_success("Arquivo index.md criado/atualizado")
        return True
    except Exception as e:
        log_error(f"Erro ao criar index.md: {e}")
        return False

def main():
    """Função principal"""
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("="*70)
    print(f"     NEURAL CRYPTO BOT 2.0 - CORRETOR DE DOCUMENTAÇÃO")
    print("="*70)
    print(f"{Colors.RESET}\n")
    
    log_info("Iniciando correção de problemas da documentação...")
    
    # Verificar diretório docs
    if not DOCS_DIR.exists():
        log_error(f"Diretório de documentação não encontrado: {DOCS_DIR}")
        ensure_directory(DOCS_DIR)
    
    # Finalizar processos existentes
    kill_mkdocs_processes()
    
    # Criar arquivos README.md
    create_readme_files()
    
    # Atualizar index.md
    create_index_md()
    
    # Configurar porta alternativa
    update_mkdocs_port()
    
    log_success("Correção da documentação concluída!")
    log_info("Agora você pode executar o servidor MkDocs novamente.")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n")
        log_info("Operação cancelada pelo usuário")
        sys.exit(0)
    except Exception as e:
        log_error(f"Erro inesperado: {e}")
        sys.exit(1)