# Neural Crypto Bot 2.0 🤖💎

[![Build Status](https://github.com/your-username/Neural-Crypto-Bot-2.0/workflows/CI/badge.svg)](https://github.com/your-username/Neural-Crypto-Bot-2.0/actions)
[![Coverage](https://img.shields.io/badge/coverage-85%25-green.svg)](./coverage.xml)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Visão Geral

O **Neural Crypto Bot 2.0** é uma plataforma avançada de trading algorítmico de criptomoedas, construída com arquitetura de microserviços e técnicas de machine learning de última geração. Desenvolvido com foco em **performance**, **confiabilidade** e **escalabilidade** para ambientes de produção.

### 🚀 Características Principais

- **🧠 Machine Learning Avançado**: Modelos ensemble com LSTM, Transformers e Reinforcement Learning
- **⚡ Ultra-Baixa Latência**: Execução de ordens em menos de 100ms
- **🔄 Multi-Exchange**: Suporte a 15+ exchanges (Binance, Coinbase, Kraken, Bybit, etc.)
- **📊 Analytics Avançado**: Dashboard em tempo real com métricas de performance
- **🛡️ Gestão de Risco**: Sistema multi-camadas com VaR dinâmico e circuit breakers
- **🔐 Segurança Enterprise**: Autenticação JWT, criptografia AES-256, gestão de segredos
- **📈 Estratégias Proprietárias**: 10+ estratégias pré-configuradas e framework para estratégias customizadas

### 🏗️ Arquitetura

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │   Mobile App     │    │   Trading APIs  │
└─────────┬───────┘    └────────┬─────────┘    └─────────┬───────┘
          │                     │                        │
          └─────────────────────┼────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      API Gateway      │
                    └───────────┬───────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
    ┌─────▼─────┐    ┌─────────▼─────────┐    ┌───────▼───────┐
    │ Execution │    │   Data Collection │    │ Model Training│
    │  Service  │    │     Service       │    │    Service    │
    └─────┬─────┘    └─────────┬─────────┘    └───────┬───────┘
          │                    │                      │
          └────────────────────┼──────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │              Message Bus (Kafka)                    │
    └──────────────────────────┬──────────────────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────┐
    │         Data Layer (PostgreSQL + Redis)            │
    └─────────────────────────────────────────────────────┘
```

## 📋 Pré-requisitos por Sistema Operacional

### 🐧 **Linux (Ubuntu/Debian)**
```bash
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y docker.io docker-compose-v2
sudo apt install -y git curl wget

# Configurar Docker
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER

# Verificar instalação
python3.11 --version
docker --version
docker compose version
```

### 🍎 **macOS**
```bash
# Instalar Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Instalar dependências
brew install python@3.11
brew install docker
brew install git

# Instalar Docker Desktop
brew install --cask docker

# Verificar instalação
python3.11 --version
docker --version
docker compose version
```

### 🪟 **Windows 10/11**

#### Opção 1: WSL2 (Recomendado)
```powershell
# Habilitar WSL2
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# Reiniciar e instalar Ubuntu
wsl --install -d Ubuntu-22.04

# No WSL2, seguir instruções do Linux
```

#### Opção 2: Windows Nativo
```powershell
# Instalar Python 3.11 (https://python.org)
# Instalar Docker Desktop (https://docker.com)
# Instalar Git (https://git-scm.com)

# Verificar no PowerShell
python --version
docker --version
docker compose version
```

### 🔍 Verificação Automática de Pré-requisitos

```bash
# Linux/macOS/WSL2
./scripts/check_prerequisites.sh

# Windows PowerShell
.\scripts\check_prerequisites.ps1
```

### 📊 Requisitos de Sistema

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **RAM** | 4GB | 8GB+ |
| **Armazenamento** | 10GB | 50GB+ SSD |
| **Internet** | 10Mbps | 100Mbps+ |

## 🚀 Instalação Detalhada por Sistema

### 🐧 **Instalação no Linux**

#### 1. Clone e Preparação
```bash
# Clone do repositório
git clone https://github.com/your-username/Neural-Crypto-Bot-2.0.git
cd Neural-Crypto-Bot-2.0

# Dar permissões de execução
chmod +x install.sh
chmod +x scripts/*.sh
```

#### 2. Instalação Automática
```bash
# Verificar pré-requisitos
./scripts/check_prerequisites.sh

# Instalação completa
./install.sh

# Ou passo a passo
./scripts/setup_poetry.sh
./scripts/setup_docker.sh
./scripts/setup_configs.sh
```

#### 3. Configuração de Environment
```bash
# Copiar template de configuração
cp .env.example .env

# Editar configurações
nano .env  # ou vim .env
```

#### 4. Inicialização
```bash
# Iniciar todos os serviços
./scripts/start_docker.sh

# Ou usar utilitários de desenvolvimento
./scripts/dev_utils.sh up
```

---

### 🍎 **Instalação no macOS**

#### 1. Preparação do Ambiente
```bash
# Verificar se Xcode Command Line Tools estão instalados
xcode-select --install

# Clone do repositório
git clone https://github.com/your-username/Neural-Crypto-Bot-2.0.git
cd Neural-Crypto-Bot-2.0
```

#### 2. Configuração de Permissões
```bash
# macOS requer permissões explícitas
chmod +x install.sh
chmod +x scripts/*.sh

# Verificar pré-requisitos
./scripts/check_prerequisites.sh
```

#### 3. Instalação
```bash
# Instalação automática
./install.sh

# Se houver problemas com Poetry, instalar manualmente:
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

#### 4. Docker Desktop
```bash
# Iniciar Docker Desktop manualmente ou
open /Applications/Docker.app

# Aguardar Docker estar rodando
docker info
```

#### 5. Inicialização
```bash
# Configurar environment
cp .env.example .env
# Editar .env com suas preferências

# Iniciar serviços
./scripts/start_docker.sh
```

---

### 🪟 **Instalação no Windows**

#### **Método 1: WSL2 (Recomendado)**

```powershell
# 1. Habilitar WSL2 no PowerShell (Admin)
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform

# 2. Reiniciar o sistema

# 3. Instalar Ubuntu
wsl --install -d Ubuntu-22.04

# 4. No terminal Ubuntu, seguir instruções do Linux
```

#### **Método 2: Windows Nativo**

##### PowerShell (Executar como Administrador)
```powershell
# 1. Clone do repositório
git clone https://github.com/your-username/Neural-Crypto-Bot-2.0.git
cd Neural-Crypto-Bot-2.0

# 2. Verificar pré-requisitos
.\scripts\check_prerequisites.ps1

# 3. Configuração manual do Poetry
# Se Poetry não estiver instalado:
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# 4. Configurar PATH do Poetry
$env:PATH += ";$env:APPDATA\Python\Scripts"
```

##### Configuração de Environment
```powershell
# Copiar template
Copy-Item .env.example .env

# Editar com Notepad
notepad .env
```

##### Inicialização
```powershell
# Verificar se Docker Desktop está rodando
docker info

# Iniciar via PowerShell script
.\start_docker.ps1

# Ou usar comandos manuais
docker-compose up -d postgres redis zookeeper kafka
Start-Sleep -Seconds 15
docker-compose up -d collector execution training api
```

#### **Método 3: Docker Desktop + VS Code**

```powershell
# 1. Instalar Docker Desktop
# Download: https://docs.docker.com/desktop/install/windows-install/

# 2. Instalar VS Code + extensões
# - Remote - Containers
# - Docker
# - Python

# 3. Abrir projeto no container
# Ctrl+Shift+P -> "Remote-Containers: Open Folder in Container"
```

---

## 🔧 Configuração Avançada por Sistema

### 🐧 **Linux - Otimizações**

```bash
# Aumentar limites de arquivo para alta performance
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf

# Otimizações de rede
echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf

# Aplicar mudanças
sudo sysctl -p
```

### 🍎 **macOS - Configurações**

```bash
# Aumentar limite de descritores de arquivo
echo 'ulimit -n 65536' >> ~/.zshrc  # ou ~/.bash_profile
source ~/.zshrc

# Configurar Docker Desktop para mais recursos
# Docker Desktop -> Settings -> Resources
# CPU: 4 cores
# Memory: 6GB
# Swap: 2GB
```

### 🪟 **Windows - Performance**

```powershell
# WSL2: Configurar recursos
# Criar arquivo .wslconfig em %USERPROFILE%
@"
[wsl2]
memory=6GB
processors=4
localhostForwarding=true
"@ | Out-File -FilePath "$env:USERPROFILE\.wslconfig" -Encoding UTF8

# Reiniciar WSL2
wsl --shutdown
wsl
```

---

## 🎛️ Comandos por Sistema Operacional

### 📋 **Comandos Unificados**

| Operação | Linux/macOS | Windows PowerShell | Windows WSL2 |
|----------|-------------|-------------------|--------------|
| **Verificar pré-requisitos** | `./scripts/check_prerequisites.sh` | `.\scripts\check_prerequisites.ps1` | `./scripts/check_prerequisites.sh` |
| **Instalação** | `./install.sh` | `.\install.ps1` | `./install.sh` |
| **Iniciar serviços** | `./scripts/start_docker.sh` | `.\start_docker.ps1` | `./scripts/start_docker.sh` |
| **Ver logs** | `./scripts/dev_utils.sh logs` | `.\scripts\dev_utils.ps1 logs` | `./scripts/dev_utils.sh logs` |
| **Parar serviços** | `./scripts/dev_utils.sh down` | `.\scripts\dev_utils.ps1 down` | `./scripts/dev_utils.sh down` |

### 🔧 **Comandos de Desenvolvimento**

```bash
# === LINUX/MACOS ===
# Ambiente virtual Poetry
poetry shell
poetry install

# Testes
poetry run pytest tests/ -v

# Linting
poetry run black src/
poetry run isort src/
poetry run ruff src/

# === WINDOWS POWERSHELL ===
# Ambiente virtual Poetry
poetry shell
poetry install

# Testes
poetry run pytest tests\ -v

# Linting  
poetry run black src\
poetry run isort src\
poetry run ruff src\

# === DOCKER (TODOS OS SISTEMAS) ===
# Build de containers
docker-compose build

# Logs em tempo real
docker-compose logs -f

# Shell de serviços
docker-compose exec api bash
docker-compose exec postgres psql -U neuralbot neuralcryptobot
```

---

## 🐛 Troubleshooting por Sistema

### 🐧 **Linux Issues**

```bash
# Problema: Permissão negada no Docker
sudo usermod -aG docker $USER
newgrp docker

# Problema: Poetry não encontrado
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# Problema: Porta em uso
sudo lsof -i :8000
sudo kill -9 <PID>
```

### 🍎 **macOS Issues**

```bash
# Problema: Docker Desktop não inicia
killall Docker && open /Applications/Docker.app

# Problema: Poetry PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Problema: Permissões Gatekeeper
sudo spctl --master-disable  # Temporário
sudo xattr -r -d com.apple.quarantine .
```

### 🪟 **Windows Issues**

```powershell
# Problema: Execution Policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Problema: WSL2 não funciona
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all
bcdedit /set hypervisorlaunchtype auto

# Problema: Docker no WSL2
# Instalar Docker Desktop e habilitar WSL2 integration

# Problema: Porta ocupada
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

---

## 🌐 URLs de Acesso por Sistema

### **Padrão (Todos os Sistemas)**
- **API Principal**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs  
- **Dashboard Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090

### **WSL2 Specific**
```bash
# Obter IP do WSL2
ip addr show eth0 | grep inet

# Acessar do Windows usando IP do WSL2
# Exemplo: http://172.20.10.2:8000
```

### **Docker Desktop Specific**
- Usar sempre `localhost` - Docker Desktop gerencia port forwarding automaticamente

## 🔧 Configuração Detalhada

### Configuração de Environment (.env)

```bash
# === CONFIGURAÇÕES ESSENCIAIS ===
ENVIRONMENT=development
DEBUG=True
SECRET_KEY=your-super-secret-key-here

# === EXCHANGES APIs ===
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret
COINBASE_API_KEY=your_coinbase_key
COINBASE_API_SECRET=your_coinbase_secret

# === CONFIGURAÇÕES DE TRADING ===
DEFAULT_TRADING_PAIRS=BTC/USDT,ETH/USDT,SOL/USDT
MAX_POSITION_SIZE=0.05  # 5% do portfólio por posição
MAX_LEVERAGE=2.0
RISK_FREE_RATE=0.03

# === INFRAESTRUTURA ===
DATABASE_URL=postgresql://neuralbot:password@localhost:5432/neuralcryptobot
REDIS_URL=redis://localhost:6379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### Configuração de Estratégias

```python
# Exemplo de configuração personalizada
STRATEGIES = {
    "momentum_lstm": {
        "enabled": True,
        "pairs": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "1h",
        "max_positions": 2,
        "stop_loss": 0.02,
        "take_profit": 0.04
    },
    "mean_reversion": {
        "enabled": True,
        "pairs": ["SOL/USDT", "AVAX/USDT"],
        "bollinger_periods": 20,
        "rsi_threshold": [30, 70]
    }
}
```

## 🎯 Inicialização Detalhada por Sistema

### 🐧 **Processo Completo - Linux**

#### **Passo 1: Verificação e Preparação**
```bash
# 1. Verificar sistema
./scripts/check_prerequisites.sh

# 2. Setup do ambiente Python
./scripts/setup_poetry.sh

# 3. Configurar Docker  
./scripts/setup_docker.sh

# 4. Gerar configurações
./scripts/setup_configs.sh

# 5. Verificar se tudo está OK
poetry --version
docker --version
docker compose version
```

#### **Passo 2: Configuração de Environment**
```bash
# Copiar e editar configurações
cp .env.example .env

# Configurações mínimas necessárias
cat >> .env << EOF
# APIs de Exchange (obrigatório)
BINANCE_API_KEY=your_binance_key_here
BINANCE_API_SECRET=your_binance_secret_here

# Configurações de Trading
DEFAULT_TRADING_PAIRS=BTC/USDT,ETH/USDT
MAX_POSITION_SIZE=0.05
ENVIRONMENT=development
DEBUG=True
EOF
```

#### **Passo 3: Inicialização dos Serviços**
```bash
# Método 1: Script automatizado
./scripts/start_docker.sh

# Método 2: Passo a passo
docker compose up -d postgres redis zookeeper  # Infraestrutura
sleep 10
docker compose up -d kafka                     # Messaging
sleep 5
docker compose up -d collector execution api   # Aplicação

# Verificar status
docker compose ps
```

#### **Passo 4: Validação**
```bash
# Testar conectividade
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/status

# Verificar logs
docker compose logs api | tail -20
docker compose logs collector | tail -20
```

---

### 🍎 **Processo Completo - macOS**

#### **Passo 1: Preparação do Ambiente**
```bash
# 1. Verificar se Command Line Tools estão instalados
xcode-select --install

# 2. Instalar Homebrew se necessário
which brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Instalar dependências
brew install python@3.11 docker git

# 4. Iniciar Docker Desktop
open /Applications/Docker.app
# Aguardar até Docker estar disponível
until docker info >/dev/null 2>&1; do sleep 1; done
```

#### **Passo 2: Setup do Projeto**
```bash
# Clone e permissões
git clone https://github.com/your-username/Neural-Crypto-Bot-2.0.git
cd Neural-Crypto-Bot-2.0
chmod +x install.sh scripts/*.sh

# Verificação completa
./scripts/check_prerequisites.sh

# Instalação
./install.sh
```

#### **Passo 3: Configuração Específica do macOS**
```bash
# Configurar Poetry PATH se necessário
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Configurar .env
cp .env.example .env
# Editar com editor preferido
open -a TextEdit .env  # ou vim .env
```

#### **Passo 4: Inicialização**
```bash
# Garantir que Docker Desktop está rodando
docker info

# Iniciar serviços
./scripts/start_docker.sh

# Acessar dashboard
open http://localhost:3000  # Grafana
open http://localhost:8000/docs  # API Docs
```

---

### 🪟 **Processo Completo - Windows**

#### **Método Recomendado: WSL2**

```powershell
# 1. Preparação do WSL2 (PowerShell como Admin)
# Habilitar recursos necessários
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

# 2. Reiniciar sistema e instalar Ubuntu
# Após reiniciar:
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2

# 3. No terminal Ubuntu WSL2
# Atualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependências
sudo apt install -y python3.11 python3.11-dev python3.11-venv
sudo apt install -y docker.io docker-compose-v2 git

# Configurar Docker
sudo usermod -aG docker $USER
newgrp docker

# 4. Seguir processo do Linux
git clone https://github.com/your-username/Neural-Crypto-Bot-2.0.git
cd Neural-Crypto-Bot-2.0
./scripts/check_prerequisites.sh
./install.sh
```

#### **Método Alternativo: Windows Nativo**

```powershell
# 1. Verificação de pré-requisitos
.\scripts\check_prerequisites.ps1

# 2. Configuração manual se necessário
# Instalar Python 3.11 de https://python.org
# Instalar Docker Desktop de https://docker.com
# Instalar Git de https://git-scm.com

# 3. Configurar Poetry
if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
    $env:PATH += ";$env:APPDATA\Python\Scripts"
}

# 4. Setup do projeto
poetry install
Copy-Item .env.example .env
notepad .env  # Editar configurações

# 5. Inicialização
# Garantir que Docker Desktop está rodando
docker info

# Iniciar via script PowerShell
.\start_docker.ps1

# Ou usar docker-compose diretamente
docker-compose up -d
```

#### **Configuração Específica WSL2**
```bash
# Dentro do WSL2 - Configurar recursos
# Criar .wslconfig no Windows
echo '[wsl2]
memory=6GB
processors=4' > /mnt/c/Users/$USER/.wslconfig

# Reiniciar WSL2 se necessário
# No PowerShell do Windows:
# wsl --shutdown
# wsl
```

---

### 🔄 **Sequência de Inicialização Padrão**

#### **1. Preparação de Infraestrutura (30-60s)**
```bash
# Subir banco de dados e cache
docker compose up -d postgres redis

# Aguardar inicialização
echo "Aguardando PostgreSQL..."
until docker compose exec postgres pg_isready -U neuralbot; do sleep 1; done

echo "Aguardando Redis..."
until docker compose exec redis redis-cli ping; do sleep 1; done
```

#### **2. Messaging e Streaming (15-30s)**
```bash
# Subir Zookeeper e Kafka
docker compose up -d zookeeper
sleep 10
docker compose up -d kafka

# Verificar Kafka
echo "Aguardando Kafka..."
until docker compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list; do sleep 2; done
```

#### **3. Serviços de Aplicação (20-40s)**
```bash
# Subir serviços principais
docker compose up -d collector execution training

# Aguardar serviços estarem healthy
sleep 15

# Subir API Gateway
docker compose up -d api

# Verificar saúde
curl -f http://localhost:8000/health || echo "API ainda não disponível"
```

#### **4. Monitoramento (10-20s)**
```bash
# Subir Prometheus e Grafana
docker compose up -d prometheus grafana

# URLs disponíveis após inicialização completa:
echo "✅ Serviços disponíveis:"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"  
echo "   Grafana: http://localhost:3000 (admin/neuralbot)"
echo "   Prometheus: http://localhost:9090"
```

---

### 📊 **Status e Monitoramento da Inicialização**

#### **Verificação de Status Completa**
```bash
# Script de verificação completa
#!/bin/bash
echo "=== STATUS DOS SERVIÇOS ==="

services=("postgres" "redis" "kafka" "api" "collector" "execution")
for service in "${services[@]}"; do
    if docker compose ps $service | grep -q "Up"; then
        echo "✅ $service: Running"
    else
        echo "❌ $service: Down"
    fi
done

echo ""
echo "=== VERIFICAÇÃO DE CONECTIVIDADE ==="

# Teste de APIs
curl -s http://localhost:8000/health && echo "✅ API Health OK" || echo "❌ API Health Failed"
curl -s http://localhost:8000/api/v1/status && echo "✅ API Status OK" || echo "❌ API Status Failed"

# Teste de banco
docker compose exec postgres pg_isready -U neuralbot && echo "✅ PostgreSQL OK" || echo "❌ PostgreSQL Failed"

# Teste de Redis
docker compose exec redis redis-cli ping && echo "✅ Redis OK" || echo "❌ Redis Failed"

echo ""
echo "=== LOGS RECENTES ==="
docker compose logs --tail=5 api
```

#### **Comandos de Debug por Sistema**
```bash
# === LINUX/MACOS ===
# Ver logs em tempo real
./scripts/dev_utils.sh logs

# Restart de serviço específico
./scripts/dev_utils.sh restart api

# Shell de debug
./scripts/dev_utils.sh shell api

# === WINDOWS POWERSHELL ===
# Ver logs
docker-compose logs -f

# Restart serviço
docker-compose restart api

# Shell de debug  
docker-compose exec api bash

# === WINDOWS WSL2 ===
# Mesmo que Linux/macOS
./scripts/dev_utils.sh logs
```

## 🌐 Interfaces de Acesso

### Principais Endpoints

| Serviço | URL | Descrição |
|---------|-----|-----------|
| **API Principal** | http://localhost:8000 | API REST e GraphQL |
| **Dashboard** | http://localhost:3000 | Interface Web (Grafana) |
| **Docs Interativa** | http://localhost:8000/docs | Swagger UI |
| **Prometheus** | http://localhost:9090 | Métricas do sistema |
| **Logs Centralizados** | `docker logs -f ncb_api` | Logs estruturados |

### Credenciais Padrão

```
Grafana Dashboard:
  Usuário: admin
  Senha: neuralbot

PostgreSQL:
  Host: localhost:5433
  Usuário: neuralbot
  Senha: password
  Database: neuralcryptobot
```

## 📊 Monitoramento e Observabilidade

### Métricas Principais

- **Performance de Trading**: ROI, Sharpe Ratio, Drawdown
- **Latência de Execução**: P95, P99 de orders
- **Saúde do Sistema**: CPU, Memória, Disk I/O
- **Errors & Alertas**: Falhas de conexão, erros de API

### Dashboards Disponíveis

1. **Trading Performance**: Métricas de P&L em tempo real
2. **System Health**: Status de todos os microserviços  
3. **Exchange Connectivity**: Latência e uptime das exchanges
4. **Risk Management**: Exposição e limites de risco
5. **ML Models**: Performance dos modelos preditivos

## 🔐 Segurança e Compliance

### Práticas de Segurança Implementadas

- ✅ **Criptografia**: TLS 1.3 para todas as comunicações
- ✅ **Autenticação**: JWT com rotação automática de tokens
- ✅ **Autorização**: RBAC (Role-Based Access Control)
- ✅ **Secrets Management**: Hashicorp Vault integration
- ✅ **API Rate Limiting**: Proteção contra abuse
- ✅ **Audit Logging**: Logs completos de todas as operações

### Configuração de Segurança

```bash
# Gerar chave secreta
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Configurar Vault (opcional)
export VAULT_ADDR=http://localhost:8200
vault kv put secret/ncb api_keys=@api_keys.json
```

## 🧪 Testes e Qualidade

### Executar Suite de Testes

```bash
# Testes unitários
poetry run pytest tests/unit -v

# Testes de integração
poetry run pytest tests/integration -v

# Testes de sistema
poetry run pytest tests/system -v

# Coverage completo
poetry run pytest --cov=src --cov-report=html
```

### Quality Gates

- **Code Coverage**: >80%
- **Type Coverage**: >90% (MyPy)
- **Security Score**: A+ (Bandit)
- **Performance**: <100ms API response

## 📈 Estratégias de Trading Incluídas

### 1. Momentum LSTM
- **Timeframe**: 1h-4h
- **Indicadores**: RSI, MACD, Volume
- **ML Model**: LSTM + Attention
- **Performance**: 15-25% anual

### 2. Mean Reversion
- **Timeframe**: 15m-1h  
- **Indicadores**: Bollinger Bands, Z-Score
- **Entry**: Oversold/Overbought extremes
- **Performance**: 12-18% anual

### 3. Arbitrage Statistical
- **Pairs**: Correlated crypto pairs
- **Method**: Cointegration analysis
- **Execution**: Delta-neutral positions
- **Performance**: 8-12% anual (baixo risco)

### 4. News Sentiment
- **Data Sources**: Twitter, Reddit, News APIs
- **NLP Model**: BERT fine-tuned
- **Timeframe**: 5m-30m reactions
- **Performance**: 20-30% anual (alta volatilidade)

## 🚀 Deployment em Produção

### Docker Swarm (Recomendado)

```bash
# Inicializar swarm
docker swarm init

# Deploy da stack
docker stack deploy -c docker-compose.prod.yml ncb

# Escalar serviços
docker service scale ncb_api=3 ncb_execution=2
```

### Kubernetes

```bash
# Deploy no cluster
kubectl apply -f deployment/kubernetes/

# Monitorar deployment
kubectl get pods -n neural-crypto-bot
kubectl logs -f deployment/api -n neural-crypto-bot
```

### Configurações de Produção

```yaml
# Recursos mínimos recomendados
CPU: 4 cores
RAM: 8GB
Storage: 100GB SSD
Network: 1Gbps
```

## 🔧 Troubleshooting

### Problemas Comuns

#### 1. Falha na Conexão com Exchange
```bash
# Verificar conectividade
curl -I https://api.binance.com/api/v3/ping

# Verificar configuração
./scripts/dev_utils.sh shell api
python -c "from src.exchanges import BinanceClient; print(BinanceClient().test_connection())"
```

#### 2. PostgreSQL não Inicia
```bash
# Verificar logs
docker logs ncb_postgres

# Reset do banco
docker volume rm neural-crypto-bot_postgres-data
./scripts/dev_utils.sh up
```

#### 3. Performance Lenta
```bash
# Verificar recursos
docker stats

# Otimizar configuração
# Editar docker-compose.yml - aumentar limites de CPU/RAM
```

### Logs de Debug

```bash
# Logs estruturados por serviço
./scripts/dev_utils.sh logs api
./scripts/dev_utils.sh logs collector  
./scripts/dev_utils.sh logs execution

# Buscar por erros específicos
docker logs ncb_api 2>&1 | grep ERROR
```

## 📚 Documentação Adicional

- 📖 [Guia de Desenvolvimento](./docs/DEVELOPMENT.md)
- 🏗️ [Arquitetura Detalhada](./docs/ARCHITECTURE.md)
- 📊 [Guia de Estratégias](./docs/STRATEGIES.md)
- 🔐 [Configuração de Segurança](./docs/SECURITY.md)
- 🚀 [Deploy em Produção](./docs/PRODUCTION.md)
- 🐛 [Troubleshooting Avançado](./docs/TROUBLESHOOTING.md)

## 🤝 Contribuição

### Setup de Desenvolvimento

```bash
# Clone e setup
git clone https://github.com/your-username/Neural-Crypto-Bot-2.0.git
cd Neural-Crypto-Bot-2.0
./scripts/dev_utils.sh init

# Instalar pre-commit hooks
poetry run pre-commit install

# Criar branch para feature
git checkout -b feature/nova-estrategia
```

### Padrões de Código

- **Python**: Black + isort + ruff + mypy
- **Commits**: Conventional Commits
- **Testes**: Pytest com >80% coverage
- **Docs**: Google-style docstrings

## 📄 Licença

Este projeto está licenciado sob a [MIT License](./LICENSE).

## ⚠️ Disclaimer

**Este software é para fins educacionais e de pesquisa. Trading de criptomoedas envolve riscos significativos. Use por sua própria conta e risco. Os desenvolvedores não se responsabilizam por perdas financeiras.**

---

<div align="center">

**Construído com ❤️ por [Igor Almeida](https://github.com/your-username)**

[🌟 Star no GitHub](https://github.com/your-username/Neural-Crypto-Bot-2.0) • [🐛 Reportar Bug](https://github.com/your-username/Neural-Crypto-Bot-2.0/issues) • [💡 Sugerir Feature](https://github.com/your-username/Neural-Crypto-Bot-2.0/discussions)

</div>