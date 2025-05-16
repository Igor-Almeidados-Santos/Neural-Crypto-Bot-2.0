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
