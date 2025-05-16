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
