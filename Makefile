# Makefile for Oasis Trading System
# ==================================
# Utiliza .PHONY para garantir que os alvos sempre executem, mesmo se um arquivo com o mesmo nome existir.
.PHONY: help install clean run-api lint format test test-coverage docs-serve docs-build docker-up docker-down docker-build

# Configurações
PYTHON_INTERPRETER = poetry run python
SRC_DIR = src
TEST_DIR = tests

# Alvo padrão, executado quando 'make' é chamado sem argumentos.
default: help

# Ajuda: Descreve todos os comandos disponíveis.
help:
	@echo "------------------------------------------------------------------------"
	@echo " Oásis Trading System - Comandos de Automação"
	@echo "------------------------------------------------------------------------"
	@echo " Comandos:"
	@echo "   install         -> Instala as dependências do projeto com Poetry."
	@echo "   clean           -> Remove arquivos temporários e caches."
	@echo "   run-api         -> Inicia o servidor da API FastAPI em modo de desenvolvimento."
	@echo "   lint            -> Executa o linter (ruff) para verificar a qualidade do código."
	@echo "   format          -> Formata todo o código do projeto com (ruff format)."
	@echo "   test            -> Executa a suíte de testes unitários e de integração."
	@echo "   test-coverage   -> Executa os testes e gera um relatório de cobertura."
	@echo "   docs-serve      -> Inicia o servidor local da documentação (MkDocs)."
	@echo "   docs-build      -> Gera a documentação estática em HTML."
	@echo "   docker-up       -> Inicia os contêineres Docker (API, DB, Cache)."
	@echo "   docker-down     -> Para e remove os contêineres Docker."
	@echo "   docker-build    -> Força a reconstrução das imagens Docker."
	@echo "------------------------------------------------------------------------"

# Instalação
install:
	@echo "📦 Instalando dependências com Poetry..."
	@poetry install

# Limpeza
clean:
	@echo "🧹 Limpando arquivos de cache e build..."
	@find . -type f -name "*.py[co]" -delete
	@find . -type d -name "__pycache__" -delete
	@rm -rf .pytest_cache .mypy_cache .ruff_cache coverage.xml htmlcov site

# Desenvolvimento
run-api:
	@echo "🚀 Iniciando a API em modo de desenvolvimento..."
	@poetry run uvicorn api.main:app --app-dir $(SRC_DIR) --reload

# Qualidade de Código
lint:
	@echo "✅ Verificando a qualidade do código com Ruff..."
	@poetry run ruff check $(SRC_DIR) $(TEST_DIR)
	@echo "✅ Verificando a tipagem com Mypy..."
	@poetry run mypy $(SRC_DIR)

format:
	@echo "🎨 Formatando o código com Ruff..."
	@poetry run ruff format $(SRC_DIR) $(TEST_DIR)

# Testes
test:
	@echo "🧪 Executando testes..."
	@poetry run pytest

test-coverage:
	@echo "🧪 Executando testes e gerando relatório de cobertura..."
	@poetry run pytest --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html

# Documentação
docs-serve:
	@echo "📚 Iniciando servidor da documentação em http://127.0.0.1:8001"
	@poetry run mkdocs serve -a 127.0.0.1:8001

docs-build:
	@echo "📚 Gerando a documentação estática..."
	@poetry run mkdocs build

# Docker
docker-up:
	@echo "🐳 Iniciando contêineres com Docker Compose..."
	@docker-compose up -d

docker-down:
	@echo "🛑 Parando contêineres Docker..."
	@docker-compose down

docker-build:
	@echo "🏗️ Reconstruindo imagens Docker..."
	@docker-compose build