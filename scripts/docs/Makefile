# Neural Crypto Bot 2.0 - Makefile para sistema de documentação
# Facilita a execução de comandos em sistemas Unix (Linux/macOS)

# Definição de variáveis
SHELL := /bin/bash
PROJECT_ROOT := $(shell pwd)
DOCS_DIR := $(PROJECT_ROOT)/docs
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts/docs
BUILD_DIR := $(PROJECT_ROOT)/site

# Cores
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
RESET := \033[0m
BOLD := \033[1m

# Detectar Python
PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
POETRY := $(shell which poetry 2>/dev/null)

# Verificar se estamos em ambiente Poetry
ifdef POETRY
	MKDOCS := poetry run mkdocs
else
	MKDOCS := mkdocs
endif

# Alvos principais
.PHONY: help setup check dirs deps config serve build deploy clean

# Ajuda
help:
	@echo -e "$(CYAN)$(BOLD)Neural Crypto Bot 2.0 - Sistema de Documentação$(RESET)"
	@echo -e "$(CYAN)Comandos disponíveis:$(RESET)"
	@echo -e "  $(BOLD)make setup$(RESET)      Configuração completa do sistema de documentação"
	@echo -e "  $(BOLD)make check$(RESET)      Verificar pré-requisitos do sistema"
	@echo -e "  $(BOLD)make dirs$(RESET)       Criar estrutura de diretórios"
	@echo -e "  $(BOLD)make deps$(RESET)       Instalar dependências"
	@echo -e "  $(BOLD)make config$(RESET)     Criar configuração MkDocs"
	@echo -e "  $(BOLD)make serve$(RESET)      Iniciar servidor local para visualização"
	@echo -e "  $(BOLD)make build$(RESET)      Gerar build da documentação"
	@echo -e "  $(BOLD)make deploy$(RESET)     Publicar documentação no GitHub Pages"
	@echo -e "  $(BOLD)make clean$(RESET)      Limpar arquivos gerados"

# Configuração completa
setup:
	@echo -e "$(CYAN)$(BOLD)=== Configuração Completa ===$(RESET)"
	@if [ -x "$(SCRIPTS_DIR)/setup_documentation.sh" ]; then \
		"$(SCRIPTS_DIR)/setup_documentation.sh"; \
	else \
		echo -e "$(RED)Script não encontrado ou não tem permissão de execução$(RESET)"; \
		echo -e "$(YELLOW)Executando: chmod +x $(SCRIPTS_DIR)/setup_documentation.sh$(RESET)"; \
		chmod +x "$(SCRIPTS_DIR)/setup_documentation.sh"; \
		"$(SCRIPTS_DIR)/setup_documentation.sh"; \
	fi

# Verificar pré-requisitos
check:
	@echo -e "$(CYAN)$(BOLD)=== Verificando Pré-requisitos ===$(RESET)"
	@if [ -x "$(SCRIPTS_DIR)/check_prerequisites.sh" ]; then \
		"$(SCRIPTS_DIR)/check_prerequisites.sh"; \
	else \
		echo -e "$(RED)Script não encontrado ou não tem permissão de execução$(RESET)"; \
		echo -e "$(YELLOW)Executando: chmod +x $(SCRIPTS_DIR)/check_prerequisites.sh$(RESET)"; \
		chmod +x "$(SCRIPTS_DIR)/check_prerequisites.sh"; \
		"$(SCRIPTS_DIR)/check_prerequisites.sh"; \
	fi

# Criar estrutura de diretórios
dirs:
	@echo -e "$(CYAN)$(BOLD)=== Criando Estrutura de Diretórios ===$(RESET)"
	@if [ -x "$(SCRIPTS_DIR)/create_directories.sh" ]; then \
		"$(SCRIPTS_DIR)/create_directories.sh"; \
	else \
		echo -e "$(RED)Script não encontrado ou não tem permissão de execução$(RESET)"; \
		echo -e "$(YELLOW)Executando: chmod +x $(SCRIPTS_DIR)/create_directories.sh$(RESET)"; \
		chmod +x "$(SCRIPTS_DIR)/create_directories.sh"; \
		"$(SCRIPTS_DIR)/create_directories.sh"; \
	fi

# Instalar dependências
deps:
	@echo -e "$(CYAN)$(BOLD)=== Instalando Dependências ===$(RESET)"
	@if [ -x "$(SCRIPTS_DIR)/install_dependencies.sh" ]; then \
		"$(SCRIPTS_DIR)/install_dependencies.sh"; \
	else \
		echo -e "$(RED)Script não encontrado ou não tem permissão de execução$(RESET)"; \
		echo -e "$(YELLOW)Executando: chmod +x $(SCRIPTS_DIR)/install_dependencies.sh$(RESET)"; \
		chmod +x "$(SCRIPTS_DIR)/install_dependencies.sh"; \
		"$(SCRIPTS_DIR)/install_dependencies.sh"; \
	fi

# Criar configuração
config:
	@echo -e "$(CYAN)$(BOLD)=== Criando Configuração ===$(RESET)"
	@if [ -x "$(SCRIPTS_DIR)/create_configuration.sh" ]; then \
		"$(SCRIPTS_DIR)/create_configuration.sh"; \
	else \
		echo -e "$(RED)Script não encontrado ou não tem permissão de execução$(RESET)"; \
		echo -e "$(YELLOW)Executando: chmod +x $(SCRIPTS_DIR)/create_configuration.sh$(RESET)"; \
		chmod +x "$(SCRIPTS_DIR)/create_configuration.sh"; \
		"$(SCRIPTS_DIR)/create_configuration.sh"; \
	fi

# Iniciar servidor local
serve:
	@echo -e "$(CYAN)$(BOLD)=== Iniciando Servidor Local ===$(RESET)"
	@if [ -f "$(PROJECT_ROOT)/mkdocs.yml" ]; then \
		$(MKDOCS) serve; \
	else \
		echo -e "$(RED)Arquivo mkdocs.yml não encontrado$(RESET)"; \
		echo -e "$(YELLOW)Execute 'make config' primeiro$(RESET)"; \
	fi

# Gerar build
build:
	@echo -e "$(CYAN)$(BOLD)=== Gerando Build da Documentação ===$(RESET)"
	@if [ -f "$(PROJECT_ROOT)/mkdocs.yml" ]; then \
		$(MKDOCS) build --strict; \
		echo -e "$(GREEN)Build concluído: $(BUILD_DIR)$(RESET)"; \
	else \
		echo -e "$(RED)Arquivo mkdocs.yml não encontrado$(RESET)"; \
		echo -e "$(YELLOW)Execute 'make config' primeiro$(RESET)"; \
	fi

# Publicar documentação
deploy:
	@echo -e "$(CYAN)$(BOLD)=== Publicando Documentação ===$(RESET)"
	@if [ -f "$(PROJECT_ROOT)/mkdocs.yml" ]; then \
		if [ -x "$(SCRIPTS_DIR)/publish_docs.sh" ]; then \
			"$(SCRIPTS_DIR)/publish_docs.sh"; \
		else \
			echo -e "$(YELLOW)Script publish_docs.sh não encontrado, usando comando direto$(RESET)"; \
			$(MKDOCS) gh-deploy --force; \
		fi \
	else \
		echo -e "$(RED)Arquivo mkdocs.yml não encontrado$(RESET)"; \
		echo -e "$(YELLOW)Execute 'make config' primeiro$(RESET)"; \
	fi

# Limpar arquivos gerados
clean:
	@echo -e "$(CYAN)$(BOLD)=== Limpando Arquivos Gerados ===$(RESET)"
	@if [ -d "$(BUILD_DIR)" ]; then \
		rm -rf "$(BUILD_DIR)"; \
		echo -e "$(GREEN)Diretório $(BUILD_DIR) removido$(RESET)"; \
	else \
		echo -e "$(YELLOW)Diretório $(BUILD_DIR) não encontrado$(RESET)"; \
	fi
	@if [ -d "$(PROJECT_ROOT)/.docs-cache" ]; then \
		rm -rf "$(PROJECT_ROOT)/.docs-cache"; \
		echo -e "$(GREEN)Cache removido$(RESET)"; \
	fi
	@echo -e "$(GREEN)Limpeza concluída$(RESET)"

# Ativar ambiente virtual se existir
activate:
	@if [ -f "$(PROJECT_ROOT)/activate-docs-env.sh" ]; then \
		source "$(PROJECT_ROOT)/activate-docs-env.sh"; \
	else \
		echo -e "$(RED)Arquivo activate-docs-env.sh não encontrado$(RESET)"; \
		echo -e "$(YELLOW)Execute 'make deps' primeiro$(RESET)"; \
	fi