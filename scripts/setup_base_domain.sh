#!/bin/bash
# setup_base_domain.sh

echo "=== Verificando arquivos de domínio base para o Trading Bot ==="

# Garante que o diretório existe
mkdir -p src/common/domain

# Verifica se os arquivos já existem (o que será o caso após clonar do GitHub)
if [ -f src/common/domain/base_entity.py ] && [ -f src/common/domain/base_value_object.py ] && [ -f src/common/domain/base_event.py ]; then
    echo "✅ Arquivos de domínio base já existem no repositório."
else
    echo "⚠️ Alguns arquivos de domínio base estão faltando no repositório."
    echo "    Isso não deveria acontecer se o repositório foi clonado corretamente."
    echo "    Os arquivos deveriam estar incluídos diretamente no GitHub."
    
    # Você pode decidir se quer criar os arquivos como fallback ou falhar
    # Opção 1: Criar os arquivos como fallback (mantém o código original)
    echo "    Criando arquivos como fallback..."
    
    # O resto do script original que cria os arquivos...
    # [Código original de criação dos arquivos]
    
    # Ou Opção 2: Falhar e instruir o usuário
    # echo "Por favor, verifique o repositório Git e clone novamente."
    # exit 1
fi

echo "✅ Verificação de arquivos de domínio base concluída!"