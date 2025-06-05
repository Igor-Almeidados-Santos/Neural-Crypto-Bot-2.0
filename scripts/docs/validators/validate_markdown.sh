#!/bin/bash
# scripts/docs/validators/validate_markdown.sh - Validador avançado de Markdown
# Neural Crypto Bot 2.0 Documentation System
# Desenvolvido por Oasis - Validação de qualidade de documentação

set -euo pipefail

# Carregar funções comuns
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$SCRIPT_DIR")/common.sh"

# ================================
# CONFIGURAÇÕES ESPECÍFICAS
# ================================

readonly VALIDATOR_VERSION="2.0.0"
readonly MAX_LINE_LENGTH=100
readonly MIN_HEADING_LENGTH=3
readonly MAX_HEADING_LENGTH=60

# Padrões de validação
readonly VALID_EXTENSIONS=("md" "markdown")
readonly HEADING_PATTERN="^#{1,6} .+"
readonly LINK_PATTERN="\[([^\]]+)\]\(([^)]+)\)"
readonly IMAGE_PATTERN="!\[([^\]]*)\]\(([^)]+)\)"
readonly CODE_BLOCK_PATTERN="^```"

# ================================
# FUNÇÕES DE VALIDAÇÃO
# ================================

validate_file_structure() {
    log_step "Validando Estrutura de Arquivos"
    
    local issues=()
    local total_files=0
    local valid_files=0
    
    # Encontrar todos os arquivos markdown
    while IFS= read -r -d '' file; do
        ((total_files++))
        
        local filename=$(basename "$file")
        local extension="${filename##*.}"
        
        # Verificar extensão
        if [[ " ${VALID_EXTENSIONS[*]} " =~ " ${extension} " ]]; then
            ((valid_files++))
        else
            issues+=("Extensão inválida: $file")
            continue
        fi
        
        # Verificar nome do arquivo
        if [[ ! "$filename" =~ ^[a-z0-9._-]+$ ]]; then
            issues+=("Nome de arquivo inválido: $filename (use apenas lowercase, números, pontos, hífens)")
        fi
        
        # Verificar se arquivo não está vazio
        if [[ ! -s "$file" ]]; then
            issues+=("Arquivo vazio: $file")
        fi
        
        # Verificar encoding UTF-8
        if ! file -b --mime-encoding "$file" | grep -q "utf-8\|us-ascii"; then
            issues+=("Encoding não-UTF-8: $file")
        fi
        
    done < <(find "$DOCS_DIR" -name "*.md" -o -name "*.markdown" -print0 2>/dev/null)
    
    log_info "Arquivos encontrados: $total_files"
    log_info "Arquivos válidos: $valid_files"
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_warning "Problemas de estrutura encontrados:"
        for issue in "${issues[@]}"; do
            log_warning "  - $issue"
        done
        return 1
    fi
    
    log_success "Estrutura de arquivos válida"
    return 0
}

validate_markdown_syntax() {
    log_step "Validando Sintaxe Markdown"
    
    local issues=()
    local files_processed=0
    
    while IFS= read -r -d '' file; do
        ((files_processed++))
        local file_issues=()
        
        log_debug "Validando: $(basename "$file")"
        
        local line_number=0
        local in_code_block=false
        local code_block_lang=""
        local has_front_matter=false
        local front_matter_end=false
        
        while IFS= read -r line; do
            ((line_number++))
            
            # Detectar front matter
            if [[ $line_number -eq 1 && "$line" == "---" ]]; then
                has_front_matter=true
                continue
            fi
            
            if [[ $has_front_matter == true && ! $front_matter_end == true ]]; then
                if [[ "$line" == "---" ]]; then
                    front_matter_end=true
                fi
                continue
            fi
            
            # Detectar blocos de código
            if [[ "$line" =~ $CODE_BLOCK_PATTERN ]]; then
                if [[ $in_code_block == false ]]; then
                    in_code_block=true
                    code_block_lang="${line#\`\`\`}"
                    
                    # Validar linguagem do código
                    if [[ -n "$code_block_lang" && ! "$code_block_lang" =~ ^[a-zA-Z0-9_+-]+$ ]]; then
                        file_issues+=("Linha $line_number: Linguagem de código inválida: '$code_block_lang'")
                    fi
                else
                    in_code_block=false
                    code_block_lang=""
                fi
                continue
            fi
            
            # Pular validação dentro de blocos de código
            if [[ $in_code_block == true ]]; then
                continue
            fi
            
            # Validar comprimento da linha
            if [[ ${#line} -gt $MAX_LINE_LENGTH ]]; then
                file_issues+=("Linha $line_number: Linha muito longa (${#line} > $MAX_LINE_LENGTH)")
            fi
            
            # Validar cabeçalhos
            if [[ "$line" =~ $HEADING_PATTERN ]]; then
                local heading_text="${line#*# }"
                local heading_level="${line%%[^#]*}"
                
                # Validar comprimento do cabeçalho
                if [[ ${#heading_text} -lt $MIN_HEADING_LENGTH ]]; then
                    file_issues+=("Linha $line_number: Cabeçalho muito curto: '$heading_text'")
                elif [[ ${#heading_text} -gt $MAX_HEADING_LENGTH ]]; then
                    file_issues+=("Linha $line_number: Cabeçalho muito longo: '$heading_text'")
                fi
                
                # Validar hierarquia (simplificado)
                if [[ ${#heading_level} -gt 4 ]]; then
                    file_issues+=("Linha $line_number: Cabeçalho muito profundo (nível ${#heading_level})")
                fi
                
                # Verificar espaço após #
                if [[ ! "$line" =~ ^#+\ .+ ]]; then
                    file_issues+=("Linha $line_number: Falta espaço após '#' no cabeçalho")
                fi
            fi
            
            # Validar links
            if [[ "$line" =~ $LINK_PATTERN ]]; then
                local link_matches
                link_matches=$(echo "$line" | grep -oE '\[([^\]]+)\]\(([^)]+)\)' || true)
                
                while IFS= read -r match; do
                    if [[ -n "$match" ]]; then
                        local link_text=$(echo "$match" | sed 's/\[\([^\]]*\)\].*/\1/')
                        local link_url=$(echo "$match" | sed 's/.*\](\([^)]*\)).*/\1/')
                        
                        # Validar texto do link
                        if [[ -z "$link_text" ]]; then
                            file_issues+=("Linha $line_number: Link sem texto: '$match'")
                        fi
                        
                        # Validar URL do link
                        if [[ -z "$link_url" ]]; then
                            file_issues+=("Linha $line_number: Link sem URL: '$match'")
                        elif [[ "$link_url" =~ ^https?:// ]]; then
                            # Link externo - validação básica
                            if [[ ! "$link_url" =~ ^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/.*)?$ ]]; then
                                file_issues+=("Linha $line_number: URL externa malformada: '$link_url'")
                            fi
                        elif [[ "$link_url" =~ ^/ ]