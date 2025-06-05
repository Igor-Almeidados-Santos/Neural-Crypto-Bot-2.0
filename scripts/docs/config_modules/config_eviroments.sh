#!/bin/bash
# scripts/docs/config_modules/config_environments.sh
# Configurações específicas por ambiente (dev/staging/prod)
# Neural Crypto Bot 2.0 Documentation System

# ================================
# CONFIGURAÇÕES DE AMBIENTE
# ================================

config_environments() {
    log_substep "Criando Configurações por Ambiente"
    
    local environments=("dev" "staging" "prod")
    local created_configs=0
    
    for env in "${environments[@]}"; do
        if create_environment_config "$env"; then
            ((created_configs++))
            log_success "✅ Configuração $env criada"
        else
            log_error "❌ Falha ao criar configuração $env"
        fi
    done
    
    log_info "Configurações de ambiente criadas: $created_configs/${#environments[@]}"
    
    if [[ $created_configs -eq ${#environments[@]} ]]; then
        log_success "Todas as configurações de ambiente criadas"
        return 0
    else
        log_error "Algumas configurações de ambiente falharam"
        return 1
    fi
}

# ================================
# CONFIGURAÇÃO DE DESENVOLVIMENTO
# ================================

create_environment_config() {
    local environment="$1"
    local config_file="$PROJECT_ROOT/mkdocs.${environment}.yml"
    
    case "$environment" in
        "dev")
            create_development_config "$config_file"
            ;;
        "staging")
            create_staging_config "$config_file"
            ;;
        "prod")
            create_production_config "$config_file"
            ;;
        *)
            log_error "Ambiente desconhecido: $environment"
            return 1
            ;;
    esac
}

create_development_config() {
    local config_file="$1"
    
    cat > "$config_file" << 'EOF'
# Development Configuration - Neural Crypto Bot 2.0
# Optimized for local development and testing

INHERIT: mkdocs.yml

# Development URLs
site_url: http://localhost:8000
use_directory_urls: false

# Development server settings
dev_addr: '0.0.0.0:8000'

# Simplified plugins for faster builds
plugins:
  - search:
      separator: '[\s\-,:!=\[\]()"`/]+|\.(?!\d)|&[lg]t;|(?!\b)(?=[A-Z][a-z])'
  
  - macros:
      module_name: docs/macros
      include_dir: docs/templates
      include_yaml:
        - docs/data/config.yml
        - docs/data/performance.yml
  
  - awesome-pages:
      filename: .pages
      collapse_single_pages: true
      strict: false
  
  - glightbox:
      touchNavigation: true
      loop: false
      effect: zoom

# Simplified theme for development
theme:
  name: material
  custom_dir: docs/overrides
  
  logo: assets/images/logo.png
  favicon: assets/images/favicon.ico
  language: en
  
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  
  font:
    text: Inter
    code: JetBrains Mono
  
  # Essential features only for development
  features:
    - navigation.instant
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - content.code.copy
    - content.code.select
    - search.suggest
    - search.highlight
    - toc.follow

# Development-specific extras
extra:
  # Disable analytics in development
  analytics: null
  
  # Development environment indicator
  environment: development
  
  # Simplified social links
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/neural-crypto-bot/neural-crypto-bot-2.0
      name: GitHub Repository

# Watch additional files in development
watch:
  - src/
  - scripts/docs/
  - docs/macros/
  - docs/stylesheets/
  - docs/javascripts/
  - mkdocs.yml
  - mkdocs.dev.yml

# Development CSS and JS
extra_css:
  - stylesheets/extra.css
  - stylesheets/ncb-theme.css

extra_javascript:
  - javascripts/mathjax.js
  - javascripts/shortcuts.js
  - javascripts/copy-code.js

# Relaxed validation for development
validation:
  omitted_files: ignore
  absolute_links: ignore
  unrecognized_links: ignore
EOF
    
    return 0
}

# ================================
# CONFIGURAÇÃO DE STAGING
# ================================

create_staging_config() {
    local config_file="$1"
    
    cat > "$config_file" << 'EOF'
# Staging Configuration - Neural Crypto Bot 2.0
# Pre-production environment for testing

INHERIT: mkdocs.yml

# Staging URLs
site_url: https://staging-docs.neuralcryptobot.com

# Full plugin suite for staging testing
plugins:
  - search:
      separator: '[\s\-,:!=\[\]()"`/]+|\.(?!\d)|&[lg]t;|(?!\b)(?=[A-Z][a-z])'
  
  - minify:
      minify_html: true
      minify_js: true
      minify_css: true
      htmlmin_opts:
        remove_comments: false  # Keep comments in staging
        remove_empty_space: true
      cache_safe: true
  
  - git-revision-date-localized:
      enable_creation_date: true
      type: datetime
      timezone: UTC
      locale: en
      fallback_to_build_date: true
  
  - macros:
      module_name: docs/macros
      include_dir: docs/templates
      include_yaml:
        - docs/data/config.yml
        - docs/data/performance.yml
  
  - awesome-pages:
      filename: .pages
      collapse_single_pages: true
      strict: false
  
  - glightbox:
      touchNavigation: true
      loop: false
      effect: zoom
      slide_effect: slide
      width: 100%
      height: auto
      zoomable: true
      draggable: true
      auto_caption: false
      caption_position: bottom
  
  - swagger-ui-tag:
      background: White
      syntaxHighlightTheme: monokai
  
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
            show_root_toc_entry: true
            show_object_full_path: false
            show_category_heading: true
            show_if_no_docstring: true
            show_signature_annotations: true
            show_symbol_type_heading: true
            show_symbol_type_toc: true
            signature_crossrefs: true
            merge_init_into_class: true
            docstring_style: google
            docstring_options:
              ignore_init_summary: true
            members_order: source
            group_by_category: true
            heading_level: 2
  
  - section-index
  
  - redirects:
      redirect_maps:
        'old-docs.md': 'index.md'
        'legacy/api.md': '08-api-reference/README.md'

# Staging-specific theme settings
theme:
  name: material
  custom_dir: docs/overrides
  
  logo: assets/images/logo.png
  favicon: assets/images/favicon.ico
  language: en
  
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: blue
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  
  font:
    text: Inter
    code: JetBrains Mono
  
  # All features enabled for staging testing
  features:
    - announce.dismiss
    - content.action.edit
    - content.action.view
    - content.code.annotate
    - content.code.copy
    - content.code.select
    - content.tabs.link
    - content.tooltips
    - header.autohide
    - navigation.expand
    - navigation.footer
    - navigation.indexes
    - navigation.instant
    - navigation.instant.prefetch
    - navigation.instant.progress
    - navigation.prune
    - navigation.sections
    - navigation.tabs
    - navigation.tabs.sticky
    - navigation.top
    - navigation.tracking
    - search.highlight
    - search.share
    - search.suggest
    - toc.follow
    - toc.integrate

# Staging-specific extras
extra:
  # Staging analytics
  analytics:
    provider: google
    property: G-STAGING-XXXX
    feedback:
      title: Was this page helpful? (Staging)
      ratings:
        - icon: material/emoticon-happy-outline
          name: This page was helpful
          data: 1
        - icon: material/emoticon-sad-outline
          name: This page could be improved
          data: 0
  
  # Environment indicator
  environment: staging
  
  # Staging banner
  announce:
    title: "🚧 Staging Environment - Not for production use"
    text: "This is a testing environment. Content may be incomplete or experimental."
  
  # Full social links for testing
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/neural-crypto-bot/neural-crypto-bot-2.0
      name: GitHub Repository
    - icon: fontawesome/brands/discord
      link: https://discord.gg/neural-crypto-bot
      name: Discord Community
    - icon: fontawesome/brands/telegram
      link: https://t.me/neural_crypto_bot
      name: Telegram Channel
    - icon: fontawesome/brands/twitter
      link: https://twitter.com/neuralcryptobot
      name: Twitter Updates

# Standard validation for staging
validation:
  omitted_files: warn
  absolute_links: warn
  unrecognized_links: warn

# All assets for staging testing
extra_css:
  - stylesheets/extra.css
  - stylesheets/ncb-theme.css
  - stylesheets/api-styles.css
  - stylesheets/print.css

extra_javascript:
  - javascripts/mathjax.js
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
  - javascripts/feedback.js
  - javascripts/shortcuts.js
  - javascripts/copy-code.js
  - javascripts/external-links.js
EOF
    
    return 0
}

# ================================
# CONFIGURAÇÃO DE PRODUÇÃO
# ================================

create_production_config() {
    local config_file="$1"
    
    cat > "$config_file" << 'EOF'
# Production Configuration - Neural Crypto Bot 2.0
# Optimized for production deployment

INHERIT: mkdocs.yml

# Production URLs
site_url: https://docs.neuralcryptobot.com

# Full production plugin suite
plugins:
  - search:
      separator: '[\s\-,:!=\[\]()"`/]+|\.(?!\d)|&[lg]t;|(?!\b)(?=[A-Z][a-z])'
  
  - minify:
      minify_html: true
      minify_js: true
      minify_css: true
      htmlmin_opts:
        remove_comments: true
        remove_empty_space: true
        remove_optional_attribute_quotes: true
        use_short_doctype: true
        remove_script_type_attributes: true
        remove_style_link_type_attributes: true
      cache_safe: true
  
  - git-revision-date-localized:
      enable_creation_date: true
      type: datetime
      timezone: UTC
      locale: en
      fallback_to_build_date: true
  
  - git-committers:
      repository: neural-crypto-bot/neural-crypto-bot-2.0
      branch: main
      enabled: !ENV [ENABLE_GIT_COMMITTERS, true]
  
  - macros:
      module_name: docs/macros
      include_dir: docs/templates
      include_yaml:
        - docs/data/config.yml
        - docs/data/performance.yml
  
  - awesome-pages:
      filename: .pages
      collapse_single_pages: true
      strict: false
  
  - glightbox:
      touchNavigation: true
      loop: false
      effect: zoom
      slide_effect: slide
      width: 100%
      height: auto
      zoomable: true
      draggable: true
      auto_caption: false
      caption_position: bottom
  
  - swagger-ui-tag:
      background: White
      syntaxHighlightTheme: monokai
  
  - gen-files:
      scripts:
      - scripts/docs/gen_ref_pages.py
  
  - literate-nav:
      nav_file: SUMMARY.md
  
  - section-index
  
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
            show_root_toc_entry: true
            show_object_full_path: false
            show_category_heading: true
            show_if_no_docstring: true