# ============================================
# AI Document Translator - Environment Variables
# ============================================
# Copy this file to .env and fill in your values
# Never commit .env to version control!
#
# Documentation: https://github.com/yourusername/ai-document-translator/blob/main/README.md

# ============================================
# API Keys (Required)
# ============================================

# OpenAI API Key
# Get yours at: https://platform.openai.com/api-keys
# Required if using OpenAI engine
OPENAI_API_KEY=sk-your-openai-api-key-here

# DeepL API Key
# Get yours at: https://www.deepl.com/pro-api
# Required if using DeepL engine
DEEPL_API_KEY=your-deepl-api-key-here

# ============================================
# Engine Configuration (Optional)
# ============================================

# Default translation engine: openai, deepl
# Overrides config.yaml setting
TRANSLATION_ENGINE=openai

# OpenAI model name (if using OpenAI)
# Options: gpt-4o-mini, gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo
OPENAI_MODEL=gpt-4o-mini

# DeepL Pro flag (if using DeepL)
# Set to 'true' for Pro API, 'false' for Free API
DEEPL_PRO=false

# ============================================
# Feature Toggles (Optional)
# ============================================

# Enable/disable caching
CACHE_ENABLED=true

# Enable/disable glossary
GLOSSARY_ENABLED=true

# ============================================
# Logging Configuration (Optional)
# ============================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL=INFO

# Log directory path
LOG_DIR=logs

# Enable colored console output
LOG_COLORS=true

# ============================================
# Database Paths (Optional)
# ============================================

# Cache database location
CACHE_DB_PATH=data/cache.db

# Glossary database location
GLOSSARY_DB_PATH=data/glossary.db

# ============================================
# Performance Settings (Optional)
# ============================================

# Maximum parallel workers for batch processing
BATCH_MAX_WORKERS=3

# Request timeout in seconds
REQUEST_TIMEOUT=30

# Maximum retry attempts
MAX_RETRIES=3

# ============================================
# Development Settings (Optional)
# ============================================

# Enable debug mode
DEBUG=false

# Development environment flag
ENVIRONMENT=production

# Disable SSL verification (DO NOT use in production)
# VERIFY_SSL=true

# ============================================
# Notification Settings (Optional)
# ============================================

# Email for notifications (if implemented)
# NOTIFY_EMAIL=your-email@example.com

# Slack webhook for notifications (if implemented)
# SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# ============================================
# Cost Limits (Optional)
# ============================================

# Maximum cost in USD before warning
# COST_WARNING_THRESHOLD=50.0

# Maximum cost in USD before stopping
# COST_LIMIT=100.0

# ============================================
# Advanced Settings (Optional)
# ============================================

# Custom OpenAI API base URL (for proxies or custom deployments)
# OPENAI_API_BASE=https://api.openai.com/v1

# Organization ID (for OpenAI)
# OPENAI_ORG_ID=org-your-organization-id

# Default language settings
# DEFAULT_SOURCE_LANG=en
# DEFAULT_TARGET_LANG=ru
# DEFAULT_DOMAIN=general

# Output directory
# OUTPUT_DIR=output

# Temporary files directory
# TEMP_DIR=temp

# ============================================
# Security Notes
# ============================================
# 
# 1. NEVER commit .env file to version control
# 2. Add .env to .gitignore
# 3. Rotate API keys regularly
# 4. Use environment-specific .env files:
#    - .env.development
#    - .env.staging
#    - .env.production
# 5. Use secrets management in production:
#    - AWS Secrets Manager
#    - Azure Key Vault
#    - HashiCorp Vault
#    - Kubernetes Secrets
#
# ============================================
# Getting API Keys
# ============================================
#
# OpenAI:
#   1. Sign up at https://platform.openai.com
#   2. Go to API Keys section
#   3. Create new secret key
#   4. Copy and paste above
#   5. Add billing information
#
# DeepL:
#   1. Sign up at https://www.deepl.com/pro-api
#   2. Choose Free or Pro plan
#   3. Copy API key from account
#   4. Paste above
#
# ============================================
# Usage
# ============================================
#
# 1. Copy this file:
#    cp .env.example .env
#
# 2. Edit .env and add your API keys
#
# 3. Verify it's loaded:
#    python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('OPENAI_API_KEY'))"
#
# 4. Test configuration:
#    python -m src.cli.cli_interface test
#
# ============================================
# Troubleshooting
# ============================================
#
# If API keys aren't being loaded:
# 1. Check .env is in project root
# 2. Verify no spaces around = sign
# 3. Check for quotes (usually not needed)
# 4. Restart application/terminal
# 5. Check file permissions
#
# If getting authentication errors:
# 1. Verify API key is correct
# 2. Check API key hasn't expired
# 3. Verify billing is set up
# 4. Check API usage limits
#
# ============================================
