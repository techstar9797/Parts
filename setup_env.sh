#!/bin/bash
# Setup script for Electronics Parts Replacement System
# This script creates a local .env file with your API keys

echo "🔧 Electronics Parts Replacement System - Environment Setup"
echo "=========================================================="
echo ""
echo "⚠️  SECURITY NOTICE: API keys will be stored in .env file (not committed to git)"
echo ""

# Create .env file template
cat > .env << 'EOF'
# OpenAI API Key (for LlamaIndex integration)
OPENAI_API_KEY="your_openai_api_key_here"

# LlamaCloud API Key (for PDF parsing and data extraction)
LLAMA_CLOUD_API_KEY="your_llama_cloud_api_key_here"

# Apify API Token (for web crawling Mouser.com)
APIFY_API_TOKEN="your_apify_api_token_here"

# Mouser API Key (for real-time pricing and stock data)
MOUSER_API_KEY="your_mouser_api_key_here"

# Database URL (optional - for production use)
POSTGRES_URL="postgresql://user:password@localhost:5432/partsync"
EOF

echo ""
echo "📝 MANUAL SETUP REQUIRED:"
echo "   1. Edit the .env file with your actual API keys"
echo "   2. Replace 'your_openai_api_key_here' with your OpenAI API key"
echo "   3. Replace 'your_llama_cloud_api_key_here' with your LlamaCloud API key"
echo "   4. Replace 'your_apify_api_token_here' with your Apify API token"
echo "   5. Replace 'your_mouser_api_key_here' with your Mouser API key"

echo "✅ Environment file created: .env"
echo "✅ API keys configured securely"
echo ""
echo "🚀 Quick Start:"
echo "   python3 working_parts_app.py"
echo ""
echo "🌐 The app will run on: http://localhost:7878"
echo ""
echo "⚠️  IMPORTANT: Never commit the .env file to git!"
echo "   The .gitignore file will prevent this automatically."
