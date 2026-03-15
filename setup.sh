#!/bin/bash
# setup.sh - Automated setup script for Mechanistic Agent

set -e  # Exit on any error

echo "🧪 Setting up Mechanistic Agent..."

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Please run this script from the project root directory"
    echo "   Expected to find pyproject.toml in current directory"
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ "$python_version" < "3.10" ]]; then
    echo "❌ Python 3.10+ required, found $python_version"
    echo "   Please install Python 3.10 or higher"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
else
    echo "📦 Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install the project in development mode
echo "🔧 Installing project in development mode..."
pip install -e .

# Install Node dependencies for rdkit_cli backend
if command -v npm >/dev/null 2>&1; then
    echo "🧪 Installing Node dependencies (rdkit_cli backend)..."
    npm install
else
    echo "⚠️ npm not found; rdkit_cli backend will be unavailable until npm install is run."
fi

# Verify setup
echo "✅ Verifying setup..."
python -c "from mechanistic_agent.config import ReactionInputs; print('✅ Package imported successfully')"

# Test basic functionality
echo "🧪 Testing basic functionality..."
python -c "
from mechanistic_agent.config import ReactionInputs
reaction = ReactionInputs(starting_materials=['C=O', 'OCCO'], products=['C1OCOC1'])
print('✅ ReactionInputs test passed:', reaction.reaction_summary)
"

# Verify LLM provider routing (OpenAI, Gemini, OpenRouter from llm.py)
echo "🤖 Verifying LLM provider routing..."
python -c "
from mechanistic_agent.llm import (
    get_provider_label,
    is_gemini_model,
    is_openrouter_model,
    get_chat_model,
)
# Provider detection
assert get_provider_label('gpt-4o') == 'OpenAI', 'OpenAI routing'
assert get_provider_label('gemini-2.0-flash') == 'Gemini', 'Gemini routing'
assert get_provider_label('anthropic/claude-opus-4.6') == 'OpenRouter', 'OpenRouter routing'
assert get_provider_label('allenai/olmo-3.1-32b-instruct') == 'OpenRouter', 'OpenRouter (OLMo) routing'
assert is_gemini_model('gemini-1.5-pro')
assert not is_gemini_model('gpt-4o')
assert is_openrouter_model('anthropic/claude-opus-4.6')
assert not is_openrouter_model('gpt-4o')
print('✅ LLM provider routing passed')
"

# Verify get_chat_model raises clear errors when API keys are missing (run with keys unset)
echo "🤖 Verifying LLM API key error messages..."
env -i PATH="${PATH:-}" HOME="${HOME:-}" OPENAI_API_KEY= OPENROUTER_API_KEY= GOOGLE_API_KEY= GEMINI_API_KEY= python -c "
from mechanistic_agent.llm import get_chat_model
try:
    get_chat_model('gpt-4o')
except RuntimeError as e:
    if 'OpenAI API key' not in str(e):
        raise AssertionError('Expected OpenAI key error, got: ' + str(e))
try:
    get_chat_model('anthropic/claude-opus-4.6')
except RuntimeError as e:
    if 'OpenRouter API key' not in str(e):
        raise AssertionError('Expected OpenRouter key error, got: ' + str(e))
try:
    get_chat_model('gemini-2.0-flash')
except RuntimeError as e:
    msg = str(e)
    if 'Gemini' not in msg and 'GOOGLE_API_KEY' not in msg and 'GEMINI_API_KEY' not in msg:
        raise AssertionError('Expected Gemini key error, got: ' + msg)
print('✅ LLM API key error checks passed')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source .venv/bin/activate"
echo "   2. Set API key(s) for the model family you use:"
echo "      • OpenAI (gpt-4o, gpt-5.x):  export OPENAI_API_KEY=sk-your-key"
echo "      • OpenRouter (Claude, OLMo): export OPENROUTER_API_KEY=sk-or-your-key"
echo "      • Gemini:                    export GOOGLE_API_KEY=your-key  # or GEMINI_API_KEY"
echo "   3. Run the agent: python main.py serve   # or: mechanistic-agent"
echo ""
echo "📖 For more information, see AGENTS.md"
