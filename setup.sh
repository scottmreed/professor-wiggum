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

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Activate the virtual environment: source .venv/bin/activate"
echo "   2. Set your OpenAI API key: export OPENAI_API_KEY=sk-your-api-key-here"
echo "   3. Run the agent: mechanistic-agent"
echo ""
echo "📖 For more information, see AGENTS.md"
