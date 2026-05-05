#!/bin/bash
# Install pre-commit guardrails and lock in Phase E

set -e

cd "$(git rev-parse --show-toplevel)"

echo "📦 Installing Phase E guardrails..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python: $python_version"

# Install pre-commit framework
echo "  Installing pre-commit..."
pip3 install -q pre-commit

# Install linting tools
echo "  Installing ruff, mypy, vulture..."
pip3 install -q ruff mypy vulture

# Install pre-commit hooks (use full path since they're in user bin)
echo "  Setting up .git hooks..."
~/.local/bin/pre-commit install 2>/dev/null || /Users/mauf/Library/Python/3.9/bin/pre-commit install

# Run on all files to establish baseline
echo ""
echo "🔍 Running guardrails on all files (baseline)..."
~/.local/bin/pre-commit run --all-files --show-diff-on-failure 2>&1 | head -100 || \
/Users/mauf/Library/Python/3.9/bin/pre-commit run --all-files --show-diff-on-failure 2>&1 | head -100 || true

echo ""
echo "✅ Phase E guardrails installed!"
echo ""
echo "What was added:"
echo "  • .pre-commit-config.yaml  — hook definitions"
echo "  • pyproject.toml           — ruff + mypy config"
echo "  • scripts/check_conventions.py — one-shot script convention enforcer"
echo "  • .git/hooks/pre-commit    — auto-runs before every commit"
echo ""
echo "How it works:"
echo "  1. Before each commit, ruff auto-fixes and checks code"
echo "  2. Vulture detects dead code (confidence ≥80%)"
echo "  3. Format standardization (imports, line length, etc.)"
echo "  4. One-shot scripts must be in tools/oneshot_* (convention checker)"
echo ""
echo "To run manually:"
echo "  pre-commit run --all-files    # check all"
echo "  pre-commit run -a <file>      # check specific file"
echo ""
echo "To disable temporarily:"
echo "  git commit --no-verify        # skip hooks"
echo ""
