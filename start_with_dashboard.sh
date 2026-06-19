#!/bin/bash

# Start AI Trading Command Center with Dashboard
# This script starts all three components in new tmux windows
# Usage: ./start_with_dashboard.sh

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"

echo "🚀 Starting AI Trading Command Center..."
echo "Project root: $PROJECT_ROOT"

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Install with: brew install tmux"
    exit 1
fi

# Kill existing session if it exists
tmux kill-session -t trading-system 2>/dev/null || true

# Create new tmux session
echo "📦 Creating tmux session..."
tmux new-session -d -s trading-system -x 200 -y 50

# Window 1: API Server
echo "🌐 Starting API server on http://localhost:8000..."
tmux new-window -t trading-system -n api
tmux send-keys -t trading-system:api "cd '$PROJECT_ROOT' && python3 api_server.py" Enter

# Window 2: Frontend
echo "🎛️ Starting dashboard on http://localhost:3000..."
tmux new-window -t trading-system -n dashboard
tmux send-keys -t trading-system:dashboard "cd '$DASHBOARD_DIR' && npm install && npm run dev" Enter

# Window 3: Trading Bot
echo "🤖 Starting trading bot..."
tmux new-window -t trading-system -n bot
tmux send-keys -t trading-system:bot "cd '$PROJECT_ROOT' && python3 main.py" Enter

# Window 0: Monitor
echo "📊 Creating monitor window..."
tmux send-keys -t trading-system:0 "clear && echo '✅ All systems starting.' && sleep 2" Enter

echo ""
echo "✅ Tmux session created: trading-system"
echo ""
echo "📖 Dashboard: http://localhost:3000"
echo "🌐 API: http://localhost:8000"
echo ""
echo "Attach to session: tmux attach -t trading-system"
echo "Kill session: tmux kill-session -t trading-system"
