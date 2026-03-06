#!/bin/bash
# Quick restart and verify script for confidence fix

echo "🔧 Restarting bot with confidence fix..."
echo ""

# Kill existing bot
echo "1️⃣ Killing existing bot process..."
pkill -f "python.*trend_hunter\|python.*octivault_trader\|python.*run_bot" || true
sleep 2
echo "   ✅ Done"

# Verify process killed
if pgrep -f "python.*run_bot" > /dev/null; then
    echo "   ⚠️  Bot still running, force killing..."
    pkill -9 -f "python.*run_bot" || true
    sleep 2
fi

echo ""
echo "2️⃣ Waiting for clean shutdown..."
sleep 3

# Restart bot
echo "3️⃣ Starting bot with new code..."
echo ""

# Check if bot script exists
if [ ! -f "run_bot.py" ]; then
    echo "❌ run_bot.py not found in current directory"
    echo "   Current directory: $(pwd)"
    echo "   Please run from octivault_trader root"
    exit 1
fi

# Start bot in background
python3 run_bot.py --debug > /tmp/bot_restart.log 2>&1 &
BOT_PID=$!
echo "   Bot PID: $BOT_PID"

# Wait for startup
echo ""
echo "4️⃣ Waiting for bot startup (10 seconds)..."
sleep 10

# Check if bot is running
if ps -p $BOT_PID > /dev/null; then
    echo "   ✅ Bot process is running"
else
    echo "   ❌ Bot process died"
    echo "   Check startup log:"
    cat /tmp/bot_restart.log
    exit 1
fi

echo ""
echo "5️⃣ Checking for confidence signals..."
sleep 5

# Look for recent confidence logs
if [ -f "logs/bot.log" ]; then
    echo ""
    echo "Recent confidence signals:"
    echo "=========================="
    tail -30 logs/bot.log | grep "heuristic for" | tail -10
    
    # Check if any confidence values vary
    CONFS=$(tail -50 logs/bot.log | grep -oP "final=\K[\d.]+")
    if [ -n "$CONFS" ]; then
        UNIQUE=$(echo "$CONFS" | sort -u | wc -l)
        if [ "$UNIQUE" -gt 1 ]; then
            echo ""
            echo "✅ SUCCESS: Confidence values vary ($UNIQUE unique values)"
            echo "   Confidence range: $(echo "$CONFS" | sort -n | head -1) to $(echo "$CONFS" | sort -n | tail -1)"
        else
            echo ""
            echo "⚠️  WARNING: All confidence values the same"
            echo "   Check if bot generating signals..."
        fi
    else
        echo ""
        echo "ℹ️  No confidence signals yet in logs"
        echo "   Wait for next signal generation..."
    fi
else
    echo "   ⚠️  Log file not found at logs/bot.log"
fi

echo ""
echo "=========================================="
echo "Restart complete! Bot is running."
echo ""
echo "To monitor logs live:"
echo "  tail -f logs/bot.log | grep heuristic"
echo ""
echo "To verify fix is working:"
echo "  python3 verify_confidence_fix.py"
echo "=========================================="
