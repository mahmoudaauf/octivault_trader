#!/bin/bash

# 6-Hour Trading Session with Checkpoints & Real-Time Monitoring
# Phase 2 Implementation Validation

set -e

WORKSPACE="/Users/mauf/Desktop/Octi AI Trading Bot/octivault_trader"
cd "$WORKSPACE"

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                            ║"
echo "║          🚀 6-HOUR TRADING SESSION - CHECKPOINTS & MONITORING 🚀          ║"
echo "║                                                                            ║"
echo "║             Phase 2 Implementation Validation                              ║"
echo "║             - Recovery Exit Min-Hold Bypass                                ║"
echo "║             - Forced Rotation MICRO Override                               ║"
echo "║             - Entry Sizing Alignment (25 USDT)                             ║"
echo "║                                                                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📋 PRE-FLIGHT CHECKS..."
echo ""

# Verify Phase 2 fixes are in place
echo "✅ Verifying Phase 2 fixes..."
python3 verify_fixes.py > /tmp/verify_output.txt 2>&1
if grep -q "16/16 CHECKS PASSED" /tmp/verify_output.txt; then
    echo "   ✅ All Phase 2 fixes verified (16/16)"
else
    echo "   ❌ Phase 2 fixes verification failed"
    cat /tmp/verify_output.txt
    exit 1
fi

echo ""
echo "✅ Checking configuration..."
python3 -c "
from core.config import *
print(f'   DEFAULT_PLANNED_QUOTE: {DEFAULT_PLANNED_QUOTE} USDT')
print(f'   MIN_ENTRY_USDT: {MIN_ENTRY_USDT} USDT')
print(f'   MIN_ENTRY_QUOTE_USDT: {MIN_ENTRY_QUOTE_USDT} USDT')
print(f'   SIGNIFICANT_POSITION_FLOOR: {SIGNIFICANT_POSITION_FLOOR} USDT')
" || echo "   ⚠️  Config check failed (non-critical)"

echo ""
echo "✅ Environment ready for 6-hour session"
echo ""

echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🎯 SESSION CONFIGURATION"
echo ""
echo "Duration:              6 hours (360 minutes)"
echo "Checkpoint Interval:   ~50 minutes"
echo "Monitoring:            Real-time Phase 2 indicators"
echo "Logging:               6hour_session_monitored.log"
echo "Reports:               JSON + Summary + Charts"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

echo "📊 STARTING TRADING SESSION..."
echo ""

# Start the monitored session
python3 RUN_6HOUR_SESSION_MONITORED.py &
SESSION_PID=$!

sleep 2

echo ""
echo "🔍 STARTING REAL-TIME LOG MONITOR (in new terminal)..."
echo ""
echo "Run this in another terminal:"
echo "   cd \"$WORKSPACE\""
echo "   python3 monitor_phase2_realtime.py 6hour_session_monitored.log"
echo ""

echo "📈 MONITORING LIVE LOGS..."
echo ""

# Tail logs with special formatting
(
    sleep 1
    while kill -0 $SESSION_PID 2>/dev/null; do
        if [ -f "6hour_session_monitored.log" ]; then
            # Show new lines with highlighting for Phase 2 events
            tail -f "6hour_session_monitored.log" | while IFS= read -r line; do
                if [[ "$line" =~ "Bypassing min-hold" ]]; then
                    echo "✅ $line"
                elif [[ "$line" =~ "MICRO restriction OVERRIDDEN" ]]; then
                    echo "✅ $line"
                elif [[ "$line" =~ "Entry:" ]] && [[ "$line" =~ "USDT" ]]; then
                    echo "💰 $line"
                elif [[ "$line" =~ "ERROR" ]] || [[ "$line" =~ "CRITICAL" ]]; then
                    echo "❌ $line"
                else
                    echo "$line"
                fi
            done
        fi
        sleep 1
    done
) &
TAIL_PID=$!

# Wait for session to complete
wait $SESSION_PID

# Stop tail
kill $TAIL_PID 2>/dev/null || true

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                      ✅ SESSION COMPLETED ✅                              ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

echo "📊 REPORTS GENERATED:"
echo ""

if [ -f "6hour_session_report_monitored.json" ]; then
    echo "✅ 6hour_session_report_monitored.json"
    echo "   Size: $(wc -c < 6hour_session_report_monitored.json) bytes"
fi

if [ -f "6hour_session_checkpoint_summary.txt" ]; then
    echo "✅ 6hour_session_checkpoint_summary.txt"
    echo ""
    echo "📝 CHECKPOINT SUMMARY:"
    cat "6hour_session_checkpoint_summary.txt"
fi

if [ -f "6hour_session_monitored.log" ]; then
    echo ""
    echo "✅ 6hour_session_monitored.log"
    echo "   Size: $(wc -l < 6hour_session_monitored.log) lines"
    
    echo ""
    echo "🔍 PHASE 2 INDICATORS IN LOGS:"
    
    RECOVERY_COUNT=$(grep -c "Bypassing min-hold" "6hour_session_monitored.log" || echo 0)
    echo "   Recovery Bypasses: $RECOVERY_COUNT"
    
    ROTATION_COUNT=$(grep -c "MICRO restriction OVERRIDDEN" "6hour_session_monitored.log" || echo 0)
    echo "   Forced Rotations: $ROTATION_COUNT"
    
    ENTRY_COUNT=$(grep -c "Entry:" "6hour_session_monitored.log" || echo 0)
    echo "   Total Entries: $ENTRY_COUNT"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📈 NEXT STEPS:"
echo ""
echo "1. Review checkpoint summary:"
echo "   cat 6hour_session_checkpoint_summary.txt"
echo ""
echo "2. Analyze detailed report:"
echo "   python3 -c \"import json; print(json.dumps(json.load(open('6hour_session_report_monitored.json')), indent=2))\""
echo ""
echo "3. Check Phase 2 effectiveness:"
echo "   grep 'Bypassing min-hold\\|MICRO restriction' 6hour_session_monitored.log"
echo ""
echo "4. If successful, deploy to production"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
