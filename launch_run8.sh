#!/bin/bash
# === OCTIVAULT FREEZE BANNER ===
# STATUS:    QUARANTINED
# CANONICAL: main.py
# REASON:    Numbered debug runner — abandoned
# POLICY:    See STEP_4_MODULE_FREEZE.md — do not import from main.py / top-level scripts.
# ===============================

# launch_run8.sh — launcher for run #8 with self-healing trio active.
set -e
cd "$(dirname "$0")"

# Load env flags
set -a
source .env.run5
set +a

ORCH=$(ls *MASTER_SYSTEM_ORCHESTRATOR.py | head -1)
LOG=/tmp/octivault_live_run_8.log

echo "=== ENV ==="
env | grep -E "APPROVE_LIVE|STARTUP_TRIM|STRICT_CAP|INSUFF_BAL|TRUTH_AUDIT|MIN_HOLD|HEAL_" | sort
echo "=== LAUNCHING $ORCH → $LOG ==="

nohup python3 "$ORCH" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/octivault_run8.pid
echo "PID=$PID"
sleep 3
ps -p $PID -o pid=,rss=,etime=,command= 2>/dev/null | head -1 || echo "❌ process died within 3s"
