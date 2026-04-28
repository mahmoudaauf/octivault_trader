#!/bin/bash
# launch_run6.sh — clean launcher for run #6 with all 5 fixes active.
set -e
cd "$(dirname "$0")"

# Load env flags
set -a
source .env.run5
set +a

# Find orchestrator (handles emoji filename safely)
ORCH=$(ls *MASTER_SYSTEM_ORCHESTRATOR.py | head -1)
LOG=/tmp/octivault_live_run_6.log

echo "=== ENV ==="
env | grep -E "APPROVE_LIVE|STARTUP_TRIM|STRICT_CAP|INSUFF_BAL" | sort
echo "=== LAUNCHING $ORCH → $LOG ==="

nohup python3 "$ORCH" > "$LOG" 2>&1 &
PID=$!
echo "$PID" > /tmp/octivault_run6.pid
echo "PID=$PID"
sleep 3
ps -p $PID -o pid=,rss=,etime=,command= 2>/dev/null | head -1 || echo "❌ process died within 3s"
