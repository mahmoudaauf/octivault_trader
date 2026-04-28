#!/usr/bin/env bash
# Run #11 launcher — NAV double-count fix + Heal-C lift (run-#10 hardening)
set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="/tmp/octivault_run11.pid"
LOGFILE="/tmp/octivault_live_run_11.log"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: run #11 already running (pid=$(cat "$PIDFILE"))"
  exit 1
fi

set -a
source .env.run5
set +a

env | grep -E "^(HEAL_|MIN_HOLD_|STRICT_CAP|STARTUP_TRIM|TRUTH_AUDIT)" | sort

echo "=== LAUNCHING 🎯_MASTER_SYSTEM_ORCHESTRATOR.py → ${LOGFILE} ==="
nohup python3 "🎯_MASTER_SYSTEM_ORCHESTRATOR.py" > "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "PID=$PID"
sleep 3
ps -p "$PID" -o pid,etime,command | head -2
