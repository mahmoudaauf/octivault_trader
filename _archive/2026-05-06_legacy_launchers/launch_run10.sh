#!/usr/bin/env bash
# === OCTIVAULT FREEZE BANNER ===
# STATUS:    QUARANTINED
# CANONICAL: main.py
# REASON:    Numbered debug runner — abandoned
# POLICY:    See STEP_4_MODULE_FREEZE.md — do not import from main.py / top-level scripts.
# ===============================

# Run #10 launcher — Heal-B/C now read LIVE get_nav_quote() (run-#9 fix)
set -euo pipefail
cd "$(dirname "$0")"

PIDFILE="/tmp/octivault_run10.pid"
LOGFILE="/tmp/octivault_live_run_10.log"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "ERROR: run #10 already running (pid=$(cat "$PIDFILE"))"
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
