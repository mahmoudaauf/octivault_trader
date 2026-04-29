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
# Portable detach: python's os.setsid() creates a new session, fully isolating from tty.
# nohup ignores SIGHUP. </dev/null prevents stdin tty issues. disown removes from job table.
# PYTHONFAULTHANDLER=1 → on segfault/SIGTERM/SIGKILL inheritor signals dumps stack to stderr.
# PYTHONUNBUFFERED=1 → real-time log flush so we never lose the last lines on crash.
nohup env PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 python3 -u -X faulthandler -c "import os, sys, faulthandler; faulthandler.enable(); os.setsid(); os.execvpe('python3', ['python3', '-u', '-X', 'faulthandler', '🎯_MASTER_SYSTEM_ORCHESTRATOR.py'], os.environ)" </dev/null > "$LOGFILE" 2>&1 &
PID=$!
disown "$PID" 2>/dev/null || true
echo "$PID" > "$PIDFILE"
echo "PID=$PID"
sleep 3
ps -p "$PID" -o pid,ppid,etime,command | head -2
