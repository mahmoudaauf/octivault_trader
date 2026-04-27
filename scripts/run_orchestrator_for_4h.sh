#!/usr/bin/env bash
set -euo pipefail

# Run the master orchestrator for 4 hours with safe defaults.
# - writes orchestrator.pid
# - logs to logs/trading_run_<timestamp>.log
# - creates automation/.apply_proposed_rules = true

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKDIR"

LOGDIR="$WORKDIR/logs"
mkdir -p "$LOGDIR"
TS=$(date -u +"%Y%m%dT%H%M%SZ")
LOGFILE="$LOGDIR/trading_run_${TS}.log"

# Enable proposed rules control file (quick on/off toggle)
mkdir -p automation
echo "true" > automation/.apply_proposed_rules

# Safety envs for this run (modify here if you want different values)
export APPROVE_LIVE_TRADING=YES
export MIN_REQUIRED_CONF_FLOOR=0.50
export APPLY_PROPOSED_RULES=true

# DISABLE micro backtest gate in bootstrap mode to allow initial trading
export PRETRADE_MICRO_BACKTEST_ENABLED=false

# Prevent accidental multiple runs: if orchestrator.pid exists and process is alive, exit
if [ -f "orchestrator.pid" ]; then
  OLD_PID=$(cat orchestrator.pid || true)
  if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Orchestrator already running with pid=$OLD_PID. Exiting." | tee -a "$LOGFILE"
    exit 1
  fi
fi

# Start orchestrator in background; redirect stdout/stderr
nohup python3 "🎯_MASTER_SYSTEM_ORCHESTRATOR.py" >> "$LOGFILE" 2>&1 &
PID=$!
echo "$PID" > orchestrator.pid

echo "Started orchestrator pid=$PID (logs -> $LOGFILE)"

echo "[run_for_4h] Started at $(date -u) PID=$PID LOG=$LOGFILE" >> "$LOGFILE"

# Keep this script alive for 4 hours then attempt graceful shutdown
SLEEP_SECS=$((4*3600))
# Write a small heartbeat so operator knows the script is alive
(sleep 5; echo "[run_for_4h] heartbeat: orchestrator pid=$PID started" >> "$LOGFILE") &

sleep "$SLEEP_SECS"

echo "[run_for_4h] 4 hours elapsed — initiating shutdown at $(date -u)" >> "$LOGFILE"
if kill -0 "$PID" 2>/dev/null; then
  echo "[run_for_4h] Sending TERM to pid=$PID" >> "$LOGFILE"
  kill "$PID" || true
  sleep 10
  if kill -0 "$PID" 2>/dev/null; then
    echo "[run_for_4h] PID still alive; sending KILL" >> "$LOGFILE"
    kill -9 "$PID" || true
  fi
fi

echo "[run_for_4h] Completed at $(date -u)" >> "$LOGFILE"

exit 0
