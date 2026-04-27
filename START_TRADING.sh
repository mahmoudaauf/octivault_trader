#!/usr/bin/env bash
# =============================================================================
# OctiVault Autonomous 24h Trading System — Production Launcher
# Usage: bash START_TRADING.sh
# =============================================================================
set -euo pipefail

BOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$BOT_DIR/logs/octivault_master_orchestrator.log"
PID_FILE="$BOT_DIR/logs/trader.pid"
RESTART_LOG="$BOT_DIR/logs/restart_history.log"
MAX_RESTARTS=999          # effectively infinite for 24/7 operation
RESTART_DELAY=8           # seconds between crash → restart

# ── colours ──────────────────────────────────────────────────────────────────
GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'

mkdir -p "$BOT_DIR/logs" "$BOT_DIR/snapshots"

banner() {
  echo -e "${BLD}${CYN}"
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║        OctiVault  ·  Autonomous Wealth Engine  ·  24h       ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo -e "${RST}"
}

kill_existing() {
  if [[ -f "$PID_FILE" ]]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
      echo -e "${YLW}⚠  Stopping existing instance (PID $OLD_PID)...${RST}"
      kill "$OLD_PID" 2>/dev/null || true
      sleep 3
    fi
    rm -f "$PID_FILE"
  fi
  # Belt-and-suspenders: kill any stray python process running the MASTER
  pkill -f "MASTER_SYSTEM_ORCHESTRATOR" 2>/dev/null || true
  sleep 1
}

run_bot() {
  cd "$BOT_DIR"
  export APPROVE_LIVE_TRADING=YES
  export PYTHONUNBUFFERED=1

  python3 "🎯_MASTER_SYSTEM_ORCHESTRATOR.py" 2>&1 | \
    tee -a "$LOG_FILE"
}

# ── main auto-restart loop ────────────────────────────────────────────────────
banner
kill_existing

echo -e "${GRN}✅ Starting OctiVault Autonomous Trading System${RST}"
echo -e "   Log  : ${CYN}$LOG_FILE${RST}"
echo -e "   Monitor: ${CYN}python3 LIVE_MONITOR.py${RST}"
echo -e "   Stop : ${YLW}Ctrl+C${RST}"
echo ""

restarts=0
trap 'echo -e "\n${YLW}⏹  Shutdown requested. Stopping bot...${RST}"; kill_existing; exit 0' INT TERM

while true; do
  START_TS=$(date "+%Y-%m-%d %H:%M:%S")
  echo -e "${GRN}[${START_TS}] 🚀 Starting trading bot (attempt $((restarts+1)))${RST}" | tee -a "$RESTART_LOG"

  # Run in subshell; capture exit code without set -e killing us
  set +e
  run_bot
  EXIT_CODE=$?
  set -e

  END_TS=$(date "+%Y-%m-%d %H:%M:%S")
  echo -e "${YLW}[${END_TS}] ⚠  Bot exited (code=$EXIT_CODE). Restarting in ${RESTART_DELAY}s...${RST}" | tee -a "$RESTART_LOG"

  restarts=$((restarts + 1))
  if [[ $restarts -ge $MAX_RESTARTS ]]; then
    echo -e "${RED}❌ Max restarts ($MAX_RESTARTS) reached. Stopping.${RST}" | tee -a "$RESTART_LOG"
    exit 1
  fi

  sleep "$RESTART_DELAY"
done
