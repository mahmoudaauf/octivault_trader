#!/bin/bash
# Keep-alive supervisor for the negative-funding-carry PAPER proof engine.
# Auto-restarts negative_carry_paper_trader.py if it ever exits, so the forward proof
# keeps accumulating unattended. Stop: touch logs/neg_carry_supervisor.stop  OR
# pkill -f neg_carry_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/neg_carry_paper.log"
SUP="logs/neg_carry_supervisor.log"
STOP="logs/neg_carry_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [NEG-CARRY-SUP] $*" | tee -a "$SUP"; }
log "supervisor started (pid $$)"
rm -f "$STOP"
while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching negative_carry_paper_trader.py"
  .venv/bin/python3 -u negative_carry_paper_trader.py >> "$LOG" 2>&1
  log "negative_carry_paper_trader.py exited (code $?)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 15
done
