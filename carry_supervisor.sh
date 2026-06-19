#!/bin/bash
# Keep-alive supervisor for the funding-carry PAPER proof engine.
# Auto-restarts carry_paper_trader.py if it ever exits, so the forward proof keeps
# accumulating unattended. Stop: touch logs/carry_supervisor.stop  OR  pkill -f carry_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/carry_paper.log"
SUP="logs/carry_supervisor.log"
STOP="logs/carry_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [CARRY-SUP] $*" | tee -a "$SUP"; }
log "supervisor started (pid $$)"
rm -f "$STOP"
while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  # Rotate the carry log if it grows past 50MB.
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching carry_paper_trader.py"
  python3 -u carry_paper_trader.py >> "$LOG" 2>&1
  log "carry_paper_trader.py exited (code $?)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 15
done
