#!/bin/bash
# Keep-alive supervisor for the delisting-exit PAPER proof engine.
# Auto-restarts delisting_exit_paper_trader.py if it ever exits, so the forward proof keeps
# accumulating unattended. Stop: touch logs/delisting_exit_supervisor.stop  OR  pkill -f delisting_exit_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/delisting_exit_paper.log"
SUP="logs/delisting_exit_supervisor.log"
STOP="logs/delisting_exit_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [DELIST-EXIT-SUP] $*" | tee -a "$SUP"; }
log "supervisor started (pid $$)"
rm -f "$STOP"
while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  # Rotate the paper log if it grows past 50MB.
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching delisting_exit_paper_trader.py"
  .venv/bin/python3 -u delisting_exit_paper_trader.py >> "$LOG" 2>&1
  log "delisting_exit_paper_trader.py exited (code $?)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 15
done
