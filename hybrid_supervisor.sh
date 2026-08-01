#!/bin/bash
# Keep-alive supervisor for the hybrid capital allocator (safe yield core +
# capped auto-speculative satellite). Auto-restarts hybrid_allocator.py if it
# ever exits. Stop: touch logs/hybrid_supervisor.stop  OR  pkill -f hybrid_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/hybrid.log"
SUP="logs/hybrid_supervisor.log"
STOP="logs/hybrid_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [HYBRID-SUP] $*" | tee -a "$SUP"; }
log "supervisor started (pid $$)"
rm -f "$STOP"
while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching hybrid_allocator.py"
  .venv/bin/python3 -u hybrid_allocator.py >> "$LOG" 2>&1
  log "hybrid_allocator.py exited (code $?)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 15
done
