#!/bin/bash
# Keep-alive supervisor for the PAPER market-maker (spread_mm_paper.py).
# No money is at risk here — this only protects the continuity of the
# measurement, since gaps bias the fill statistics.
# Stop: touch logs/spread_mm_supervisor.stop  OR  pkill -f spread_mm_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/spread_mm.log"
SUP="logs/spread_mm_supervisor.log"
STOP="logs/spread_mm_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [MM-SUP] $*" | tee -a "$SUP"; }
# Wired by ./hybrid_alert.sh being present + executable — this is bash and
# never sources .env, which is why HYBRID_ALERT_CMD there reached nothing.
ALERT_CMD="${ALERT_CMD:-$([ -x ./hybrid_alert.sh ] && echo ./hybrid_alert.sh)}"
alert(){ log "🚨 ALERT: $*"; [ -n "$ALERT_CMD" ] && ( eval "$ALERT_CMD" "\"$*\"" >/dev/null 2>&1 & ); }
log "supervisor started (pid $$)"
on_signal(){
  if [ -f "$STOP" ]; then log "signal — stop flag present, exiting for good"; exit 0; fi
  log "signal — exiting (no stop flag; paper market-maker should be restarted)"
  alert "paper market-maker supervisor killed by signal with no stop flag — not running"
  exit 75
}
trap on_signal INT TERM

while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching spread_mm_paper.py"
  .venv/bin/python3 -u spread_mm_paper.py >> "$LOG" 2>&1
  log "spread_mm_paper.py exited (code $?)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 15
done
