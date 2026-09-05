#!/bin/bash
# Keep-alive supervisor for the delta-neutral staking-yield PAPER trader.
# No money is at risk here — this only protects the continuity of the
# measurement, since gaps bias the fill statistics.
# Stop: touch logs/halal_supervisor.stop  OR  pkill -f halal_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/halal.log"
SUP="logs/halal_supervisor.log"
STOP="logs/halal_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [HALAL-SUP] $*" | tee -a "$SUP"; }
# Wired by ./hybrid_alert.sh being present + executable — this is bash and
# never sources .env, which is why HYBRID_ALERT_CMD there reached nothing.
ALERT_CMD="${ALERT_CMD:-$([ -x ./hybrid_alert.sh ] && echo ./hybrid_alert.sh)}"
alert(){ log "🚨 ALERT: $*"; [ -n "$ALERT_CMD" ] && ( eval "$ALERT_CMD" "\"$*\"" >/dev/null 2>&1 & ); }
# Synchronous variant for the EXIT path. alert() backgrounds the command so a
# crash-restart is never blocked, but on exit the shell dies immediately and
# takes the background job with it — so the alert for a silently-killed
# supervisor was itself silently lost. Verified 2026-08-31.
alert_sync(){ log "🚨 ALERT: $*"; [ -n "$ALERT_CMD" ] && eval "$ALERT_CMD" "\"$*\"" >/dev/null 2>&1; }
log "supervisor started (pid $$)"
on_signal(){
  if [ -f "$STOP" ]; then log "signal — stop flag present, exiting for good"; exit 0; fi
  log "signal — exiting (no stop flag; halal allocator should be restarted)"
  alert_sync "halal allocator supervisor killed by signal with no stop flag — not running"
  exit 75
}
trap on_signal INT TERM

while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching halal_allocator.py"
  .venv/bin/python3 -u halal_allocator.py >> "$LOG" 2>&1
  log "halal_allocator.py exited (code $?)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 15
done
