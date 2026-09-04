#!/bin/bash
# Keep-alive supervisor for the IBKR allocator (ibkr_allocator.py, no args).
# READ-ONLY daemon: fetches a Flex statement, records NAV, alerts on drift.
# It never places an order and holds no credential that could place one, so a
# crash here risks a stale ledger, never money.
# Stop: touch logs/ibkr_supervisor.stop  OR  pkill -f ibkr_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/ibkr.log"
SUP="logs/ibkr_supervisor.log"
STOP="logs/ibkr_supervisor.stop"
log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [IBKR-SUP] $*" | tee -a "$SUP"; }
ALERT_CMD="${ALERT_CMD:-$([ -x ./hybrid_alert.sh ] && echo ./hybrid_alert.sh)}"
alert(){ log "🚨 ALERT: $*"; [ -n "$ALERT_CMD" ] && ( eval "$ALERT_CMD" "\"$*\"" >/dev/null 2>&1 & ); }
alert_sync(){ log "🚨 ALERT: $*"; [ -n "$ALERT_CMD" ] && eval "$ALERT_CMD" "\"$*\"" >/dev/null 2>&1; }
log "supervisor started (pid $$)"
on_signal(){
  if [ -f "$STOP" ]; then log "signal — stop flag present, exiting for good"; exit 0; fi
  log "signal — exiting (no stop flag; ibkr allocator should be restarted)"
  alert_sync "IBKR allocator supervisor killed by signal with no stop flag — not running"
  exit 75
}
trap on_signal INT TERM

while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching ibkr_allocator.py"
  venv/bin/python3 -u ibkr_allocator.py >> "$LOG" 2>&1
  code=$?
  log "ibkr_allocator.py exited (code $code)"
  [ "$code" -ne 0 ] && alert "IBKR allocator daemon exited with code $code — statement refresh may be stalled"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  sleep 30
done
