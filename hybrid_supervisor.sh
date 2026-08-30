#!/bin/bash
# Keep-alive supervisor for the hybrid capital allocator — HARDENED
# (gap audit: G4 hung-daemon watchdog + G9 alerting).
#
# Restarts hybrid_allocator.py if it crashes AND if it HANGS (a wedged daemon
# holding a position would otherwise freeze its stops indefinitely while looking
# alive). Emits alerts on crash/hang; optionally runs $HYBRID_ALERT_CMD (e.g. a
# webhook/notifier) so a silent death doesn't leave money in an unmanaged trade.
#
# Stop:  touch logs/hybrid_supervisor.stop   OR   pkill -f hybrid_supervisor
cd "$(dirname "$0")" || exit 1
mkdir -p logs
LOG="logs/hybrid.log"
SUP="logs/hybrid_supervisor.log"
STOP="logs/hybrid_supervisor.stop"
STALL_MIN="${HYBRID_STALL_MIN:-25}"      # kill a daemon that hasn't logged this long (poll=15m)
# Defaults to ./hybrid_alert.sh when present. This script is bash and never
# sources .env, so HYBRID_ALERT_CMD set there would never arrive — which is
# why every crash alert went nowhere until 2026-08-30.
ALERT_CMD="${HYBRID_ALERT_CMD:-$([ -x ./hybrid_alert.sh ] && echo ./hybrid_alert.sh)}"

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') [HYBRID-SUP] $*" | tee -a "$SUP"; }
alert(){ log "🚨 ALERT: $*"; [ -n "$ALERT_CMD" ] && ( eval "$ALERT_CMD" "\"$*\"" >/dev/null 2>&1 & ); }
log_mtime(){ stat -f %m "$LOG" 2>/dev/null || stat -c %Y "$LOG" 2>/dev/null || echo 0; }

log "supervisor started (pid $$) stall_watchdog=${STALL_MIN}m"
rm -f "$STOP"
trap 'log "signal — exiting"; [ -n "$WD" ] && kill "$WD" 2>/dev/null; exit 0' INT TERM

while true; do
  [ -f "$STOP" ] && { log "stop flag present — exiting"; exit 0; }
  sz=$(du -m "$LOG" 2>/dev/null | cut -f1); [ "${sz:-0}" -ge 50 ] && : > "$LOG"
  log "launching hybrid_allocator.py"
  .venv/bin/python3 -u hybrid_allocator.py >> "$LOG" 2>&1 &
  DPID=$!

  # ── G4: hung-daemon watchdog — force-restart if the log goes stale ──
  ( while kill -0 "$DPID" 2>/dev/null; do
      sleep 60
      now=$(date +%s); mt=$(log_mtime)
      if [ "$((now - mt))" -ge "$((STALL_MIN * 60))" ]; then
        alert "hybrid_allocator HUNG (no log for ${STALL_MIN}m, pid $DPID) — force-restarting"
        kill -TERM "$DPID" 2>/dev/null; sleep 5; kill -9 "$DPID" 2>/dev/null
        break
      fi
    done ) &
  WD=$!

  wait "$DPID"; ec=$?
  kill "$WD" 2>/dev/null

  log "hybrid_allocator.py exited (code $ec)"
  [ -f "$STOP" ] && { log "stop flag present — not restarting"; exit 0; }
  [ "$ec" != "0" ] && [ "$ec" != "143" ] && alert "hybrid_allocator crashed (code $ec) — restarting in 15s"
  sleep 15
done
