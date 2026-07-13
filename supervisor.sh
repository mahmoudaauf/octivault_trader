#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Octivault autonomous supervisor.
#
# Keeps main.py running unattended: restarts it if it ever exits (crash, OOM,
# unhandled exception), with a crash-loop guard so a hard failure doesn't hammer
# the exchange. Rotates run_latest.log in place so the disk can't fill. main.py's
# own PID lock prevents duplicate instances, so this is the single entry point.
#
# Usage:   nohup ./supervisor.sh >> logs/supervisor.out 2>&1 &
# Stop:    touch logs/supervisor.stop      (graceful)  OR  pkill -f supervisor.sh
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")" || exit 1

RUN_LOG="logs/run_latest.log"
SUP_LOG="logs/supervisor.log"
STOP_FLAG="logs/supervisor.stop"
# NOTE: run_latest.log rotation is now handled IN-PROCESS via RotatingFileHandler
# (50 MB × 3 backups = 200 MB max). The variables below are kept as a fallback
# safety net only — the rotator loop still runs but will rarely trigger.
MAX_LOG_MB=150            # safety-net threshold (in-process rotates at 50 MB)
KEEP_ARCHIVES=3           # supervisor gz archives to keep as emergency backup
RESTART_BACKOFF=10        # seconds between normal restarts
FAST_WINDOW=300          # crash-loop detection window (s)
MAX_FAST_RESTARTS=5      # this many restarts inside the window → long back-off
LONG_BACKOFF=300         # cool-off when a crash loop is detected
STALL_TIMEOUT=180        # force-restart if no NEW trading cycle for this long (s)
WATCHDOG_INTERVAL=30     # how often the watchdog checks cycle progress (s)
TERM_GRACE=5             # seconds to wait after SIGTERM before SIGKILL

mkdir -p logs
log_sup() { echo "$(date '+%Y-%m-%d %H:%M:%S') [SUPERVISOR] $*" | tee -a "$SUP_LOG"; }

rotate_log() {
  [ -f "$RUN_LOG" ] || return 0
  local sz; sz=$(du -m "$RUN_LOG" 2>/dev/null | cut -f1)
  if [ "${sz:-0}" -ge "$MAX_LOG_MB" ]; then
    # Archive a gzipped tail, then TRUNCATE IN PLACE so the running process keeps
    # writing to the same file descriptor (mv/rename would orphan its handle).
    tail -c 20000000 "$RUN_LOG" 2>/dev/null | gzip > "logs/run_$(date +%Y%m%d_%H%M%S).log.gz" 2>/dev/null
    : > "$RUN_LOG"
    log_sup "rotated run log (was ${sz}MB → kept gzipped tail)"
    # shellcheck disable=SC2012
    ls -t logs/run_2*.log.gz 2>/dev/null | tail -n +$((KEEP_ARCHIVES + 1)) | xargs rm -f 2>/dev/null
  fi
}

# Background rotation loop (runs while main.py runs, since the foreground blocks).
( while true; do sleep 600; rotate_log; done ) &
ROTATOR_PID=$!

# Cycle-staleness watchdog. main.py can wedge "alive but not trading" (e.g. a
# WebSocket reconnect storm starving the event loop). It then ignores SIGTERM and the
# supervisor — which only relaunches on a clean *exit* — never recovers it (this is
# exactly how the bot was silently down ~30m). This external watchdog detects no
# trading-cycle progress for STALL_TIMEOUT seconds and force-restarts main.py
# (SIGTERM, then SIGKILL after a grace period). It keys on the run log's CYCLE counter,
# not the file mtime — the log keeps updating with reconnect spam during a hang.
(
  wd_last_cycle=""; wd_last_change=$(date +%s)
  while true; do
    sleep "$WATCHDOG_INTERVAL"
    wd_cur=$(tail -c 100000 "$RUN_LOG" 2>/dev/null | grep -oE "cycle [0-9]+" | tail -1)
    wd_now=$(date +%s)
    if [ -n "$wd_cur" ] && [ "$wd_cur" != "$wd_last_cycle" ]; then
      wd_last_cycle="$wd_cur"; wd_last_change=$wd_now
      continue
    fi
    if [ $((wd_now - wd_last_change)) -ge "$STALL_TIMEOUT" ]; then
      wd_pid=$(pgrep -f "[Pp]ython.* main.py" | head -1)
      if [ -n "$wd_pid" ]; then
        log_sup "WATCHDOG: no new cycle for ${STALL_TIMEOUT}s — restarting hung main.py (pid $wd_pid)"
        kill -TERM "$wd_pid" 2>/dev/null
        sleep "$TERM_GRACE"
        kill -9 "$wd_pid" 2>/dev/null   # escalate: hung process ignores SIGTERM
      fi
      wd_last_change=$wd_now            # give the relaunch time before re-checking
    fi
  done
) &
WATCHDOG_PID=$!

cleanup() { kill "$ROTATOR_PID" "$WATCHDOG_PID" 2>/dev/null; log_sup "supervisor stopped"; exit 0; }
trap cleanup INT TERM

log_sup "supervisor started (pid $$)"
rm -f "$STOP_FLAG"
fast=0; window_start=$(date +%s)

while true; do
  [ -f "$STOP_FLAG" ] && { log_sup "stop flag present — exiting"; cleanup; }
  rotate_log
  log_sup "launching main.py"
  # Use venv python so all dependencies are available.
  # main.py owns its own RotatingFileHandler → logs/run_latest.log (INFO+).
  # We redirect stderr here as a safety net for uncaught exceptions that bypass
  # the logging system (e.g. segfaults, early-boot errors before loggers init).
  .venv/bin/python3 main.py 2>>"$RUN_LOG"
  ec=$?
  log_sup "main.py exited (code $ec)"
  [ -f "$STOP_FLAG" ] && { log_sup "stop flag present — not restarting"; cleanup; }

  now=$(date +%s)
  [ $((now - window_start)) -gt $FAST_WINDOW ] && { fast=0; window_start=$now; }
  fast=$((fast + 1))
  if [ "$fast" -ge "$MAX_FAST_RESTARTS" ]; then
    log_sup "CRASH LOOP: $fast restarts in <${FAST_WINDOW}s — cooling off ${LONG_BACKOFF}s"
    sleep "$LONG_BACKOFF"
    fast=0; window_start=$(date +%s)
  else
    sleep "$RESTART_BACKOFF"
  fi
done
